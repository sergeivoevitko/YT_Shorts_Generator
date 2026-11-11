"""Free and local AI providers (anything with an OpenAI-compatible chat API).

Claude stays in analyze.py / story.py through the Anthropic SDK. This module covers
every other provider with one small HTTP client: Gemini, Groq, Cerebras, OpenRouter,
Mistral, and local servers such as Ollama and LM Studio.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Type

import httpx
from pydantic import BaseModel, ValidationError

from .. import config

# chunk = how many transcript characters to send per request (free tiers and local
# models have small limits; a long video is analyzed in several requests).
PROVIDERS: dict[str, dict] = {
    "claude": {"name": "Claude (paid, best quality)", "paid": True, "key_url": "https://console.anthropic.com/settings/keys"},
    "gemini": {"name": "Google Gemini (free tier)", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-3.5-flash", "chunk": 300_000, "key_url": "https://aistudio.google.com/apikey"},
    "groq": {"name": "Groq (free tier, very fast)", "base_url": "https://api.groq.com/openai/v1",
             "model": "openai/gpt-oss-120b", "chunk": 16_000, "key_url": "https://console.groq.com/keys"},
    "cerebras": {"name": "Cerebras (free tier, very fast)", "base_url": "https://api.cerebras.ai/v1",
                 "model": "gpt-oss-120b", "chunk": 60_000, "key_url": "https://cloud.cerebras.ai"},
    "openrouter": {"name": "OpenRouter (free models)", "base_url": "https://openrouter.ai/api/v1",
                   "model": "meta-llama/llama-3.3-70b-instruct:free", "chunk": 60_000, "key_url": "https://openrouter.ai/keys"},
    "mistral": {"name": "Mistral (free tier)", "base_url": "https://api.mistral.ai/v1",
                "model": "mistral-small-latest", "chunk": 80_000, "key_url": "https://console.mistral.ai/api-keys"},
    "ollama": {"name": "Ollama (runs on this PC, free)", "base_url": "http://localhost:11434/v1",
               "model": "llama3.1", "chunk": 7_000, "local": True, "key_url": "https://ollama.com/download"},
    "lmstudio": {"name": "LM Studio (runs on this PC, free)", "base_url": "http://localhost:1234/v1",
                 "model": "", "chunk": 5_000, "local": True, "key_url": "https://lmstudio.ai"},
    "custom": {"name": "Other (any OpenAI-compatible API)", "base_url": "", "model": "", "chunk": 30_000, "key_url": ""},
}


def current() -> dict:
    """The provider chosen in AI settings, with its saved key, model and URL."""
    saved = config.load_settings()
    key = saved.get("provider") or "claude"
    if key not in PROVIDERS:
        key = "claude"
    preset = PROVIDERS[key]
    if key == "claude":
        return {"provider": key, "name": "Claude", "model": config.get_model(), "api_key": config.get_api_key(),
                "base_url": "", "ready": bool(config.get_api_key()), "local": False, "chunk": 0}
    per = (saved.get("providers") or {}).get(key, {})
    base = (per.get("base_url") or preset["base_url"]).rstrip("/")
    model = per.get("model") or preset["model"]
    if key == "gemini" and model in ("gemini-2.5-flash", "gemini-2.5-pro"):
        model = preset["model"]  # retired by Google; saved by an earlier version of this app
    api_key = (per.get("api_key") or "").strip()
    local = bool(preset.get("local")) or base.startswith(("http://localhost", "http://127.0.0.1"))
    return {"provider": key, "name": preset["name"].split(" (")[0], "model": model, "api_key": api_key,
            "base_url": base, "ready": bool(base and model and (api_key or local)), "local": local,
            "chunk": int(preset["chunk"])}


def ready() -> bool:
    return current()["ready"]


def _headers(cfg: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    return headers


def _explain(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, list) and err:
            err = err[0].get("error", err[0]) if isinstance(err[0], dict) else err[0]
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = resp.text
    return f"{resp.status_code}: {str(msg)[:300]}"


def list_models(cfg: dict | None = None) -> list[str]:
    cfg = cfg or current()
    resp = httpx.get(cfg["base_url"] + "/models", headers=_headers(cfg), timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(_explain(resp))
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else data
    names = [str(m.get("id") or m.get("name") or "") for m in items or [] if isinstance(m, dict)]
    return sorted(n.removeprefix("models/") for n in names if n)


def _retry_delay(resp: httpx.Response) -> float | None:
    """Seconds the provider asks us to wait, from the header or the error body."""
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        pass
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', resp.text) or re.search(r"retry in ([\d.]+)\s*s", resp.text, re.I)
    return float(m.group(1)) if m else None


def _scrub(text: str, cfg: dict) -> str:
    return text.replace(cfg["api_key"], "***") if cfg.get("api_key") else text


def _ollama_chat(system: str, user: str, cfg: dict, want_json: bool, max_tokens: int, schema: dict | None,
                 say: Callable[[str], None], read_timeout: float) -> str:
    """Ollama through its own API instead of the OpenAI-compatible one. Only the native
    API lets the request set the context window; with the default (2,048 or 4,096 tokens)
    Ollama silently cuts long prompts from the front, which removes the instructions. It
    also accepts a JSON schema that the answer is forced to follow."""
    host = re.sub(r"/v1/?$", "", cfg["base_url"])
    out_tokens = min(max_tokens, 3000)
    need = int((len(system) + len(user)) / 3.0) + out_tokens + 256
    num_ctx = next(n for n in (4096, 8192, 16384, 32768, 65536) if n >= need or n == 65536)
    payload = {"model": cfg["model"], "stream": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "options": {"num_ctx": max(num_ctx, 8192), "num_predict": out_tokens, "temperature": 0.4}}
    if schema is not None:
        payload["format"] = schema
    elif want_json:
        payload["format"] = "json"
    url = host + "/api/chat"
    started = time.time()
    config.applog(f"[ai] request to Ollama model={cfg['model']} url={url} prompt_chars={len(system) + len(user)} "
                  f"num_ctx={payload['options']['num_ctx']} schema={'yes' if schema is not None else 'no'}")
    try:
        resp = httpx.post(url, json=payload, timeout=httpx.Timeout(read_timeout, connect=15.0))
    except httpx.ConnectError as exc:
        say(f"could not connect to {url}: {exc}")
        raise RuntimeError(f"Could not connect to Ollama at {host}. Is Ollama running on this PC?") from exc
    except httpx.TimeoutException as exc:
        say(f"no answer from Ollama model {cfg['model']} within {read_timeout:.0f}s")
        raise RuntimeError(f"Ollama did not answer within {read_timeout:.0f} seconds with model '{cfg['model']}'. "
                           "The model may be too large for this PC; try a smaller one.") from exc
    config.applog(f"[ai] answer status={resp.status_code} in {time.time() - started:.1f}s body={resp.text[:300]!r}")
    if resp.status_code != 200:
        detail = _explain(resp)
        say(f"Ollama error: {detail}")
        if resp.status_code == 404:
            raise RuntimeError(f"Ollama does not have the model '{cfg['model']}'. Download it first "
                               f"(ollama pull {cfg['model']}) or pick another model in AI settings.")
        raise RuntimeError(f"Ollama returned an error ({detail}).")
    data = resp.json()
    seen = int(data.get("prompt_eval_count") or 0)
    if seen and seen < (len(system) + len(user)) / 6.5:
        say(f"Ollama read only {seen} tokens of a {len(system) + len(user)} character request; "
            "the model's context window is too small, so part of the request was cut off.")
    return (data.get("message") or {}).get("content") or ""


def chat(system: str, user: str, cfg: dict | None = None, want_json: bool = True, max_tokens: int = 8000,
         log: Callable[[str], None] | None = None, retries: int = 3, read_timeout: float | None = None,
         schema: dict | None = None) -> str:
    """One chat completion. Waits and retries when the free tier says to slow down."""
    cfg = cfg or current()
    say = lambda msg: (config.applog(_scrub(f"[ai] {msg}", cfg)), log and log(msg))
    if cfg["provider"] == "ollama":
        return _ollama_chat(system, user, cfg, want_json, max_tokens, schema, say, read_timeout or 900.0)
    payload = {"model": cfg["model"], "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if want_json:
        payload["response_format"] = {"type": "json_object"}
    read_timeout = read_timeout or (900.0 if cfg["local"] else 150.0)
    timeout = httpx.Timeout(read_timeout, connect=15.0)
    url = cfg["base_url"] + "/chat/completions"
    last = ""
    for attempt in range(retries + 1):
        started = time.time()
        config.applog(f"[ai] request to {cfg['name']} model={cfg['model']} url={url} "
                      f"prompt_chars={len(system) + len(user)} attempt={attempt + 1}")
        try:
            resp = httpx.post(url, headers=_headers(cfg), json=payload, timeout=timeout)
        except httpx.ConnectError as exc:
            hint = " Is it running on this PC?" if cfg["local"] else " Check your internet connection or firewall."
            say(f"could not connect to {url}: {exc}")
            raise RuntimeError(f"Could not connect to {cfg['name']} at {cfg['base_url']}.{hint}") from exc
        except httpx.TimeoutException as exc:
            say(f"no answer from {cfg['name']} model {cfg['model']} within {read_timeout:.0f}s")
            raise RuntimeError(f"{cfg['name']} did not answer within {read_timeout:.0f} seconds with model "
                               f"'{cfg['model']}'. That model may be overloaded or not offered on the free tier. "
                               "Choose another model in AI settings.") from exc
        except httpx.HTTPError as exc:
            say(f"network error: {type(exc).__name__}: {exc}")
            raise RuntimeError(f"Network problem while talking to {cfg['name']}: {exc}") from exc
        config.applog(_scrub(f"[ai] answer status={resp.status_code} in {time.time() - started:.1f}s "
                             f"body={resp.text[:300]!r}", cfg))
        if resp.status_code == 200:
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"{cfg['name']} sent an answer in an unexpected format.") from exc
        last = _explain(resp)
        if resp.status_code == 400 and "response_format" in payload:
            payload.pop("response_format")  # not every server supports JSON mode; the prompt asks for JSON anyway
            continue
        if resp.status_code in (429, 500, 502, 503, 529):
            asked = _retry_delay(resp)
            if resp.status_code == 429 and asked is not None and asked > 90:
                say(f"quota used up for model {cfg['model']} (retry in {asked:.0f}s)")
                raise RuntimeError(f"The free quota for model '{cfg['model']}' at {cfg['name']} is used up "
                                   f"(resets in about {asked / 3600:.1f} hours). Choose another model in AI settings.")
            if attempt < retries:
                wait = min(max(asked if asked is not None else 8.0 * (attempt + 1), 2.0), 65.0)
                say(f"{cfg['name']} asked to slow down ({resp.status_code}); waiting {wait:.0f}s.")
                time.sleep(wait)
                continue
        break
    say(f"{cfg['name']} error: {last}")
    raise RuntimeError(f"{cfg['name']} returned an error ({last}).")


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object in the answer")


def chat_json(system: str, user: str, schema: Type[BaseModel], shape: str,
              log: Callable[[str], None] | None = None, max_tokens: int = 8000) -> BaseModel:
    """Chat completion parsed and validated against a pydantic model, with one repair attempt."""
    cfg = current()
    system = (system + "\n\nAnswer with one JSON object and nothing else, no markdown fences and no commentary. "
              "Use exactly this shape:\n" + shape)
    prompt = user
    error = ""
    for _ in range(2):
        text = chat(system, prompt, cfg, True, max_tokens, log, schema=schema.model_json_schema())
        try:
            return schema.model_validate(_extract_json(text))
        except (ValueError, ValidationError) as exc:
            error = str(exc)[:400]
            prompt = (user + "\n\nYour previous answer could not be used because of this problem:\n" + error +
                      "\nAnswer again with valid JSON in exactly the requested shape.")
    raise RuntimeError(f"{cfg['name']} did not return usable JSON ({error}).")


def test() -> str:
    cfg = current()
    if not cfg["ready"]:
        raise RuntimeError("Fill in the missing settings first (key or model).")
    text = chat("You are a connection test.", 'Reply with this exact JSON: {"ok": true}', cfg, True, 300,
                retries=0, read_timeout=30.0)
    return f"{cfg['name']} answered with model {cfg['model']}: {text.strip()[:80]}"
