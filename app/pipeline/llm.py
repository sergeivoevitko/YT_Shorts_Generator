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
    "github": {"name": "GitHub Models (free with a GitHub account)", "base_url": "https://models.github.ai/inference",
               "model": "openai/gpt-4.1-mini", "chunk": 20_000, "key_url": "https://github.com/settings/personal-access-tokens"},
    "sambanova": {"name": "SambaNova (free tier, fast)", "base_url": "https://api.sambanova.ai/v1",
                  "model": "Meta-Llama-3.3-70B-Instruct", "chunk": 40_000, "key_url": "https://cloud.sambanova.ai/apis"},
    "nvidia": {"name": "NVIDIA NIM (free developer access)", "base_url": "https://integrate.api.nvidia.com/v1",
               "model": "meta/llama-3.3-70b-instruct", "chunk": 60_000, "key_url": "https://build.nvidia.com"},
    "zai": {"name": "Z.ai GLM (free flash models)", "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-4.5-flash", "chunk": 60_000, "key_url": "https://z.ai/manage-apikey/apikey-list"},
    "huggingface": {"name": "Hugging Face (small free allowance)", "base_url": "https://router.huggingface.co/v1",
                    "model": "meta-llama/Llama-3.3-70B-Instruct", "chunk": 30_000, "key_url": "https://huggingface.co/settings/tokens"},
    "cohere": {"name": "Cohere (free trial key)", "base_url": "https://api.cohere.ai/compatibility/v1",
               "model": "command-a-03-2025", "chunk": 60_000, "key_url": "https://dashboard.cohere.com/api-keys"},
    "pollinations": {"name": "Pollinations (free, no key needed)", "base_url": "https://text.pollinations.ai/openai",
                     "model": "openai", "chunk": 12_000, "keyless": True, "key_url": ""},
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
    keyless = bool(preset.get("keyless"))
    return {"provider": key, "name": preset["name"].split(" (")[0], "model": model, "api_key": api_key,
            "base_url": base, "ready": bool(base and model and (api_key or local or keyless)), "local": local,
            "keyless": keyless,
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


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------

_NOT_GENERAL = ("embed", "coder", "code", "bge", "minilm", "rerank", "guard", "whisper", "llava", "vision", "moondream", "ocr")
_FAMILY_RANK = ("qwen3", "qwen2.5", "llama3.3", "llama3.2", "llama3.1", "gemma3", "gemma2", "mistral-nemo", "mistral",
                "phi4", "phi3", "deepseek-r1", "gpt-oss", "llama", "qwen", "gemma")
_ollama_cache: dict = {"at": 0.0, "data": None}


def _ollama_exe() -> str:
    import os
    import shutil

    found = shutil.which("ollama")
    if found:
        return found
    for path in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
                 "/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
                 "/Applications/Ollama.app/Contents/Resources/ollama"):
        if path and os.path.exists(path):
            return path
    return ""


def _params(model: dict) -> float:
    """Model size in billions of parameters, from Ollama's details or the tag name."""
    text = str((model.get("details") or {}).get("parameter_size") or "")
    m = re.search(r"([\d.]+)\s*([BM])", text, re.I) or re.search(r"[:\-]([\d.]+)([bm])\b", str(model.get("name", "")), re.I)
    if not m:
        return 0.0
    return float(m.group(1)) * (1.0 if m.group(2).upper() == "B" else 0.001)


def _rank_ollama(model: dict) -> float:
    """Higher is better for picking moments and writing short text on a normal PC."""
    name = str(model.get("name", "")).lower()
    if any(x in name for x in _NOT_GENERAL):
        return -1.0
    size = _params(model)
    # 7 to 14 billion parameters is the sweet spot: smaller models follow instructions
    # poorly, larger ones are slow on most graphics cards.
    fit = 10 - abs(size - 9) * 0.6 if size else 3
    if size and size < 3:
        fit -= 4
    family = next((len(_FAMILY_RANK) - i for i, f in enumerate(_FAMILY_RANK) if name.startswith(f)), 0)
    return max(fit, 0.1) + family * 0.15


def ollama_status(force: bool = False) -> dict:
    """Is Ollama installed and running here, which version, and which models are usable."""
    if not force and _ollama_cache["data"] is not None and time.time() - _ollama_cache["at"] < 20:
        return _ollama_cache["data"]
    saved = (config.load_settings().get("providers") or {}).get("ollama", {})
    host = re.sub(r"/v1/?$", "", (saved.get("base_url") or PROVIDERS["ollama"]["base_url"]).rstrip("/"))
    exe = _ollama_exe()
    out = {"installed": bool(exe), "running": False, "version": "", "host": host, "models": [], "recommended": "",
           "suggest_pull": ""}
    try:
        ver = httpx.get(host + "/api/version", timeout=httpx.Timeout(2.0, connect=1.0))
        if ver.status_code == 200:
            out["running"], out["installed"] = True, True
            out["version"] = str(ver.json().get("version", ""))
            tags = httpx.get(host + "/api/tags", timeout=httpx.Timeout(5.0, connect=1.0)).json().get("models") or []
            models = [{"name": m.get("name", ""), "params": round(_params(m), 1),
                       "gb": round((m.get("size") or 0) / 1e9, 1), "rank": round(_rank_ollama(m), 2)} for m in tags]
            models.sort(key=lambda m: -m["rank"])
            for m in models:
                m["general"] = m["rank"] > 0
            out["models"] = models
            if models and models[0]["rank"] > 0:
                out["recommended"] = models[0]["name"]
            else:
                out["suggest_pull"] = "qwen2.5:7b"
    except Exception:
        pass
    _ollama_cache.update(at=time.time(), data=out)
    return out


def start_ollama() -> dict:
    """Start the Ollama server if it is installed but not running."""
    import subprocess
    import sys

    status = ollama_status(force=True)
    exe = _ollama_exe()
    if status["running"] or not exe:
        return status
    flags = {"creationflags": 0x08000000 | 0x00000008} if sys.platform.startswith("win") else {"start_new_session": True}
    try:
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, **flags)
    except OSError as exc:
        config.applog(f"[ai] could not start Ollama: {exc}")
        return status
    for _ in range(20):
        time.sleep(0.5)
        status = ollama_status(force=True)
        if status["running"]:
            config.applog(f"[ai] started Ollama {status['version']}")
            break
    return status


def auto_configure() -> None:
    """On a PC where no AI was ever chosen, use Ollama if it is running with a suitable model."""
    saved = config.load_settings()
    if saved.get("provider") or config.get_api_key():
        return
    status = ollama_status(force=True)
    if status["running"] and status["recommended"]:
        allp = dict(saved.get("providers") or {})
        allp["ollama"] = dict(allp.get("ollama") or {}, model=status["recommended"])
        config.save_settings({"provider": "ollama", "providers": allp})
        config.applog(f"[ai] no AI was set up; found Ollama {status['version']} and selected model {status['recommended']}")


def _version_tuple(text: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3]) or (0,)


# ---------------------------------------------------------------------------
# Finding a model that works
# ---------------------------------------------------------------------------

_PREFER = ("gpt-oss-120b", "llama-3.3-70b", "llama3.3-70b", "llama-4", "qwen3", "qwen-3", "gemini-3.5-flash",
           "gemini-flash-latest", "gpt-4.1-mini", "gpt-4o-mini", "command-a", "mistral-small", "mistral-medium",
           "glm-4.5-flash", "glm-4.7-flash", "kimi", "deepseek-v3", "deepseek-chat", "gemma-3", "gemma-4", "llama")
_SKIP = ("tts", "image", "embedding", "embed", "audio", "live", "veo", "lyria", "imagen", "aqa", "banana", "robotics",
         "computer-use", "deep-research", "whisper", "guard", "rerank", "moderation", "coder", "codestral", "vision",
         "ocr", "transcribe", "speech", "safeguard", "antigravity")


def usable_models(names: list[str]) -> list[str]:
    """Drop models that cannot do this job and put the likely best first."""
    keep = [n for n in names if not any(x in n.lower() for x in _SKIP)]

    def score(n: str) -> int:
        low = n.lower()
        return next((i for i, p in enumerate(_PREFER) if p in low), len(_PREFER) + (5 if "preview" in low else 0))
    return sorted(keep, key=lambda n: (score(n), n))


def find_working_model(max_tries: int = 8) -> dict:
    """Try the provider's models one by one until one answers, then save it."""
    cfg = current()
    if cfg["provider"] in ("claude", "ollama"):
        raise RuntimeError("This is only for cloud providers with a model list.")
    if not (cfg["api_key"] or cfg["local"] or cfg.get("keyless")):
        raise RuntimeError("Paste the API key first.")
    candidates = [m for m in [cfg["model"], PROVIDERS[cfg["provider"]].get("model", "")] if m]
    try:
        candidates += usable_models(list_models(cfg))
    except Exception as exc:
        config.applog(f"[ai] could not list models for {cfg['name']}: {exc}")
    seen, tried = set(), []
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        if len(tried) >= max_tries:
            break
        try:
            chat("You are a connection test.", 'Reply with this exact JSON: {"ok": true}', dict(cfg, model=model),
                 True, 300, retries=0, read_timeout=20.0)
        except Exception as exc:
            tried.append({"model": model, "error": str(exc)[:160]})
            continue
        saved = config.load_settings()
        allp = dict(saved.get("providers") or {})
        allp[cfg["provider"]] = dict(allp.get(cfg["provider"]) or {}, model=model)
        config.save_settings({"providers": allp})
        config.applog(f"[ai] {cfg['name']}: model {model} works and was selected")
        return {"ok": True, "model": model, "tried": tried}
    return {"ok": False, "model": "", "tried": tried}


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
    # Answer schemas need Ollama 0.5 or newer; older versions only know plain JSON mode.
    version = ollama_status().get("version", "")
    if schema is not None and (not version or _version_tuple(version) >= (0, 5)):
        payload["format"] = schema
    elif want_json or schema is not None:
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
