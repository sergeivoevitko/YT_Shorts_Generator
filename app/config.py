"""Paths, persistent settings and external tool discovery."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONTS_DIR = ASSETS / "fonts"
MODELS_DIR = ASSETS / "models"
WORKSPACE = Path(os.environ.get("CLIPFORGE_WORKSPACE", ROOT / "workspace"))
JOBS_DIR = WORKSPACE / "jobs"
SETTINGS_FILE = WORKSPACE / "settings.json"

CLAUDE_MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5 (best balance, default)",
    "claude-fable-5-2": "Claude Fable 5.2 (most capable, priciest)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (fast, cheaper)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (cheapest)",
}
DEFAULT_MODEL = "claude-opus-5-5"


LOG_FILE = WORKSPACE / "clipforge.log"


def applog(message: str) -> None:
    """Append a line to the app log (shown under Log in the app). Never pass secrets."""
    import time

    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    try:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
            LOG_FILE.write_text(LOG_FILE.read_text(encoding="utf-8", errors="replace")[-300_000:], encoding="utf-8")
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    ensure_dirs()
    current = load_settings()
    current.update(data)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def get_api_key() -> str:
    return (load_settings().get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def get_model() -> str:
    model = load_settings().get("model") or DEFAULT_MODEL
    return model if model in CLAUDE_MODELS else DEFAULT_MODEL


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """Prefer an ffmpeg on PATH, fall back to the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def ffmpeg_capabilities() -> dict:
    exe = ffmpeg_path()
    caps = {"found": bool(exe), "path": exe, "libass": False, "version": ""}
    if not exe:
        return caps
    try:
        out = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
        caps["libass"] = " ass " in out
        ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20).stdout
        caps["version"] = ver.splitlines()[0] if ver else ""
    except Exception:
        pass
    return caps


def js_runtimes() -> dict:
    """JavaScript runtimes yt-dlp can use to solve YouTube's player challenges."""
    runtimes = {}
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            runtimes[name] = {"path": path}
    return runtimes


def _ai_status() -> dict:
    from .pipeline import llm

    cur = llm.current()
    return {"provider": cur["provider"], "name": cur["name"], "model": cur["model"], "ready": cur["ready"],
            "local": cur["local"], "base_url": cur["base_url"], "has_key": bool(cur["api_key"])}


def system_status() -> dict:
    caps = ffmpeg_capabilities()
    return {
        "ffmpeg": caps,
        "js_runtime": sorted(js_runtimes().keys()),
        "api_key_set": bool(get_api_key()),
        "ai": _ai_status(),
        "model": get_model(),
        "face_model": (MODELS_DIR / "yunet.onnx").exists(),
    }
