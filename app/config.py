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


def preload_runtime() -> None:
    """Windows only: load the system's Microsoft C++ runtime before any library brings
    its own. Libraries built with a recent compiler crash (access violation) when an
    older copy of msvcp140.dll is the one in the process."""
    import sys

    if not sys.platform.startswith("win"):
        return
    import ctypes

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "concrt140.dll"):
        try:
            ctypes.WinDLL(str(system32 / name))
        except OSError:
            pass


def _file_version(path: str) -> str:
    """Version of a Windows DLL as 'major.minor.build', or '' if unknown."""
    try:
        import ctypes
        from ctypes import wintypes

        ver = ctypes.windll.version
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        ver.GetFileVersionInfoW(path, 0, size, buf)
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return ""
        info = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        ms, ls = info[2], info[3]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}"
    except Exception:
        return ""


def system_report() -> dict:
    """Facts about this machine that matter when something crashes."""
    import platform
    import sys
    from importlib import metadata

    report = {"python": sys.version.split()[0], "os": platform.platform(), "cpu": platform.processor() or platform.machine()}
    for pkg in ("ctranslate2", "faster-whisper", "onnxruntime", "kokoro-onnx", "numpy", "opencv-python-headless", "yt-dlp"):
        try:
            report[pkg] = metadata.version(pkg)
        except Exception:
            report[pkg] = "missing"
    report["msvc_runtime"] = ""
    if sys.platform.startswith("win"):
        report["msvc_runtime"] = _file_version(str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "msvcp140.dll")) or "not installed"
    return report


def msvc_runtime_old() -> bool:
    """True when the Microsoft C++ runtime is missing or older than 14.40 (Windows only)."""
    import sys

    if not sys.platform.startswith("win"):
        return False
    ver = system_report()["msvc_runtime"]
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
    except ValueError:
        return True
    return (major, minor) < (14, 40)


def groq_key() -> str:
    return ((load_settings().get("providers") or {}).get("groq") or {}).get("api_key", "").strip()


GPU_ENCODERS = [("h264_nvenc", "NVIDIA"), ("h264_qsv", "Intel Quick Sync"), ("h264_amf", "AMD"),
                ("h264_videotoolbox", "Apple")]


def enable_cuda_libraries() -> None:
    """Make the NVIDIA libraries installed by pip (cuBLAS, cuDNN) findable, so speech
    recognition can run on the graphics card without a separate CUDA install."""
    import site
    import sys

    roots = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            roots += [p for p in nvidia.glob("*/bin") if p.is_dir()] + [p for p in nvidia.glob("*/lib") if p.is_dir()]
    for path in roots:
        try:
            if sys.platform.startswith("win"):
                os.add_dll_directory(str(path))
                os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            elif sys.platform.startswith("linux"):
                import ctypes

                for lib in sorted(path.glob("lib*.so*")):
                    try:
                        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
        except OSError:
            pass


@lru_cache(maxsize=1)
def gpu_status() -> dict:
    """Which GPU features work on this machine, found by trying them once."""
    status = {"encoder": "", "encoder_name": "", "speech": False}
    exe = ffmpeg_path()
    if exe:
        for codec, name in GPU_ENCODERS:
            try:
                proc = subprocess.run(
                    [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30",
                     "-frames:v", "10", "-c:v", codec, "-f", "null", "-"], capture_output=True, timeout=30)
            except Exception:
                continue
            if proc.returncode == 0:
                status["encoder"], status["encoder_name"] = codec, name
                break
    try:
        import ctranslate2

        status["speech"] = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    applog(f"[gpu] video encoder: {status['encoder'] or 'none (CPU)'}, speech recognition on GPU: {status['speech']}")
    return status


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
        "gpu": gpu_status(),
        "groq_key": bool(groq_key()),
        "msvc_runtime_old": msvc_runtime_old(),
        "model": get_model(),
        "face_model": (MODELS_DIR / "yunet.onnx").exists(),
    }
