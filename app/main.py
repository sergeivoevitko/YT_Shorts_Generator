"""ClipForge web server: serves the GUI and the job API on localhost."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .jobs import DEFAULTS, JobManager, new_id
from .pipeline import captions, llm, render, story, transcribe, voice

app = FastAPI(title="ClipForge")
manager = JobManager()
STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
LOGO = config.WORKSPACE / "brand" / "logo.png"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def get_config():
    return {
        "defaults": DEFAULTS,
        "aspects": list(render.ASPECTS),
        "resolutions": list(render.RESOLUTIONS),
        "caption_styles": {k: {"name": v["name"], "desc": v["desc"], "font": v["font"], "upper": v["upper"],
                               "primary": v["primary"], "active": v["active"], "outline": v["outline_color"]}
                           for k, v in captions.STYLES.items()},
        "hook_styles": {k: v["name"] for k, v in captions.HOOK_STYLES.items()},
        "whisper_models": transcribe.MODELS,
        "claude_models": config.CLAUDE_MODELS,
        "fonts": captions.FONTS,
        "voices": voice.VOICES,
        "story_themes": story.THEMES,
        "story_tones": story.TONES,
        "voice_model": all((voice.model_dir() / f).exists() for f in voice.MODEL_FILES),
        "logo": LOGO.exists(),
        "status": config.system_status(),
    }


@app.post("/api/settings")
def set_settings(payload: dict):
    """Save the AI provider choice. Keys are kept per provider, only on this computer."""
    data: dict = {}
    provider = payload.get("provider")
    if provider in llm.PROVIDERS:
        data["provider"] = provider
    provider = provider if provider in llm.PROVIDERS else llm.current()["provider"]
    key = str(payload.get("api_key") or "").strip()
    model = str(payload.get("model") or "").strip()[:120]
    if provider == "claude":
        if key:
            data["api_key"] = key
        if model in config.CLAUDE_MODELS:
            data["model"] = model
    else:
        allp = dict(config.load_settings().get("providers") or {})
        per = dict(allp.get(provider) or {})
        if key:
            per["api_key"] = key
        if "model" in payload:
            per["model"] = model
        if "base_url" in payload:
            url = str(payload.get("base_url") or "").strip().rstrip("/")
            if url and not re.match(r"^https?://", url):
                raise HTTPException(400, "The server address must start with http:// or https://")
            per["base_url"] = url
        allp[provider] = per
        data["providers"] = allp
    config.save_settings(data)
    return config.system_status()


@app.get("/api/ai/providers")
def ai_providers():
    saved = config.load_settings().get("providers") or {}
    out = {}
    for key, preset in llm.PROVIDERS.items():
        per = saved.get(key, {})
        out[key] = {"name": preset["name"], "key_url": preset.get("key_url", ""), "local": bool(preset.get("local")),
                    "default_model": preset.get("model", ""), "default_url": preset.get("base_url", ""),
                    "model": per.get("model") or preset.get("model", ""),
                    "base_url": per.get("base_url") or preset.get("base_url", ""),
                    "has_key": bool(per.get("api_key")) if key != "claude" else bool(config.get_api_key())}
    out["claude"]["model"] = config.get_model()
    return out


@app.post("/api/ai/test")
def ai_test():
    cur = llm.current()
    try:
        if cur["provider"] == "claude":
            if not cur["ready"]:
                raise RuntimeError("Paste your Anthropic API key first.")
            import anthropic

            client = anthropic.Anthropic(api_key=cur["api_key"], timeout=60.0)
            msg = client.messages.create(model=cur["model"], max_tokens=64,
                                         messages=[{"role": "user", "content": "Reply with the single word: ready"}])
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return {"ok": True, "message": f"Claude answered with model {msg.model}: {text.strip()[:60]}"}
        return {"ok": True, "message": llm.test()}
    except Exception as exc:
        config.applog(f"[ai] connection test failed: {exc}")
        return {"ok": False, "message": str(exc)[:500]}


@app.get("/api/log")
def get_log(lines: int = 300):
    try:
        text = config.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return {"path": str(config.LOG_FILE), "text": "\n".join(text.splitlines()[-max(10, min(lines, 2000)):])}


@app.exception_handler(Exception)
async def log_unhandled(request, exc):
    import traceback

    config.applog(f"[server] {request.method} {request.url.path} crashed: {exc!r}\n"
                  + "".join(traceback.format_exception(exc, limit=6)))
    return JSONResponse({"detail": f"Internal error: {exc}"}, status_code=500)


@app.get("/api/ai/models")
def ai_models():
    cur = llm.current()
    if cur["provider"] == "claude":
        return {"models": list(config.CLAUDE_MODELS)}
    try:
        names = llm.list_models(cur)
        # Hide models that cannot do this job (speech, images, embeddings, live audio).
        skip = ("tts", "image", "embedding", "audio", "live", "veo", "lyria", "imagen", "aqa", "banana", "robotics",
                "computer-use", "deep-research", "whisper", "guard")
        return {"models": [n for n in names if not any(x in n.lower() for x in skip)]}
    except Exception as exc:
        return {"models": [], "error": str(exc)[:300]}


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, length=1024 * 1024)


@app.post("/api/jobs")
def create_job(url: str = Form(""), settings: str = Form("{}"), reuse: str = Form(""), mode: str = Form("clips"),
               video: UploadFile | None = File(None), music: UploadFile | None = File(None),
               footage: list[UploadFile] = File([])):
    try:
        opts = json.loads(settings)
    except ValueError:
        raise HTTPException(400, "Settings are not valid JSON.")
    job_id = new_id()
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source: dict = {}
    url = url.strip()

    old = manager.get(reuse) if reuse else None
    if mode == "story" or (old and old.source.get("kind") == "story"):
        if old:
            files = [f for f in old.source.get("files", []) if Path(f).exists()]
            if not files:
                raise HTTPException(400, "The earlier run's footage is no longer available. Upload the clips again.")
            source = {"kind": "story", "files": files, "reuse": old.source.get("reuse") or old.id, "title": "Narrated video"}
            if old.source.get("music") and Path(old.source["music"]).exists():
                source["music"] = old.source["music"]
        else:
            files = []
            for i, up in enumerate(f for f in footage if f.filename):
                ext = Path(up.filename).suffix.lower()
                if ext not in VIDEO_EXT:
                    raise HTTPException(400, f"Unsupported video type '{ext}' ({up.filename}).")
                dest = job_dir / f"source_{i + 1:02d}{ext}"
                _save_upload(up, dest)
                files.append(str(dest))
            if not files:
                raise HTTPException(400, "Choose at least one video clip of your footage.")
            source = {"kind": "story", "files": files, "title": "Narrated video"}
    elif reuse:
        root = (old.source.get("reuse") or old.id) if old else ""
        if not root or not (config.JOBS_DIR / root / "source.json").exists():
            raise HTTPException(400, "The earlier run's video is no longer available. Start from the link again.")
        source = {"reuse": root, "url": old.source.get("url", ""), "title": old.title}
        if old.source.get("music") and Path(old.source["music"]).exists():
            source["music"] = old.source["music"]
    elif video is not None and video.filename:
        ext = Path(video.filename).suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"Unsupported video type '{ext}'.")
        dest = job_dir / f"source{ext}"
        _save_upload(video, dest)
        source = {"file": str(dest), "title": Path(video.filename).stem}
    elif url:
        if not re.match(r"^https?://", url):
            raise HTTPException(400, "Paste a full link that starts with http:// or https://")
        source = {"url": url}
    else:
        raise HTTPException(400, "Paste a YouTube link or choose a video file.")

    if music is not None and music.filename:
        ext = Path(music.filename).suffix.lower()
        if ext not in AUDIO_EXT:
            raise HTTPException(400, f"Unsupported music type '{ext}'.")
        dest = job_dir / f"music{ext}"
        _save_upload(music, dest)
        source["music"] = str(dest)

    job = manager.create(source, opts, job_id)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"ok": manager.cancel(job_id)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    return {"ok": manager.delete(job_id)}


@app.post("/api/jobs/{job_id}/open")
def open_folder(job_id: str):
    job = manager.get(job_id)
    if not job or not job.dir.exists():
        raise HTTPException(404, "Folder not found")
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(job.dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(job.dir)])
        else:
            subprocess.Popen(["xdg-open", str(job.dir)])
    except Exception as exc:
        return JSONResponse({"ok": False, "path": str(job.dir), "error": str(exc)})
    return {"ok": True, "path": str(job.dir)}


@app.get("/api/jobs/{job_id}/clips/{index}/words")
def clip_words(job_id: str, index: int):
    job = manager.get(job_id)
    if not job or not 1 <= index <= len(job.clips):
        raise HTTPException(404, "Clip not found")
    try:
        return manager.clip_words(job, index)
    except (OSError, ValueError, KeyError, IndexError):
        raise HTTPException(410, "The transcript for this run is no longer available.")


@app.post("/api/jobs/{job_id}/clips/{index}/rerender")
def rerender_clip(job_id: str, index: int, edit: dict):
    if not manager.rerender_clip(job_id, index, edit):
        raise HTTPException(409, "This run is busy or the clip does not exist.")
    return manager.get(job_id).to_dict()


@app.post("/api/brand/logo")
def upload_logo(logo: UploadFile = File(...)):
    """Store the brand logo as PNG (ffmpeg converts whatever image was uploaded)."""
    ext = Path(logo.filename or "").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Use a PNG, JPG or WebP image. PNG with a transparent background works best.")
    LOGO.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOGO.with_name("upload" + ext)
    _save_upload(logo, tmp)
    proc = subprocess.run([config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
                           "-frames:v", "1", "-vf", "scale='min(1024,iw)':-1", str(LOGO)], capture_output=True)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0 or not LOGO.exists():
        raise HTTPException(400, "That image could not be read.")
    return {"ok": True}


@app.delete("/api/brand/logo")
def delete_logo():
    LOGO.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/brand/logo")
def get_logo():
    if not LOGO.exists():
        raise HTTPException(404, "No logo")
    return FileResponse(LOGO, headers={"Cache-Control": "no-store"})


@app.get("/media/{job_id}/{name}")
def media(job_id: str, name: str, download: int = 0):
    job = manager.get(job_id)
    if not job or not re.fullmatch(r"[\w.\-]+", name) or Path(name).suffix not in (".mp4", ".jpg", ".txt"):
        raise HTTPException(404, "Not found")
    path = job.dir / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    if download:
        return FileResponse(path, filename=name)
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import threading
    import webbrowser

    import uvicorn

    host = os.environ.get("CLIPFORGE_HOST", "127.0.0.1")
    port = int(os.environ.get("CLIPFORGE_PORT", "8765"))
    if os.environ.get("CLIPFORGE_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    config.applog(f"[server] started on http://{host}:{port}, ffmpeg={config.ffmpeg_path() or 'missing'}, "
                  f"ai={llm.current()['name']}/{llm.current()['model']}")
    print(f"\n  ClipForge is running at http://{host}:{port}\n  Keep this window open. Press Ctrl+C to stop.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
