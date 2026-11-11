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
from .pipeline import captions, render, transcribe

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
        "logo": LOGO.exists(),
        "status": config.system_status(),
    }


@app.post("/api/settings")
def set_settings(payload: dict):
    data = {}
    if "api_key" in payload:
        data["api_key"] = str(payload["api_key"]).strip()
    if payload.get("model") in config.CLAUDE_MODELS:
        data["model"] = payload["model"]
    config.save_settings(data)
    return config.system_status()


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, length=1024 * 1024)


@app.post("/api/jobs")
def create_job(url: str = Form(""), settings: str = Form("{}"), reuse: str = Form(""),
               video: UploadFile | None = File(None), music: UploadFile | None = File(None)):
    try:
        opts = json.loads(settings)
    except ValueError:
        raise HTTPException(400, "Settings are not valid JSON.")
    job_id = new_id()
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source: dict = {}
    url = url.strip()

    if reuse:
        old = manager.get(reuse)
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
    print(f"\n  ClipForge is running at http://{host}:{port}\n  Keep this window open. Press Ctrl+C to stop.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
