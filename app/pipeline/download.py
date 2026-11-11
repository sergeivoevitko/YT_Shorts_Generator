"""Download a YouTube video with yt-dlp."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import config


class DownloadError(RuntimeError):
    pass


def download(url: str, out_dir: Path, max_height: int, progress: Callable[[float, str], None],
             cancelled: Callable[[], bool]) -> dict:
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if cancelled():
            raise DownloadError("Cancelled")
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress(done / total, f"Downloading video ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")

    # H.264 first: it decodes everywhere (OpenCV included). Above 1080p YouTube
    # only offers VP9/AV1, so the codec preference is dropped for those sizes.
    if max_height <= 1080:
        fmt = (f"bv*[vcodec^=avc1][height<={max_height}]+ba[ext=m4a]/"
               f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b")
    else:
        fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"

    opts = {
        "format": fmt,
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
    }
    if config.ffmpeg_path():
        opts["ffmpeg_location"] = config.ffmpeg_path()
    runtimes = config.js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError:
        raise
    except Exception as exc:  # yt-dlp raises many types
        msg = str(exc)
        hint = ""
        if not runtimes:
            hint = (" No JavaScript runtime (Deno or Node.js) was found, which YouTube downloads now need. "
                    "Run install.bat again or install Deno from https://deno.com.")
        elif "Sign in" in msg or "bot" in msg:
            hint = " YouTube asked for a sign-in. Try again later, or download the video yourself and use the upload option."
        raise DownloadError(f"Could not download the video: {msg}.{hint}") from exc

    files = [p for p in out_dir.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
    if not files:
        raise DownloadError("Download finished but no video file was produced.")
    return {
        "path": str(files[0]),
        "title": info.get("title") or "Untitled video",
        "duration": float(info.get("duration") or 0),
        "uploader": info.get("uploader") or "",
        "url": info.get("webpage_url") or url,
    }
