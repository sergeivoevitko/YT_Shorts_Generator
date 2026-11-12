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

    opts.update(config.youtube_auth_options())
    signed_in = bool(config.youtube_auth_options())

    def is_bot_check(text: str) -> bool:
        return "Sign in to confirm" in text or "not a bot" in text

    # YouTube sometimes answers "sign in to confirm you're not a bot" for one way of asking and
    # not for another, so try a few before giving up.
    routes = [None, ["tv", "web_safari"], ["android_vr"], ["mweb", "tv_simply"]]
    info, last = None, None
    for n, clients in enumerate(routes):
        attempt = dict(opts)
        if clients:
            attempt["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                info = ydl.extract_info(url, download=True)
            if n:
                config.applog(f"[download] worked through route {n + 1} ({', '.join(clients)})")
            break
        except DownloadError:
            raise
        except Exception as exc:  # yt-dlp raises many types
            last = exc
            if cancelled():
                raise DownloadError("Cancelled") from exc
            text = str(exc)
            config.applog(f"[download] route {n + 1} failed: {text[:200]}")
            if "youtube" not in text.lower() and "youtu" not in url.lower():
                break
            if not is_bot_check(text) and "cookies" not in text.lower() and n >= 1:
                break  # a different kind of error: more routes will not help
            progress(0.0, "YouTube refused the download, trying another way")
    if info is None:
        msg = str(last)
        hint = ""
        if not runtimes:
            hint = (" No JavaScript runtime (Deno or Node.js) was found, which YouTube downloads now need. "
                    "Run install.bat again or install Deno from https://deno.com.")
        elif "could not find" in msg.lower() and "cookies" in msg.lower() or "cookie database" in msg.lower() \
                or "decrypt" in msg.lower():
            hint = (" The app could not read that browser's sign-in (Chrome and Edge lock it on Windows). In "
                    "YouTube access, choose Firefox or use a cookie file instead.")
        elif is_bot_check(msg):
            hint = (" This is YouTube's bot check, not a problem with the video or the app. It is triggered by many "
                    "downloads from one internet connection and usually clears within a few hours. "
                    + ("Your YouTube sign-in was used and still refused; try again later. " if signed_in else
                       "To carry on now, press 'YouTube access' under the link field and let the app use your "
                       "YouTube sign-in, ")
                    + "or download the video yourself and use the upload option.")
        raise DownloadError(f"Could not download the video: {msg[:300]}.{hint}") from last

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
