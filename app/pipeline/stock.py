"""Free stock footage from Pexels, for narrated videos made without your own footage.

Pexels videos are free to use, including commercially, with no attribution required
(attribution is appreciated). A free API key is needed: https://www.pexels.com/api/
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx

from .. import config

API = "https://api.pexels.com/videos/search"
KEY_URL = "https://www.pexels.com/api/"


def api_key() -> str:
    return str(config.load_settings().get("pexels_key") or "").strip()


def _pick_file(video: dict, portrait: bool, max_height: int) -> dict | None:
    """The mp4 rendition closest to the size we need, without going far above it."""
    files = [f for f in video.get("video_files") or []
             if str(f.get("file_type", "")).endswith("mp4") and f.get("link") and f.get("width") and f.get("height")]
    if not files:
        return None
    want = max_height if portrait else int(max_height * 9 / 16)  # long side for portrait, short side otherwise
    side = (lambda f: max(f["width"], f["height"])) if portrait else (lambda f: min(f["width"], f["height"]))
    ok = [f for f in files if side(f) <= want * 1.15]
    return max(ok, key=side) if ok else min(files, key=side)


def search(term: str, portrait: bool, key: str, per_page: int = 8) -> list[dict]:
    params = {"query": term, "per_page": per_page, "size": "medium",
              "orientation": "portrait" if portrait else "landscape"}
    resp = httpx.get(API, params=params, headers={"Authorization": key}, timeout=30)
    if resp.status_code in (401, 403):
        raise RuntimeError("Pexels refused the key. Check the Pexels API key in the story settings.")
    if resp.status_code == 429:
        raise RuntimeError("Pexels' free limit for this hour is used up. Try again later.")
    resp.raise_for_status()
    return resp.json().get("videos") or []


def fetch(terms: list[str], out_dir: Path, portrait: bool, max_height: int, need_seconds: float,
          progress: Callable[[float, str], None], cancelled: Callable[[], bool],
          log: Callable[[str], None]) -> list[dict]:
    """Download one clip per search term, in order. Returns [{"path", "term", "credit", "url"}]."""
    key = api_key()
    if not key:
        raise RuntimeError("Stock footage needs a free Pexels API key. Get one at pexels.com/api and paste it in "
                           "the story settings, or upload your own clips instead.")
    out_dir.mkdir(parents=True, exist_ok=True)
    terms = [t for t in dict.fromkeys(t.strip().lower() for t in terms) if t][:12]
    used: set[int] = set()
    clips: list[dict] = []
    total = 0.0
    for n, term in enumerate(terms):
        if cancelled():
            raise RuntimeError("Cancelled")
        progress(n / max(len(terms), 1), f"Finding stock footage: {term}")
        try:
            videos = search(term, portrait, key)
            if not videos and " " in term:  # a narrower search found nothing: try its last word
                videos = search(term.split()[-1], portrait, key)
        except RuntimeError:
            raise
        except Exception as exc:
            log(f"Stock search for '{term}' failed: {exc}")
            continue
        choice = next((v for v in videos if v.get("id") not in used and (v.get("duration") or 0) >= 4), None)
        if not choice:
            log(f"No stock footage found for '{term}'.")
            continue
        f = _pick_file(choice, portrait, max_height)
        if not f:
            continue
        dest = out_dir / f"stock_{len(clips) + 1:02d}.mp4"
        try:
            with httpx.stream("GET", f["link"], timeout=httpx.Timeout(120.0, connect=20.0), follow_redirects=True) as r:
                r.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        if cancelled():
                            raise RuntimeError("Cancelled")
                        fh.write(chunk)
        except RuntimeError:
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:
            dest.unlink(missing_ok=True)
            log(f"Downloading stock footage for '{term}' failed: {exc}")
            continue
        used.add(choice.get("id"))
        user = choice.get("user") or {}
        clips.append({"path": str(dest), "term": term, "url": choice.get("url", ""),
                      "credit": user.get("name", "Pexels")})
        total += float(choice.get("duration") or 0)
    if not clips:
        raise RuntimeError("No stock footage could be found for this script. Try another topic or upload your own clips.")
    log(f"Stock footage: {len(clips)} clips from Pexels for {len(terms)} searches ({total:.0f}s in total).")
    if total < need_seconds * 0.6:
        log("There is less stock footage than narration, so some shots will repeat.")
    return clips
