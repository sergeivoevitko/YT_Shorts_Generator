"""Results tracker: remember where clips were posted, fetch their view counts, and
work out which choices (hook type, caption style, length, cold open) perform best."""
from __future__ import annotations

import json
import threading
import time
import uuid

from . import config

FILE = config.WORKSPACE / "tracker.json"
_lock = threading.Lock()


def _load() -> list[dict]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8")).get("entries", [])
    except (OSError, ValueError):
        return []


def _save(entries: list[dict]) -> None:
    config.ensure_dirs()
    FILE.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def platform_of(url: str) -> str:
    low = url.lower()
    for key, name in (("tiktok.", "TikTok"), ("instagram.", "Instagram"), ("youtu", "YouTube"), ("facebook.", "Facebook")):
        if key in low:
            return name
    return "Other"


def fetch_stats(url: str) -> dict:
    """Public view and like counts for a posted video, read with yt-dlp (no account needed)."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    runtimes = config.js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    opts.update(config.youtube_auth_options())
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("view_count") is None:
        raise RuntimeError("the platform did not report a view count")
    return {"views": int(info["view_count"]), "likes": info.get("like_count"), "comments": info.get("comment_count")}


def add(job_id: str, clip: dict, url: str) -> dict:
    entry = {
        "id": uuid.uuid4().hex[:10], "job_id": job_id, "file": clip.get("file", ""), "title": clip.get("title", ""),
        "hook": clip.get("hook", ""), "hook_type": clip.get("hook_type", ""), "style": clip.get("style") or "none",
        "duration": clip.get("duration", 0), "cold_open": bool(clip.get("cold_open")), "variant": clip.get("variant", ""),
        "platform": platform_of(url), "url": url, "added": time.time(),
        "views": None, "likes": None, "updated": 0, "error": "",
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return refresh(entry["id"]) or entry


def refresh(entry_id: str | None = None) -> dict | None:
    """Update one entry, or all of them when entry_id is None. Returns the entry for a single update."""
    with _lock:
        entries = _load()
    target = None
    for e in entries:
        if entry_id and e["id"] != entry_id:
            continue
        try:
            stats = fetch_stats(e["url"])
            e.update(views=stats["views"], likes=stats["likes"], updated=time.time(), error="")
        except Exception as exc:
            e["error"] = "Could not read the view count automatically. Type it in by hand."
            config.applog(f"[tracker] {e['url']}: {str(exc)[:200]}")
        target = e
    with _lock:
        current = {x["id"]: x for x in _load()}
        for e in entries:
            if e["id"] in current:
                current[e["id"]].update({k: e[k] for k in ("views", "likes", "updated", "error")})
        _save(list(current.values()))
    return target if entry_id else None


def set_views(entry_id: str, views: int) -> bool:
    with _lock:
        entries = _load()
        for e in entries:
            if e["id"] == entry_id:
                e.update(views=max(0, int(views)), updated=time.time(), error="")
                _save(entries)
                return True
    return False


def remove(entry_id: str) -> None:
    with _lock:
        _save([e for e in _load() if e["id"] != entry_id])


def _bucket(seconds: float) -> str:
    return "under 20s" if seconds < 20 else "20 to 35s" if seconds < 35 else "35 to 60s" if seconds < 60 else "over 60s"


DIMENSIONS = [
    ("Hook type", lambda e: e.get("hook_type") or ""),
    ("Caption style", lambda e: e.get("style") or ""),
    ("Length", lambda e: _bucket(float(e.get("duration") or 0))),
    ("Cold open", lambda e: "with cold open" if e.get("cold_open") else "without cold open"),
    ("Platform", lambda e: e.get("platform") or ""),
]


def insights(entries: list[dict] | None = None) -> list[dict]:
    """Average views per choice, for every dimension that has something to compare."""
    entries = [e for e in (entries if entries is not None else _load()) if e.get("views") is not None]
    out = []
    for name, key in DIMENSIONS:
        groups: dict[str, list[int]] = {}
        for e in entries:
            label = key(e)
            if label:
                groups.setdefault(label, []).append(int(e["views"]))
        rows = [{"label": k, "count": len(v), "avg": round(sum(v) / len(v))} for k, v in groups.items()]
        if len(rows) >= 2:
            out.append({"dimension": name, "rows": sorted(rows, key=lambda r: -r["avg"])})
    return out


def summary_for_ai() -> str:
    """One paragraph for the analysis prompt, once there is enough data to mean something."""
    entries = [e for e in _load() if e.get("views") is not None]
    if len(entries) < 6:
        return ""
    lines = []
    for block in insights(entries):
        if block["dimension"] == "Platform":
            continue
        solid = [r for r in block["rows"] if r["count"] >= 2]
        if len(solid) >= 2 and solid[0]["avg"] >= solid[-1]["avg"] * 1.3:
            lines.append(f"{block['dimension']}: '{solid[0]['label']}' averages {solid[0]['avg']} views "
                         f"against {solid[-1]['avg']} for '{solid[-1]['label']}'")
    if not lines:
        return ""
    return ("Results from this creator's own posted clips so far (small sample, treat as a lean, not a rule): "
            + "; ".join(lines) + ".")
