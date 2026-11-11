"""Job queue: one background worker runs the pipeline for each submitted video."""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config
from .pipeline import analyze, captions, download, reframe, render, transcribe

DEFAULTS = {
    "aspect": "9:16", "resolution": "1080p", "fps": 30, "quality": "balanced",
    "num_clips": 5, "min_len": 20, "max_len": 60,
    "reframe": "smart",
    "caption_style": "mix", "caption_position": "lower", "caption_scale": 1.0,
    "hook_style": "banner", "hook_seconds": 3.5,
    "remove_silence": True, "silence_gap": 0.5,
    "zooms": True, "zoom_amount": 1.12,
    "progress_bar": True, "normalize_audio": True, "music_volume": 0.12,
    "whisper_model": "small.en", "use_ai": True, "instructions": "",
}

STAGES = [("download", 0.10), ("transcribe", 0.30), ("analyze", 0.08), ("render", 0.52)]


def _num(value, lo, hi, default, cast=float):
    try:
        return max(lo, min(hi, cast(value)))
    except (TypeError, ValueError):
        return default


def clean_settings(raw: dict) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    s["aspect"] = s["aspect"] if s["aspect"] in render.ASPECTS else "9:16"
    s["resolution"] = s["resolution"] if s["resolution"] in render.RESOLUTIONS else "1080p"
    s["quality"] = s["quality"] if s["quality"] in render.QUALITY else "balanced"
    s["fps"] = 60 if str(s["fps"]) == "60" else 30
    s["num_clips"] = _num(s["num_clips"], 1, 20, 5, int)
    s["min_len"] = _num(s["min_len"], 5, 170, 20)
    s["max_len"] = _num(s["max_len"], s["min_len"] + 5, 180, max(60, s["min_len"] + 5))
    s["reframe"] = s["reframe"] if s["reframe"] in ("smart", "center", "blur") else "smart"
    if s["caption_style"] not in captions.STYLES and s["caption_style"] not in ("mix", "none"):
        s["caption_style"] = "mix"
    s["caption_position"] = s["caption_position"] if s["caption_position"] in ("lower", "center", "top") else "lower"
    s["caption_scale"] = _num(s["caption_scale"], 0.6, 1.6, 1.0)
    s["hook_style"] = s["hook_style"] if s["hook_style"] in captions.HOOK_STYLES or s["hook_style"] == "none" else "banner"
    s["hook_seconds"] = _num(s["hook_seconds"], 1, 10, 3.5)
    s["silence_gap"] = _num(s["silence_gap"], 0.25, 2.0, 0.5)
    s["zoom_amount"] = _num(s["zoom_amount"], 1.03, 1.4, 1.12)
    s["music_volume"] = _num(s["music_volume"], 0.0, 1.0, 0.12)
    s["whisper_model"] = s["whisper_model"] if s["whisper_model"] in transcribe.MODELS else "small.en"
    for key in ("remove_silence", "zooms", "progress_bar", "normalize_audio", "use_ai"):
        s[key] = bool(s[key])
    s["instructions"] = str(s["instructions"])[:2000]
    return s


class Job:
    def __init__(self, job_id: str, source: dict, settings: dict):
        self.id = job_id
        self.dir = config.JOBS_DIR / job_id
        self.source = source          # {"url"} | {"file"} | {"reuse"}
        self.settings = settings
        self.status = "queued"        # queued | running | done | error | cancelled
        self.stage = "queued"
        self.progress = 0.0
        self.message = "Waiting to start"
        self.error = ""
        self.title = source.get("title", "")
        self.engine = ""
        self.clips: list[dict] = []
        self.log: list[str] = []
        self.created = time.time()
        self.finished = 0.0
        self.cancel_flag = False

    def to_dict(self, full: bool = True) -> dict:
        data = {
            "id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 4),
            "message": self.message, "error": self.error, "title": self.title, "engine": self.engine,
            "created": self.created, "finished": self.finished, "clip_count": len(self.clips),
            "source_url": self.source.get("url", ""),
        }
        if full:
            data.update({"settings": self.settings, "clips": self.clips, "log": self.log[-200:]})
        return data

    def save(self) -> None:
        try:
            data = self.to_dict()
            data["source"] = self.source
            (self.dir / "job.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        config.ensure_dirs()
        self._load_history()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_history(self) -> None:
        for path in sorted(config.JOBS_DIR.glob("*/job.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(data["id"], data.get("source", {}), data.get("settings", {}))
                for key in ("status", "stage", "progress", "message", "error", "title", "engine", "clips",
                            "log", "created", "finished"):
                    setattr(job, key, data.get(key, getattr(job, key)))
                if job.status in ("queued", "running"):
                    job.status, job.message = "error", "Interrupted when the app was closed"
                    job.error = job.message
                self.jobs[job.id] = job
            except (OSError, ValueError, KeyError):
                continue

    def create(self, source: dict, settings: dict, job_id: str | None = None) -> Job:
        job = Job(job_id or new_id(), source, clean_settings(settings))
        job.dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
        job.save()
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.to_dict(full=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.cancel_flag = True
        if job.status == "queued":
            job.status, job.message, job.finished = "cancelled", "Cancelled", time.time()
            job.save()
        return True

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in ("queued", "running"):
            return False
        # Keep the folder if a later job reuses this one's download and transcript.
        if any(j.source.get("reuse") == job_id for j in self.jobs.values()):
            for p in job.dir.glob("*"):
                if p.suffix in (".mp4", ".jpg") and not p.name.startswith("source"):
                    p.unlink(missing_ok=True)
        else:
            shutil.rmtree(job.dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job_id, None)
        return True

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job = self.jobs.get(self.queue.get())
            if not job or job.status != "queued":
                continue
            job.status = "running"
            try:
                self._run(job)
                job.status, job.stage, job.progress = "done", "done", 1.0
                job.message = f"Done: {len(job.clips)} clips ready"
            except Exception as exc:
                if job.cancel_flag:
                    job.status, job.message = "cancelled", "Cancelled"
                else:
                    job.status, job.error = "error", str(exc)
                    job.message = "Failed"
                    job.log.append(traceback.format_exc(limit=4))
            job.finished = time.time()
            job.save()

    def _run(self, job: Job) -> None:
        s = job.settings
        cancelled = lambda: job.cancel_flag
        offsets, acc = {}, 0.0
        for name, weight in STAGES:
            offsets[name] = (acc, weight)
            acc += weight

        def log(msg: str) -> None:
            job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        def stage(name: str):
            base, weight = offsets[name]
            job.stage = name

            def update(p: float, msg: str) -> None:
                if job.cancel_flag:
                    raise RuntimeError("Cancelled")
                job.progress = base + weight * max(0.0, min(p, 1.0))
                job.message = msg
            return update

        caps = config.ffmpeg_capabilities()
        if not caps["found"]:
            raise RuntimeError("FFmpeg was not found. Run install.bat again or install FFmpeg and restart the app.")
        if not caps["libass"] and (s["caption_style"] != "none" or s["hook_style"] != "none"):
            raise RuntimeError("This FFmpeg build cannot draw subtitles (libass is missing). "
                               "Install the full FFmpeg build (winget install Gyan.FFmpeg) and restart.")

        # 1. Source video -------------------------------------------------
        up = stage("download")
        src_dir = job.dir
        if job.source.get("reuse"):
            src_dir = config.JOBS_DIR / job.source["reuse"]
            meta = json.loads((src_dir / "source.json").read_text(encoding="utf-8"))
            log("Reusing the downloaded video and transcript from the earlier run.")
        elif job.source.get("file"):
            meta = {"path": job.source["file"], "title": job.source.get("title") or "Uploaded video", "url": ""}
        else:
            up(0.0, "Starting download")
            short = render.RESOLUTIONS[s["resolution"]]
            meta = download.download(job.source["url"], job.dir, 1080 if short <= 1080 else 2160, up, cancelled)
            log(f"Downloaded: {meta['title']}")
        info = reframe.probe(meta["path"])
        meta.update(info)
        if not job.source.get("reuse"):
            (job.dir / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        job.title = meta["title"]
        log(f"Source: {info['width']}x{info['height']}, {info['fps']:.0f} fps, {info['duration'] / 60:.1f} min")
        job.save()

        # 2. Transcript ---------------------------------------------------
        up = stage("transcribe")
        tpath = src_dir / f"transcript.{s['whisper_model']}.json"
        if tpath.exists():
            transcript = json.loads(tpath.read_text(encoding="utf-8"))
            log("Using the saved transcript.")
        else:
            if not info["has_audio"]:
                raise RuntimeError("This video has no audio track, so there is no speech to build clips from.")
            transcript = transcribe.transcribe(meta["path"], s["whisper_model"], up, cancelled, info["duration"])
            tpath.write_text(json.dumps(transcript), encoding="utf-8")
            log(f"Transcribed {len(transcript['words'])} words in {len(transcript['sentences'])} sentences.")
        words, sentences = transcript["words"], transcript["sentences"]

        # 3. Find the moments ----------------------------------------------
        up = stage("analyze")
        up(0.1, "Finding the best moments" + (" with Claude" if s["use_ai"] and config.get_api_key() else ""))
        plans, engine = analyze.analyze(sentences, meta["title"], s["num_clips"], s["min_len"], s["max_len"],
                                        s["use_ai"], s["instructions"], log)
        job.engine = engine
        log(f"Selected {len(plans)} clips using {'Claude' if engine == 'claude' else 'offline scoring'}.")
        (job.dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")

        # 4. Render ---------------------------------------------------------
        up = stage("render")
        fonts = job.dir / "fonts"
        if not fonts.exists():
            shutil.copytree(config.FONTS_DIR, fonts)
        music = job.source.get("music")
        job.clips = []
        for i, plan in enumerate(plans, start=1):
            if s["caption_style"] == "none":
                style = None
            elif s["caption_style"] == "mix":
                style = captions.MIX_ORDER[(i - 1) % len(captions.MIX_ORDER)]
            else:
                style = s["caption_style"]
            n = len(plans)
            result = render.render_clip(
                meta["path"], info, plan, words, s, job.dir, i, style, music,
                lambda p, msg, i=i: up((i - 1 + p) / n, f"{msg} ({i} of {n})"), cancelled, log)
            job.clips.append(result)
            job.save()
        shutil.rmtree(fonts, ignore_errors=True)
        _write_post_text(job)


def _write_post_text(job: Job) -> None:
    lines = [f"{job.title}", ""]
    for c in job.clips:
        tags = " ".join("#" + h for h in c["hashtags"])
        lines += [f"=== {c['file']}  (score {c['score']}, {c['duration']}s) ===", f"Title: {c['title']}",
                  f"Hook: {c['hook']}", f"Caption: {c['description']} {tags}".rstrip(), f"Why: {c['reason']}", ""]
    (job.dir / "post_text.txt").write_text("\n".join(lines), encoding="utf-8")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
