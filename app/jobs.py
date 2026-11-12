"""Job queue: one background worker runs the pipeline for each submitted video."""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config, tracker
from .pipeline import analyze, captions, download, footage, llm, reframe, render, story, transcribe, voice

DEFAULTS = {
    "aspect": "9:16", "resolution": "1080p", "fps": "source", "quality": "balanced", "encoder": "auto",
    "num_clips": 5, "min_len": 20, "max_len": 60,
    "reframe": "smart",
    "caption_style": "mix", "caption_position": "lower", "caption_scale": 1.0,
    "hook_style": "banner", "hook_seconds": 3.5,
    "remove_silence": True, "silence_gap": 0.5,
    "zooms": True, "zoom_amount": 1.12,
    "progress_bar": True, "normalize_audio": True, "music_volume": 0.12,
    "remove_fillers": True, "sfx": True, "sfx_volume": 0.5, "emojis": True,
    "logo_enabled": False, "logo_position": "top-right", "logo_scale": 0.16, "logo_opacity": 0.9,
    "brand_color": "", "brand_font": "", "handle_text": "",
    # "My footage + story" mode
    "story_theme": "motivation", "story_tone": "cinematic", "story_text": "", "story_seconds": 40,
    "num_videos": 1, "voice": voice.DEFAULT_VOICE, "voice_speed": 0.95, "transition": "crossfade",
    "footage_order": "upload", "ambient_pad": True,
    "whisper_model": "small.en", "speech_engine": "local", "use_ai": True, "instructions": "",
    "review_picks": True,
    "cold_open": True, "header": True, "header_text": "", "ab_versions": False, "platform": "universal",
}

STAGES = [("download", 0.10), ("transcribe", 0.30), ("analyze", 0.08), ("render", 0.52)]
STORY_STAGES = [("footage", 0.12), ("story", 0.05), ("voice", 0.23), ("render", 0.60)]


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
    s["platform"] = s["platform"] if s["platform"] in captions.SAFE_ZONES else "universal"
    s["header_text"] = str(s["header_text"])[:44]
    s["speech_engine"] = "groq" if s["speech_engine"] == "groq" else "local"
    s["encoder"] = "cpu" if s["encoder"] == "cpu" else "auto"
    s["fps"] = str(s["fps"]) if str(s["fps"]) in ("30", "60") else "source"
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
    s["story_theme"] = s["story_theme"] if s["story_theme"] in story.THEMES else "motivation"
    s["story_tone"] = s["story_tone"] if s["story_tone"] in story.TONES else "cinematic"
    s["story_text"] = str(s["story_text"])[:4000]
    s["story_seconds"] = _num(s["story_seconds"], 15, 90, 40)
    s["num_videos"] = _num(s["num_videos"], 1, 8, 1, int)
    s["voice"] = s["voice"] if s["voice"] in voice.VOICES else voice.DEFAULT_VOICE
    s["voice_speed"] = _num(s["voice_speed"], 0.7, 1.3, 0.95)
    s["transition"] = s["transition"] if s["transition"] in ("crossfade", "cut") else "crossfade"
    s["footage_order"] = s["footage_order"] if s["footage_order"] in ("upload", "best") else "upload"
    s["ambient_pad"] = bool(s["ambient_pad"])
    s["sfx_volume"] = _num(s["sfx_volume"], 0.0, 1.5, 0.5)
    s["logo_scale"] = _num(s["logo_scale"], 0.05, 0.5, 0.16)
    s["logo_opacity"] = _num(s["logo_opacity"], 0.1, 1.0, 0.9)
    if s["logo_position"] not in ("top-left", "top-right", "bottom-left", "bottom-right"):
        s["logo_position"] = "top-right"
    s["brand_color"] = s["brand_color"] if re.fullmatch(r"#[0-9A-Fa-f]{6}", str(s["brand_color"])) else ""
    s["brand_font"] = s["brand_font"] if s["brand_font"] in captions.FONTS else ""
    s["handle_text"] = str(s["handle_text"])[:40]
    for key in ("remove_silence", "zooms", "progress_bar", "normalize_audio", "use_ai", "remove_fillers", "sfx",
                "emojis", "logo_enabled", "review_picks", "cold_open", "header", "ab_versions"):
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
        self.candidates: list[dict] = []   # moments waiting for the user's choice (status "review")
        self.log: list[str] = []
        self.created = time.time()
        self.finished = 0.0
        self.cancel_flag = False

    def to_dict(self, full: bool = True) -> dict:
        data = {
            "id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 4),
            "message": self.message, "error": self.error, "title": self.title, "engine": self.engine,
            "created": self.created, "finished": self.finished, "clip_count": len(self.clips),
            "source_url": self.source.get("url", ""), "kind": self.source.get("kind", "clips"),
        }
        if full:
            data.update({"settings": self.settings, "clips": self.clips, "log": self.log[-200:],
                         "candidates": self.candidates if self.status == "review" else []})
        return data

    def save(self) -> None:
        try:
            data = self.to_dict()
            data["source"] = self.source
            data["candidates"] = self.candidates
            (self.dir / "job.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[tuple] = queue.Queue()
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
                            "log", "created", "finished", "candidates"):
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
        self.queue.put((job.id, None))
        return job

    def rerender_clip(self, job_id: str, index: int, edit: dict) -> bool:
        """Queue one clip of a finished job to be rendered again with edits."""
        job = self.jobs.get(job_id)
        if not job or job.status in ("queued", "running") or not 1 <= index <= len(job.clips):
            return False
        job.status, job.stage, job.progress, job.error = "queued", "render", 0.0, ""
        job.message = f"Waiting to re-render clip {index}"
        job.cancel_flag = False
        self.queue.put((job.id, {"index": index, "edit": edit}))
        return True

    def render_selected(self, job_id: str, picks: list[dict]) -> bool:
        """Continue a job that is waiting for the user's choice of moments."""
        job = self.jobs.get(job_id)
        if not job or job.status != "review" or not picks:
            return False
        job.status, job.error, job.cancel_flag = "queued", "", False
        job.message = "Waiting to render your selection"
        self.queue.put((job.id, {"render": picks}))
        return True

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.to_dict(full=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in ("queued", "running", "review"):
            return False
        job.cancel_flag = True
        if job.status in ("queued", "review"):
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
            job_id, task = self.queue.get()
            job = self.jobs.get(job_id)
            if not job or job.status != "queued":
                continue
            job.status = "running"
            try:
                paused = False
                if task and "render" in task:
                    self._run_selected(job, task["render"])
                elif task:
                    self._run_clip(job, task["index"], task["edit"])
                elif job.source.get("kind") == "story":
                    self._run_story(job)
                else:
                    paused = self._run(job) == "review"
                if paused:
                    job.status, job.stage = "review", "review"
                    job.message = "Choose the moments you want as clips"
                    job.save()
                    continue
                job.status, job.stage, job.progress = "done", "done", 1.0
                job.message = f"Done: {len(job.clips)} clips ready"
            except Exception as exc:
                if task and "index" in task and job.clips:
                    # The other clips are still fine: report the problem but keep the results.
                    job.status, job.stage, job.progress = "done", "done", 1.0
                    job.message = "Cancelled" if job.cancel_flag else f"Re-render failed: {exc}"
                    job.log.append(f"Re-render of clip {task['index']} failed: {exc}")
                elif job.cancel_flag:
                    job.status, job.message = "cancelled", "Cancelled"
                else:
                    job.status, job.error = "error", str(exc)
                    job.message = "Failed"
                    job.log.append(traceback.format_exc(limit=4))
                    config.applog(f"[job {job.id}] FAILED: {exc}\n{traceback.format_exc(limit=6)}")
            job.finished = time.time()
            job.save()

    def _source_dir(self, job: Job) -> Path:
        return config.JOBS_DIR / job.source["reuse"] if job.source.get("reuse") else job.dir

    def clip_words(self, job: Job, index: int, pad: int = 40) -> dict:
        """Words of one clip plus some context, for the editor."""
        plans = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        plan = plans[index - 1]
        tpath = self._source_dir(job) / f"transcript.{job.settings['whisper_model']}.json"
        words = json.loads(tpath.read_text(encoding="utf-8"))["words"]
        lo, hi = max(0, plan["w0"] - pad), min(len(words) - 1, plan["w1"] + pad)
        edits = plan.get("word_edits") or {}
        return {
            "w0": plan["w0"], "w1": plan["w1"], "hook": plan.get("hook", ""), "title": plan.get("title", ""),
            "hooks": plan.get("hooks") or [], "titles": plan.get("titles") or [],
            "description": plan.get("description", ""),
            "words": [{"i": i, "w": edits.get(str(i), words[i]["w"]), "orig": words[i]["w"],
                       "s": words[i]["s"], "e": words[i]["e"]} for i in range(lo, hi + 1)],
        }

    def _run_clip(self, job: Job, index: int, edit: dict) -> None:
        s = job.settings
        src_dir = self._source_dir(job)
        meta = json.loads((src_dir / "source.json").read_text(encoding="utf-8"))
        words = json.loads((src_dir / f"transcript.{s['whisper_model']}.json").read_text(encoding="utf-8"))["words"]
        plans = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        plan = plans[index - 1]

        last = len(words) - 1
        w0 = max(0, min(int(edit.get("w0", plan["w0"])), last))
        w1 = max(w0, min(int(edit.get("w1", plan["w1"])), last))
        if words[w1]["e"] - words[w0]["s"] > 200:
            raise RuntimeError("The clip would be longer than 200 seconds.")
        plan["w0"], plan["w1"] = w0, w1
        for key in ("hook", "title", "description"):
            if key in edit:
                plan[key] = str(edit[key])[:300]
        merged = dict(plan.get("word_edits") or {})
        for k, v in (edit.get("word_edits") or {}).items():
            if str(k).isdigit():
                merged[str(k)] = str(v)[:60]
        plan["word_edits"] = {k: v for k, v in merged.items() if v != words[int(k)]["w"]}
        (job.dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")

        old = job.clips[index - 1]
        version = int(old.get("version", 0)) + 1
        fonts = job.dir / "fonts"
        if not fonts.exists():
            shutil.copytree(config.FONTS_DIR, fonts)
        job.stage = "render"

        def update(p: float, msg: str) -> None:
            if job.cancel_flag:
                raise RuntimeError("Cancelled")
            job.progress, job.message = max(0.0, min(p, 1.0)), msg

        def log(msg: str) -> None:
            job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            config.applog(f"[job {job.id}] {msg}")

        try:
            result = render.render_clip(meta["path"], meta, plan, words, s, job.dir, index, old.get("style"),
                                        job.source.get("music"), update, lambda: job.cancel_flag, log, version)
        finally:
            shutil.rmtree(fonts, ignore_errors=True)
        for key in ("file", "thumb"):
            if old.get(key) and old[key] != result[key]:
                (job.dir / old[key]).unlink(missing_ok=True)
        job.clips[index - 1] = result
        _write_post_text(job)

    def _run_story(self, job: Job) -> None:
        """Narrated video from the user's own footage."""
        s = job.settings
        cancelled = lambda: job.cancel_flag
        offsets, acc = {}, 0.0
        for name, weight in STORY_STAGES:
            offsets[name] = (acc, weight)
            acc += weight

        def log(msg: str) -> None:
            job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            config.applog(f"[job {job.id}] {msg}")

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

        up = stage("footage")
        files = [f for f in job.source.get("files", []) if Path(f).exists()]
        if not files:
            raise RuntimeError("The uploaded footage is no longer available. Upload the clips again.")
        clips = []
        for i, path in enumerate(files):
            up(i / len(files), f"Looking through your footage ({i + 1} of {len(files)})")
            try:
                clips.append(footage.analyze_clip(path))
            except Exception as exc:
                log(f"Skipped {Path(path).name}: {exc}")
        clips = [c for c in clips if c["duration"] >= 1.0]
        if not clips:
            raise RuntimeError("None of the uploaded files could be read as video.")
        log(f"Footage: {len(clips)} clips, {sum(c['duration'] for c in clips):.0f} seconds in total.")

        up = stage("story")
        n = s["num_videos"]
        up(0.2, "Writing the story" if not s["story_text"].strip() else "Preparing your text")
        stories, origin = story.get_stories(n, s, log)
        job.engine = origin
        job.title = stories[0]["title"] if n == 1 else f"{n} narrated videos"
        (job.dir / "stories.json").write_text(json.dumps(stories, indent=2), encoding="utf-8")
        job.save()

        fonts = job.dir / "fonts"
        if not fonts.exists():
            shutil.copytree(config.FONTS_DIR, fonts)
        if s.get("logo_enabled") and not s.get("logo_path"):
            logo = config.WORKSPACE / "brand" / "logo.png"
            if logo.exists():
                shutil.copy(logo, job.dir / "logo.png")
                s["logo_path"] = str(job.dir / "logo.png")
        music = job.source.get("music")
        job.clips = []
        narrations: dict[str, tuple[str, float, list]] = {}
        for i, st in enumerate(stories, start=1):
            up = stage("voice")
            vshare = lambda p, msg, i=i: up((i - 1 + p) / n, msg)
            if st["script"] not in narrations:  # pasted text is voiced once and reused
                wav = job.dir / f"narration_{i:02d}.wav"
                dur = voice.synthesize(st["script"], s["voice"], s["voice_speed"], wav,
                                       lambda p, msg: vshare(p * 0.7, msg))
                vshare(0.7, "Timing the captions to the voice")
                heard = transcribe.transcribe(str(wav), s["whisper_model"], lambda p, msg: vshare(0.7 + p * 0.3, "Timing the captions to the voice"),
                                              cancelled, dur, s["speech_engine"])["words"]
                narrations[st["script"]] = (str(wav), dur, story.align(st["script"], heard, dur))
                words_per_sec = len(st["script"].split()) / max(dur, 1)
                log(f"Voiceover {i}: {dur:.1f}s, {len(st['script'].split())} words ({words_per_sec:.1f} per second).")
            wav_path, dur, words = narrations[st["script"]]
            st["engine"] = origin

            up = stage("render")
            if s["caption_style"] == "none":
                style = None
            elif s["caption_style"] == "mix":
                style = captions.MIX_ORDER[(i - 1) % len(captions.MIX_ORDER)]
            else:
                style = s["caption_style"]
            result = footage.render_story(st, words, wav_path, dur, clips, s, job.dir, i, style, music,
                                          lambda p, msg, i=i: up((i - 1 + p) / n, f"{msg} ({i} of {n})"),
                                          cancelled, log)
            job.clips.append(result)
            job.save()
        shutil.rmtree(fonts, ignore_errors=True)
        _write_post_text(job)

    def _helpers(self, job: Job):
        """(log, stage) closures shared by the clip pipeline steps."""
        offsets, acc = {}, 0.0
        for name, weight in STAGES:
            offsets[name] = (acc, weight)
            acc += weight

        def log(msg: str) -> None:
            job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            config.applog(f"[job {job.id}] {msg}")

        def stage(name: str):
            base, weight = offsets[name]
            job.stage = name

            def update(p: float, msg: str) -> None:
                if job.cancel_flag:
                    raise RuntimeError("Cancelled")
                job.progress = base + weight * max(0.0, min(p, 1.0))
                job.message = msg
            return update
        return log, stage

    def _run_selected(self, job: Job, picks: list[dict]) -> None:
        """Render the moments the user ticked in the review step."""
        log, stage = self._helpers(job)
        src_dir = self._source_dir(job)
        meta = json.loads((src_dir / "source.json").read_text(encoding="utf-8"))
        words = json.loads((src_dir / f"transcript.{job.settings['whisper_model']}.json").read_text(encoding="utf-8"))["words"]
        candidates = json.loads((job.dir / "candidates.json").read_text(encoding="utf-8"))
        plans = []
        for pick in picks:
            idx = int(pick.get("index", 0))
            if not 1 <= idx <= len(candidates):
                continue
            plan = dict(candidates[idx - 1])
            for key in ("hook", "title"):
                if str(pick.get(key) or "").strip():
                    plan[key] = str(pick[key]).strip()[:200]
            if "cold_open" in pick:
                plan["use_teaser"] = bool(pick["cold_open"])
            plans.append(plan)
        if not plans:
            raise RuntimeError("No moments were selected.")
        log(f"You chose {len(plans)} of {len(candidates)} proposed moments.")
        plans = _with_variants(plans, job.settings)
        (job.dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
        job.candidates = []
        self._render_plans(job, meta, meta, words, plans, log, stage)

    def _render_plans(self, job: Job, meta: dict, info: dict, words: list[dict], plans: list[dict], log, stage) -> None:
        s = job.settings
        cancelled = lambda: job.cancel_flag
        up = stage("render")
        fonts = job.dir / "fonts"
        if not fonts.exists():
            shutil.copytree(config.FONTS_DIR, fonts)
        music = job.source.get("music")
        job.clips = []
        if s.get("logo_enabled") and not s.get("logo_path"):
            logo = config.WORKSPACE / "brand" / "logo.png"
            if logo.exists():  # keep a copy so later re-renders use the same logo
                shutil.copy(logo, job.dir / "logo.png")
                s["logo_path"] = str(job.dir / "logo.png")
        for i, plan in enumerate(plans, start=1):
            order = captions.MIX_ORDER
            shift = int(plan.get("style_shift", 0))
            if s["caption_style"] == "none":
                style = None
            elif s["caption_style"] == "mix":
                style = order[(int(plan.get("ordinal", i)) - 1 + shift) % len(order)]
            else:  # version B of a fixed style uses the next style, so the two versions differ
                style = order[(order.index(s["caption_style"]) + shift) % len(order)] if shift else s["caption_style"]
            n = len(plans)
            result = render.render_clip(
                meta["path"], info, plan, words, s, job.dir, i, style, music,
                lambda p, msg, i=i: up((i - 1 + p) / n, f"{msg} ({i} of {n})"), cancelled, log)
            job.clips.append(result)
            job.save()
        shutil.rmtree(fonts, ignore_errors=True)
        _write_post_text(job)

    def _run(self, job: Job) -> str | None:
        s = job.settings
        cancelled = lambda: job.cancel_flag
        log, stage = self._helpers(job)

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
            transcript = transcribe.transcribe(meta["path"], s["whisper_model"], up, cancelled, info["duration"],
                                               s["speech_engine"])
            tpath.write_text(json.dumps(transcript), encoding="utf-8")
            log(f"Transcribed {len(transcript['words'])} words in {len(transcript['sentences'])} sentences.")
        words, sentences = transcript["words"], transcript["sentences"]

        # 3. Find the moments ----------------------------------------------
        up = stage("analyze")
        ai = llm.current()
        up(0.1, "Finding the best moments" + (f" with {ai['name']}" if s["use_ai"] and ai["ready"] else ""))
        want = s["num_clips"]
        limit = min(max(want * 2, want + 3), 14) if s["review_picks"] else want
        learned = tracker.summary_for_ai()
        if learned:
            log("Passing your tracked results to the AI.")
        plans, engine = analyze.analyze(sentences, meta["title"], want, s["min_len"], s["max_len"],
                                        s["use_ai"], (s["instructions"] + "\n" + learned).strip(), log, limit)
        job.engine = engine
        log(f"Found {len(plans)} moments using {'offline scoring' if engine == 'offline' else engine}.")

        if s["review_picks"] and len(plans) > 1:
            (job.dir / "candidates.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
            job.candidates = [{
                "index": i, "title": p["title"], "titles": p.get("titles") or [p["title"]],
                "hook": p.get("hook", ""), "hooks": p.get("hooks") or [],
                "score": p.get("score", 0), "scores": p.get("scores") or {}, "reason": p.get("reason", ""),
                "start": round(p["start"], 1), "end": round(p["end"], 1), "duration": round(p["end"] - p["start"], 1),
                "text": " ".join(w["w"] for w in words[p["w0"]:p["w1"] + 1])[:420],
                "teaser": " ".join(w["w"] for w in words[p["teaser_w0"]:p["teaser_w1"] + 1])[:120]
                if s["cold_open"] and p.get("teaser_w0") is not None else "",
                "selected": i <= want,
            } for i, p in enumerate(plans, start=1)]
            job.progress = 0.48
            return "review"

        plans = _with_variants(plans, s)
        (job.dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
        self._render_plans(job, meta, info, words, plans, log, stage)
        return None


def _with_variants(plans: list[dict], settings: dict) -> list[dict]:
    """Number the plans and, for A/B testing, add a second version of each with a
    different hook and caption style."""
    out = []
    for ordinal, plan in enumerate(plans, start=1):
        base = dict(plan, ordinal=ordinal)
        if not settings.get("ab_versions"):
            out.append(base)
            continue
        hooks = [h["text"] for h in plan.get("hooks") or [] if h.get("text") and h["text"] != plan.get("hook")]
        out.append(dict(base, variant="A"))
        out.append(dict(base, variant="B", style_shift=2, hook=hooks[0] if hooks else plan.get("hook", ""),
                        use_teaser=not plan.get("use_teaser", True) if plan.get("teaser_w0") is not None else False))
    return out


def _write_post_text(job: Job) -> None:
    lines = [f"{job.title}", ""]
    for c in job.clips:
        tags = " ".join("#" + h for h in c["hashtags"])
        lines += [f"=== {c['file']}  ({c['duration']}s) ===", f"Title: {c['title']}",
                  f"Hook: {c['hook']}", f"Caption: {c['description']} {tags}".rstrip()]
        lines += [f"Script: {c['transcript']}", ""] if c.get("kind") == "story" else [f"Score: {c['score']}. {c['reason']}", ""]
    (job.dir / "post_text.txt").write_text("\n".join(lines), encoding="utf-8")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
