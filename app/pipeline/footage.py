"""Build a narrated video out of the user's own footage: pick good stretches,
cut them to the voiceover, and finish with captions and music."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config
from . import captions, reframe, render

LEAD_IN = 0.6    # footage before the voice starts
TAIL = 1.3       # footage after the last word
MIN_SHOT, MAX_SHOT = 2.2, 5.0
AW, AH = 160, 90


def analyze_clip(path: str) -> dict:
    """Score a clip second by second so shaky, dark or blurry stretches can be avoided.
    Only keyframes are decoded, which keeps long 4K files fast."""
    import cv2

    info = reframe.probe(path)
    cmd = [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-i", path, "-an",
           "-vf", f"scale={AW}:{AH}", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    frames = np.frombuffer(proc.stdout, np.uint8)
    n = len(frames) // (AW * AH)
    dur = info["duration"]
    scores: list[float] = []
    if n >= 2 and dur > 0:
        frames = frames[:n * AW * AH].reshape(n, AH, AW)
        sharp = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in frames])
        bright = frames.reshape(n, -1).mean(axis=1)
        motion = np.concatenate([[0.0], np.abs(np.diff(frames.astype(np.int16), axis=0)).reshape(n - 1, -1).mean(axis=1)])
        steady = np.abs(motion - np.median(motion))            # sudden changes = shake or whip pans
        q = np.log1p(sharp) / max(np.log1p(sharp).max(), 1e-6)  # 0..1 within the clip
        q -= np.clip(steady / 25.0, 0, 0.6)
        q -= np.where((bright < 28) | (bright > 235), 0.5, 0.0)
        # Spread keyframe scores over whole seconds.
        times = np.arange(n) * dur / n
        for sec in range(int(dur) + 1):
            near = q[(times >= sec - 0.5) & (times < sec + 1.5)]
            scores.append(float(near.mean()) if len(near) else float(q.mean()))
    info.update({"path": path, "scores": scores, "quality": float(np.mean(scores)) if scores else 0.0})
    return info


def best_window(clip: dict, need: float, used: list[tuple[float, float]]) -> tuple[float, float]:
    """(start, available) of the best unused stretch of this clip."""
    dur = clip["duration"]
    margin = 0.5 if dur > need + 1.5 else 0.0
    usable = dur - 2 * margin
    if usable <= need:
        return margin, max(usable, 0.2)
    best, best_score = margin, -1e9
    step = 0.5
    t = margin
    while t + need <= dur - margin + 1e-6:
        a, b = int(t), max(int(t) + 1, int(t + need) + 1)
        seg = clip["scores"][a:b] or [clip["quality"]]
        score = sum(seg) / len(seg)
        overlap = sum(max(0.0, min(t + need, e) - max(t, s)) for s, e in used)
        score -= 2.0 * overlap / need          # prefer footage that has not been shown yet
        if score > best_score:
            best, best_score = t, score
        t += step
    return best, need


def plan_shots(words: list[dict], total: float) -> list[tuple[float, float]]:
    """Cut on sentence starts, keeping every shot between MIN_SHOT and MAX_SHOT seconds."""
    cuts = [0.0]
    for i, w in enumerate(words[:-1]):
        if re.search(r"[.!?…]$", w["w"]):
            t = max(0.0, words[i + 1]["s"] - 0.12)
            if t - cuts[-1] >= MIN_SHOT and total - t >= MIN_SHOT:
                cuts.append(t)
    cuts.append(total)
    shots: list[tuple[float, float]] = []
    for a, b in zip(cuts, cuts[1:]):
        parts = max(1, int(np.ceil((b - a) / MAX_SHOT)))
        for k in range(parts):
            shots.append((a + (b - a) * k / parts, a + (b - a) * (k + 1) / parts))
    return shots


def _run(cmd: list[str], cwd: Path, what: str) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while {what}: {proc.stderr.strip()[-500:]}")


AMBIENT = ("sine=f=110:r=48000,volume=0.5[p1];sine=f=164.81:r=48000,volume=0.35[p2];"
           "sine=f=220:r=48000,volume=0.3[p3];sine=f=329.63:r=48000,volume=0.16[p4];"
           "[p1][p2][p3][p4]amix=inputs=4:normalize=0,tremolo=f=0.12:d=0.5,lowpass=f=900,"
           "aecho=0.8:0.7:900|1400:0.45|0.3,volume=0.9")


def render_story(story: dict, words: list[dict], narration: str, narration_dur: float, clips: list[dict],
                 settings: dict, job_dir: Path, index: int, style_key: str | None, music: str | None,
                 progress: Callable[[float, str], None], cancelled: Callable[[], bool],
                 log: Callable[[str], None]) -> dict:
    # Use the frame rate most of the footage was shot at, so frames are not repeated.
    by_rate: dict[float, float] = {}
    for c in clips:
        by_rate[round(c["fps"], 2)] = by_rate.get(round(c["fps"], 2), 0.0) + c["duration"]
    fps, fps_s = render.resolve_fps(settings["fps"], max(by_rate, key=by_rate.get))
    width, height = render.output_size(settings["aspect"], settings["resolution"])
    aw, ah = render.ASPECTS[settings["aspect"]]
    total = LEAD_IN + narration_dur + TAIL
    timed = [{"w": w["w"], "s": w["s"] + LEAD_IN, "e": w["e"] + LEAD_IN} for w in words]
    shots = plan_shots(timed, total)
    fade = 0.4 if settings["transition"] == "crossfade" and len(shots) > 1 else 0.0
    name = f"{index:02d}_{render.slugify(story['title'])}"

    # ---- 1. cut each shot from the footage --------------------------------
    used: dict[int, list[tuple[float, float]]] = {}
    order = sorted(range(len(clips)), key=lambda i: -clips[i]["quality"]) if settings["footage_order"] == "best" \
        else list(range(len(clips)))
    order = order[(index - 1) % len(order):] + order[:(index - 1) % len(order)]  # vary between videos
    segs: list[str] = []
    for k, (a, b) in enumerate(shots):
        if cancelled():
            raise RuntimeError("Cancelled")
        need = (b - a) + (fade if k < len(shots) - 1 else 0.0) + 0.1
        ci = order[k % len(order)]
        # Slowing a clip down repeats frames and stutters, so prefer one that is long enough.
        for j in range(len(order)):
            cand = order[(k + j) % len(order)]
            if clips[cand]["duration"] - 1.0 >= need:
                ci = cand
                break
        clip = clips[ci]
        start, avail = best_window(clip, need, used.setdefault(ci, []))
        used[ci].append((start, start + avail))
        slow = max(1.0, min(need / max(avail, 0.1), 3.0))  # stretch clips that are too short
        vf = [f"fps={fps_s}"]
        if slow > 1.01:
            vf.append(f"setpts={slow:.4f}*PTS")
        if settings["reframe"] == "blur":
            graph = (f"[0:v]{','.join(vf)},split[bga][fga];"
                     f"[bga]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
                     f"boxblur=luma_radius={max(8, width // 40)}:luma_power=2,eq=brightness=-0.08[bg];"
                     f"[fga]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                     f"[bg][fg]overlay=(W-w)/2:(H-h)/2,")
        else:
            cw, ch = reframe.crop_size(clip["width"], clip["height"], aw, ah)
            graph = f"[0:v]{','.join(vf)},crop={cw}:{ch},scale={width}:{height}:flags=lanczos,"
        graph += f"setsar=1,tpad=stop_mode=clone:stop_duration=3,trim=duration={need:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
        seg = f"{name}_s{k:02d}.mp4"

        def build_seg(codec: list[str], graph=graph, start=start, avail=avail, clip=clip, seg=seg) -> list[str]:
            return [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
                    "-t", f"{avail + 0.2:.3f}", "-i", clip["path"], "-an", "-filter_complex", graph, "-map", "[v]",
                    *codec, "-pix_fmt", "yuv420p", "-r", fps_s, "-progress", "pipe:1", "-nostats", seg]

        try:
            render.run_encode(build_seg, settings, job_dir, need, lambda p: None, cancelled, log, fast=True)
        except RuntimeError as exc:
            if "Cancelled" in str(exc):
                raise
            raise RuntimeError(f"Cutting shot {k + 1} failed. {exc}") from exc
        segs.append(seg)
        progress(0.55 * (k + 1) / len(shots), f"Video {index}: cutting shot {k + 1} of {len(shots)}")

    # ---- 2. join, caption, mix -------------------------------------------
    hook_seconds = float(settings["hook_seconds"]) if settings["hook_style"] != "none" else 0.0
    hook_text = story.get("hook", "") if hook_seconds else ""
    brand = {"color": settings.get("brand_color", ""), "font": settings.get("brand_font", ""),
             "handle": settings.get("handle_text", "")}
    ass = captions.build_ass(timed, total, width, height, style_key, story.get("emphasis_words", []), hook_text,
                             settings["hook_style"], hook_seconds, settings["caption_position"],
                             float(settings["caption_scale"]), brand)
    ass_name = f"{name}.ass"
    (job_dir / ass_name).write_text(ass, encoding="utf-8")

    inputs: list[str] = []
    n_inputs = 0

    def add_input(*args: str) -> int:
        nonlocal n_inputs
        inputs.extend(args)
        n_inputs += 1
        return n_inputs - 1

    for seg in segs:
        add_input("-i", seg)
    chains: list[str] = []
    if len(segs) == 1:
        chains.append("[0:v]null[joined]")
    elif fade:
        prev = "[0:v]"
        for k in range(1, len(segs)):
            out = "[joined]" if k == len(segs) - 1 else f"[xf{k}]"
            chains.append(f"{prev}[{k}:v]xfade=transition=fade:duration={fade}:offset={shots[k][0]:.3f}{out}")
            prev = out
    else:
        chains.append("".join(f"[{k}:v]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[joined]")
    chains.append(f"[joined]fade=t=in:d=0.5,fade=t=out:st={total - 0.6:.3f}:d=0.6[base]")
    cur, cues = render.add_overlays(chains, "[base]", add_input, settings, style_key, story, timed, total,
                                    width, height, fps_s, ass_name, brand)
    chains.append(f"{cur}null[v]")

    vi = add_input("-i", narration)
    voice = (f"[{vi}:a]aresample=48000,adelay={int(LEAD_IN * 1000)}:all=1,loudnorm=I=-15:TP=-1.5:LRA=9,"
             f"aresample=48000,apad=whole_dur={total:.3f}")
    bed = None
    if music:
        mi = add_input("-stream_loop", "-1", "-i", music)
        bed = f"[{mi}:a]aresample=48000,aformat=channel_layouts=stereo"
    elif settings["ambient_pad"]:
        chains.append(AMBIENT + "[amb]")
        bed = "[amb]aformat=channel_layouts=stereo"
    if bed:
        vol = float(settings["music_volume"]) * 2.2  # the bed sits under a voice, so it can be louder than under speech video
        chains.append(voice + ",aformat=channel_layouts=stereo,asplit[vo][vk]")
        chains.append(f"{bed},atrim=duration={total:.3f},volume={vol:.3f},afade=t=in:d=0.8,"
                      f"afade=t=out:st={max(0.0, total - 1.2):.3f}:d=1.2[bedraw]")
        # Duck the music whenever the narrator speaks.
        chains.append("[bedraw][vk]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=600[duck]")
        chains.append("[vo][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[a]")
    else:
        chains.append(voice + "[a]")

    out_name = f"{name}.mp4"

    def build(codec: list[str]) -> list[str]:
        return [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
                "-filter_complex", ";".join(chains), "-map", "[v]", "-map", "[a]",
                *codec, "-pix_fmt", "yuv420p", "-r", fps_s,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
                "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_name]

    try:
        render.run_encode(build, settings, job_dir, total,
                          lambda p: progress(0.55 + 0.44 * p, f"Video {index}: adding voice and captions"),
                          cancelled, log)
    except RuntimeError as exc:
        if "Cancelled" in str(exc):
            raise
        raise RuntimeError(f"Finishing video {index} failed. {exc}") from exc
    finally:
        for seg in segs:
            (job_dir / seg).unlink(missing_ok=True)
        (job_dir / ass_name).unlink(missing_ok=True)
    if not (job_dir / out_name).exists():
        raise RuntimeError(f"Finishing video {index} produced no file.")

    (job_dir / f"{name}.srt").write_text(captions.srt(timed), encoding="utf-8")
    thumb = f"{name}.jpg"
    subprocess.run([config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", "1.5", "-i", out_name,
                    "-frames:v", "1", "-vf", "scale=360:-2", thumb], cwd=str(job_dir), capture_output=True)
    log(f"Video {index} rendered: {total:.1f}s, {len(shots)} shots from {len(used)} clips, "
        f"style {style_key or 'no captions'}, {len(cues)} emojis.")
    return {
        "index": index, "kind": "story", "file": out_name, "thumb": thumb, "duration": round(total, 2),
        "source_start": 0, "source_end": 0, "style": style_key, "width": width, "height": height, "fps": round(fps, 2),
        "title": story["title"], "hook": story.get("hook", ""), "description": story.get("description", ""),
        "hashtags": story.get("hashtags", []), "score": 0, "reason": "", "scores": {},
        "engine": story.get("engine", ""), "transcript": story["script"], "srt": f"{name}.srt",
    }
