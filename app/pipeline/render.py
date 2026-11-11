"""Cut, reframe, edit and caption one clip with a single ffmpeg pass."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from .. import config
from . import captions, reframe

ASPECTS = {"9:16": (9, 16), "4:5": (4, 5), "1:1": (1, 1), "16:9": (16, 9)}
RESOLUTIONS = {"720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160}
QUALITY = {"fast": ("veryfast", 23), "balanced": ("medium", 20), "high": ("slow", 17)}


def output_size(aspect: str, resolution: str) -> tuple[int, int]:
    aw, ah = ASPECTS.get(aspect, (9, 16))
    short = RESOLUTIONS.get(resolution, 1080)
    if aw <= ah:
        w, h = short, short * ah / aw
    else:
        w, h = short * aw / ah, short
    return int(round(w)) // 2 * 2, int(round(h)) // 2 * 2


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:limit].strip("-")
    return slug or "clip"


def _keep_segments(words: list[dict], dur: float, fps: int, max_gap: float) -> list[tuple[int, int]]:
    """Frame ranges [a, b) to keep so that pauses longer than max_gap are cut out."""
    if not words:
        return [(0, int(dur * fps))]
    runs = [[words[0]["s"], words[0]["e"]]]
    for w in words[1:]:
        if w["s"] - runs[-1][1] > max_gap:
            runs.append([w["s"], w["e"]])
        else:
            runs[-1][1] = max(runs[-1][1], w["e"])
    segs = []
    for i, (s, e) in enumerate(runs):
        lo = 0.0 if i == 0 else max(s - 0.1, (runs[i - 1][1] + s) / 2)
        hi = dur if i == len(runs) - 1 else min(e + 0.14, (e + runs[i + 1][0]) / 2)
        a, b = int(round(lo * fps)), int(round(hi * fps))
        if b > a:
            segs.append((a, b))
    return segs


def _remap(words: list[dict], segs: list[tuple[int, int]], fps: int) -> list[dict]:
    """Move word times from the uncut clip timeline onto the jump-cut timeline."""
    bounds = []
    offset = 0.0
    for a, b in segs:
        bounds.append((a / fps, b / fps, offset))
        offset += (b - a) / fps

    def conv(t: float) -> float:
        for lo, hi, off in bounds:
            if t < lo:
                return off
            if t <= hi:
                return off + (t - lo)
        return offset

    out = []
    for w in words:
        s, e = conv(w["s"]), conv(w["e"])
        out.append({"w": w["w"], "s": s, "e": max(e, s + 0.04)})
    return out


def _zoom_windows(words: list[dict], dur: float, hook_seconds: float) -> list[tuple[float, float]]:
    """Alternate punch-ins on sentence boundaries, a few seconds apart."""
    beats = [0.0]
    for i, w in enumerate(words[:-1]):
        if re.search(r"[.!?]$", w["w"]) and words[i + 1]["s"] - beats[-1] >= 2.8:
            beats.append(words[i + 1]["s"])
    beats.append(dur)
    windows = []
    zoomed = False
    for a, b in zip(beats, beats[1:]):
        if zoomed and a >= hook_seconds * 0.5 and b - a >= 1.2:
            windows.append((a, b))
        zoomed = not zoomed
    return windows


def render_clip(src: str, info: dict, clip: dict, all_words: list[dict], settings: dict, job_dir: Path,
                index: int, style_key: str | None, music: str | None,
                progress: Callable[[float, str], None], cancelled: Callable[[], bool],
                log: Callable[[str], None]) -> dict:
    fps = int(settings["fps"])
    width, height = output_size(settings["aspect"], settings["resolution"])
    aw, ah = ASPECTS[settings["aspect"]]

    # Clip boundaries: a little air around the first and last word, never into neighbours.
    w0, w1 = clip["w0"], clip["w1"]
    prev_end = all_words[w0 - 1]["e"] if w0 > 0 else 0.0
    next_start = all_words[w1 + 1]["s"] if w1 + 1 < len(all_words) else info["duration"] or 1e9
    start = max(0.0, prev_end, all_words[w0]["s"] - 0.15)
    end = min(info["duration"] or 1e9, all_words[w1]["e"] + 0.35, max(all_words[w1]["e"] + 0.08, next_start))
    src_dur = end - start
    words = [{"w": w["w"], "s": max(0.0, w["s"] - start), "e": min(src_dur, w["e"] - start)}
             for w in all_words[w0:w1 + 1]]

    name = f"{index:02d}_{slugify(clip['title'])}"
    progress(0.02, f"Clip {index}: tracking faces")
    plan = reframe.plan_crop(src, start, src_dur, info["width"], info["height"], aw, ah, settings["reframe"],
                             lambda p: progress(0.02 + p * 0.18, f"Clip {index}: tracking faces"))

    segs = [(0, int(round(src_dur * fps)))]
    if settings["remove_silence"]:
        cand = _keep_segments(words, src_dur, fps, float(settings["silence_gap"]))
        saved = src_dur - sum(b - a for a, b in cand) / fps
        if len(cand) > 1 or saved > 0.3:
            segs = cand
            words = _remap(words, segs, fps)
    out_dur = sum(b - a for a, b in segs) / fps
    cut = len(segs) > 1 or segs[0][0] > 0 or abs(out_dur - src_dur) > 0.05

    hook_seconds = float(settings["hook_seconds"]) if settings["hook_style"] != "none" else 0.0
    ass = captions.build_ass(
        words, out_dur, width, height, style_key, clip.get("emphasis_words", []),
        clip.get("hook", ""), settings["hook_style"], hook_seconds,
        settings["caption_position"], float(settings["caption_scale"]))
    ass_name = f"{name}.ass"
    (job_dir / ass_name).write_text(ass, encoding="utf-8")

    # ---- video chain -------------------------------------------------------
    v = [f"fps={fps}"]
    if settings["reframe"] == "blur":
        graph = (f"[0:v]fps={fps},split[bga][fga];"
                 f"[bga]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
                 f"boxblur=luma_radius={max(8, width // 40)}:luma_power=2,eq=brightness=-0.08[bg];"
                 f"[fga]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0];[v0]")
        v = []
    else:
        graph = "[0:v]"
        v.append(f"crop={plan['cw']}:{plan['ch']}:x='{plan['x_expr']}':y={plan['y']}")
    # Half-frame offsets keep the comparisons away from exact frame timestamps.
    sel = "+".join(f"gte(t,{(a - 0.5) / fps:.5f})*lt(t,{(b - 0.5) / fps:.5f})" for a, b in segs)
    if cut:
        v += [f"select='{sel}'", f"setpts=N/({fps}*TB)"]
    v += [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
    if settings["zooms"]:
        windows = _zoom_windows(words, out_dur, hook_seconds)
        if windows:
            cond = "+".join(f"between(in_time,{a:.3f},{b - 0.001:.3f})" for a, b in windows)
            z = float(settings["zoom_amount"])
            v.append(f"zoompan=z='if({cond},{z:.3f},1)':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*0.38'"
                     f":d=1:s={width}x{height}:fps={fps}")
    v.append("format=yuv420p")
    graph += ",".join(v)
    if settings["progress_bar"]:
        bar_h = max(4, int(height * 0.007) // 2 * 2)
        color = captions.STYLES.get(style_key or "", {}).get("active", "#FFE100").replace("#", "0x")
        if color.upper() == "0XFFFFFF":
            color = "0xFFE100"
        graph += (f"[v1];color=c={color}:s={width}x{bar_h}:r={fps}[bar];"
                  f"[v1][bar]overlay=x='-w+w*t/{out_dur:.3f}':y=H-h:shortest=1")
    graph += f",ass={ass_name}:fontsdir=fonts[v];"

    # ---- audio chain -------------------------------------------------------
    inputs = ["-ss", f"{start:.3f}", "-t", f"{src_dur:.3f}", "-i", src]
    if info.get("has_audio", True):
        a = ["[0:a]aresample=48000"]
        if cut:
            a += [f"aselect='{sel}'", "asetpts=N/SR/TB"]
    else:
        inputs += ["-f", "lavfi", "-t", f"{out_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        a = ["[1:a]anull"]
    if settings["normalize_audio"] and info.get("has_audio", True):
        a += ["loudnorm=I=-14:TP=-1.5:LRA=11", "aresample=48000"]
    graph += ",".join(a)
    if music:
        music_index = 2 if not info.get("has_audio", True) else 1
        inputs += ["-stream_loop", "-1", "-i", music]
        vol = float(settings["music_volume"])
        graph += (f"[a0];[{music_index}:a]aresample=48000,volume={vol:.3f}[mus];"
                  f"[a0][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                  f"afade=t=out:st={max(0.0, out_dur - 0.5):.3f}:d=0.5")
    graph += "[a]"

    preset, crf = QUALITY.get(settings["quality"], QUALITY["balanced"])
    out_name = f"{name}.mp4"
    cmd = [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{out_dur:.3f}",
           "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_name]

    proc = subprocess.Popen(cmd, cwd=str(job_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace")
    try:
        for line in proc.stdout:
            if cancelled():
                proc.kill()
                raise RuntimeError("Cancelled")
            if line.startswith("out_time_ms=") or line.startswith("out_time_us="):
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                progress(0.2 + 0.78 * min(done / max(out_dur, 0.1), 1.0), f"Clip {index}: rendering")
        err = proc.stderr.read()
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if code != 0 or not (job_dir / out_name).exists():
        raise RuntimeError(f"ffmpeg failed while rendering clip {index}: {err.strip()[-600:]}")

    thumb = f"{name}.jpg"
    subprocess.run([config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss",
                    f"{min(1.0, out_dur / 2):.2f}", "-i", out_name, "-frames:v", "1", "-vf", "scale=360:-2", thumb],
                   cwd=str(job_dir), capture_output=True)
    (job_dir / ass_name).unlink(missing_ok=True)
    log(f"Clip {index} rendered: {out_dur:.1f}s, style {style_key or 'no captions'}, "
        f"{len(segs) - 1} silence cuts, {plan['pieces']} camera positions.")
    return {
        "index": index, "file": out_name, "thumb": thumb, "duration": round(out_dur, 2),
        "source_start": round(start, 2), "source_end": round(end, 2), "style": style_key,
        "width": width, "height": height,
        "title": clip["title"], "hook": clip.get("hook", ""), "description": clip.get("description", ""),
        "hashtags": clip.get("hashtags", []), "score": clip.get("score", 0), "reason": clip.get("reason", ""),
        "engine": clip.get("engine", "offline"),
        "transcript": " ".join(w["w"] for w in all_words[w0:w1 + 1]),
    }
