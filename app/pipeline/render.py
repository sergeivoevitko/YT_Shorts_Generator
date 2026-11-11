"""Cut, reframe, edit and caption one clip with a single ffmpeg pass."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from .. import config
from . import captions, emoji, reframe

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


FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mmm", "eh"}


def _is_filler(word: str) -> bool:
    return re.sub(r"[^a-z]", "", word.lower()) in FILLERS


def _keep_segments(words: list[dict], dur: float, fps: int, max_gap: float) -> list[tuple[int, int]]:
    """Frame ranges [a, b) to keep. Pauses longer than max_gap and words flagged
    "cut" (filler words) are removed."""
    kept = [w for w in words if not w.get("cut")]
    if not kept:
        return [(0, int(dur * fps))]
    cuts = [(w["s"], w["e"]) for w in words if w.get("cut")]

    def cut_between(a: float, b: float) -> tuple[float, float] | None:
        inside = [c for c in cuts if c[0] >= a - 0.01 and c[1] <= b + 0.01]
        return (min(c[0] for c in inside), max(c[1] for c in inside)) if inside else None

    runs = [[kept[0]["s"], kept[0]["e"]]]
    for w in kept[1:]:
        if w["s"] - runs[-1][1] > max_gap or cut_between(runs[-1][1], w["s"]):
            runs.append([w["s"], w["e"]])
        else:
            runs[-1][1] = max(runs[-1][1], w["e"])
    segs = []
    for i, (s, e) in enumerate(runs):
        before = cut_between(0.0 if i == 0 else runs[i - 1][1], s)
        after = cut_between(e, dur if i == len(runs) - 1 else runs[i + 1][0])
        if before:
            lo = max(s - 0.06, before[1])
        else:
            lo = 0.0 if i == 0 else max(s - 0.1, (runs[i - 1][1] + s) / 2)
        if after:
            hi = min(e + 0.08, after[0])
        else:
            hi = dur if i == len(runs) - 1 else min(e + 0.14, (e + runs[i + 1][0]) / 2)
        a, b = int(round(lo * fps)), int(round(hi * fps))
        if b > a:
            segs.append((a, b))
    return segs or [(0, int(dur * fps))]


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
    if not words:
        return []
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


def _make_sfx(job_dir: Path) -> tuple[str, str]:
    """Synthesize a whoosh and a pop once per job (no audio assets needed)."""
    recipes = {
        "sfx_whoosh.wav": ("anoisesrc=d=0.42:c=pink:a=0.9:r=48000", "highpass=f=700,lowpass=f=7000,"
                           "afade=t=in:d=0.26:curve=exp,afade=t=out:st=0.26:d=0.16"),
        "sfx_pop.wav": ("sine=f=660:d=0.11:r=48000", "asetrate=48000*1.25,aresample=48000,"
                        "afade=t=out:st=0.015:d=0.08,volume=0.8"),
    }
    for name, (src, flt) in recipes.items():
        if not (job_dir / name).exists():
            subprocess.run([config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src,
                            "-af", flt, "-ac", "2", name], cwd=str(job_dir), capture_output=True)
    return "sfx_whoosh.wav", "sfx_pop.wav"


def _emoji_cues(clip: dict, words: list[dict], dur: float) -> list[tuple[float, float, str]]:
    """(start, end, png path) for each emoji that can be placed on a spoken word."""
    cues = clip.get("emojis") or emoji.offline_cues([w["w"] for w in words])
    norm = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())
    out: list[tuple[float, float, str]] = []
    last = -10.0
    for cue in cues[:8]:
        target = norm(cue.get("word", ""))
        hit = next((w for w in words if norm(w["w"]) == target and w["s"] >= last + 1.8), None) if target else None
        if not hit or hit["s"] > dur - 0.6:
            continue
        png = emoji.png_path(str(cue.get("emoji", "")))
        if png:
            out.append((hit["s"], min(dur, hit["s"] + 1.3), png))
            last = hit["s"]
    return sorted(out)


def render_clip(src: str, info: dict, clip: dict, all_words: list[dict], settings: dict, job_dir: Path,
                index: int, style_key: str | None, music: str | None,
                progress: Callable[[float, str], None], cancelled: Callable[[], bool],
                log: Callable[[str], None], version: int = 0) -> dict:
    fps = int(settings["fps"])
    width, height = output_size(settings["aspect"], settings["resolution"])
    aw, ah = ASPECTS[settings["aspect"]]
    portrait = height >= width

    # Clip boundaries: a little air around the first and last word, never into neighbours.
    w0, w1 = clip["w0"], clip["w1"]
    prev_end = all_words[w0 - 1]["e"] if w0 > 0 else 0.0
    next_start = all_words[w1 + 1]["s"] if w1 + 1 < len(all_words) else info["duration"] or 1e9
    start = max(0.0, prev_end, all_words[w0]["s"] - 0.15)
    end = min(info["duration"] or 1e9, all_words[w1]["e"] + 0.35, max(all_words[w1]["e"] + 0.08, next_start))
    src_dur = end - start
    edits = clip.get("word_edits") or {}
    words = []
    for i in range(w0, w1 + 1):
        w = all_words[i]
        text = edits.get(str(i), w["w"])
        words.append({"w": text, "s": max(0.0, w["s"] - start), "e": min(src_dur, w["e"] - start),
                      "cut": bool(settings["remove_fillers"]) and _is_filler(w["w"]), "hidden": not text.strip()})

    name = f"{index:02d}_{slugify(clip['title'])}" + (f"_v{version}" if version else "")
    progress(0.02, f"Clip {index}: tracking faces")
    plan = reframe.plan_crop(src, start, src_dur, info["width"], info["height"], aw, ah, settings["reframe"],
                             lambda p: progress(0.02 + p * 0.18, f"Clip {index}: tracking faces"))

    segs = [(0, int(round(src_dur * fps)))]
    fillers = sum(1 for w in words if w["cut"])
    if settings["remove_silence"] or fillers:
        gap = float(settings["silence_gap"]) if settings["remove_silence"] else 1e9
        cand = _keep_segments(words, src_dur, fps, gap)
        saved = src_dur - sum(b - a for a, b in cand) / fps
        if len(cand) > 1 or saved > 0.3:
            segs = cand
    out_dur = sum(b - a for a, b in segs) / fps
    cut = len(segs) > 1 or segs[0][0] > 0 or abs(out_dur - src_dur) > 0.05
    words = _remap([w for w in words if not w["cut"] and not w["hidden"]], segs, fps)

    hook_seconds = float(settings["hook_seconds"]) if settings["hook_style"] != "none" else 0.0
    hook_text = clip.get("hook", "") if hook_seconds else ""
    brand = {"color": settings.get("brand_color", ""), "font": settings.get("brand_font", ""),
             "handle": settings.get("handle_text", "")}
    ass = captions.build_ass(
        words, out_dur, width, height, style_key, clip.get("emphasis_words", []),
        hook_text, settings["hook_style"], hook_seconds,
        settings["caption_position"], float(settings["caption_scale"]), brand)
    ass_name = f"{name}.ass"
    (job_dir / ass_name).write_text(ass, encoding="utf-8")

    inputs = ["-ss", f"{start:.3f}", "-t", f"{src_dur:.3f}", "-i", src]
    n_inputs = 1

    def add_input(*args: str) -> int:
        nonlocal n_inputs
        inputs.extend(args)
        n_inputs += 1
        return n_inputs - 1

    # ---- video chain -------------------------------------------------------
    chains: list[str] = []
    v = [f"fps={fps}"]
    if settings["reframe"] == "blur":
        chains.append(
            f"[0:v]fps={fps},split[bga][fga];"
            f"[bga]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"boxblur=luma_radius={max(8, width // 40)}:luma_power=2,eq=brightness=-0.08[bg];"
            f"[fga]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        head, v = "[v0]", []
    else:
        head = "[0:v]"
        v.append(f"crop={plan['cw']}:{plan['ch']}:x='{plan['x_expr']}':y={plan['y']}")
    # Half-frame offsets keep the comparisons away from exact frame timestamps.
    sel = "+".join(f"gte(t,{(a - 0.5) / fps:.5f})*lt(t,{(b - 0.5) / fps:.5f})" for a, b in segs)
    if cut:
        v += [f"select='{sel}'", f"setpts=N/({fps}*TB)"]
    v += [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
    windows = _zoom_windows(words, out_dur, hook_seconds) if settings["zooms"] else []
    if windows:
        cond = "+".join(f"between(in_time,{a:.3f},{b - 0.001:.3f})" for a, b in windows)
        z = float(settings["zoom_amount"])
        v.append(f"zoompan=z='if({cond},{z:.3f},1)':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*0.38'"
                 f":d=1:s={width}x{height}:fps={fps}")
    v.append("format=yuv420p")
    chains.append(head + ",".join(v) + "[vx0]")
    cur, step = "[vx0]", 0

    def push(expr: str) -> None:
        nonlocal cur, step
        step += 1
        chains.append(expr.replace("{in}", cur) + f"[vx{step}]")
        cur = f"[vx{step}]"

    accent = brand["color"] or captions.STYLES.get(style_key or "", {}).get("active", "#FFE100")
    if accent.upper() == "#FFFFFF":
        accent = "#FFE100"
    if settings["progress_bar"]:
        bar_h = max(4, int(height * 0.007) // 2 * 2)
        chains.append(f"color=c={accent.replace('#', '0x')}:s={width}x{bar_h}:r={fps}[bar]")
        push(f"{{in}}[bar]overlay=x='-w+w*t/{out_dur:.3f}':y=H-h:shortest=1")

    cues = _emoji_cues(clip, words, out_dur) if settings["emojis"] and style_key else []
    if cues:
        size = int(min(width, height * 0.8) * 0.14) // 2 * 2
        pos = settings["caption_position"]
        if pos == "center":
            ey = int(height * 0.40) - size
        elif pos == "top":
            ey = int(height * (0.36 if portrait else 0.30))
        else:
            ey = int(height * (0.665 if portrait else 0.76)) - size
        for a, b, png in cues:
            idx = add_input("-loop", "1", "-i", png)
            chains.append(f"[{idx}:v]scale={size}:{size}:flags=lanczos,format=rgba[em{idx}]")
            push(f"{{in}}[em{idx}]overlay=x=(W-w)/2:y={max(0, ey)}:enable='between(t,{a:.3f},{b:.3f})':eof_action=pass")

    push(f"{{in}}ass={ass_name}:fontsdir=fonts")

    logo = settings.get("logo_path") if settings.get("logo_enabled") else None
    if logo and Path(logo).exists():
        idx = add_input("-loop", "1", "-i", logo)
        lw = max(16, int(width * float(settings["logo_scale"])) // 2 * 2)
        m = int(min(width, height) * 0.04)
        x = f"{m}" if "left" in settings["logo_position"] else f"W-w-{m}"
        y = f"{m}" if "top" in settings["logo_position"] else f"H-h-{m * 2}"
        chains.append(f"[{idx}:v]scale={lw}:-1:flags=lanczos,format=rgba,"
                      f"colorchannelmixer=aa={float(settings['logo_opacity']):.2f}[logo]")
        push(f"{{in}}[logo]overlay=x={x}:y={y}:eof_action=pass")
    chains.append(f"{cur}null[v]")

    # ---- audio chain -------------------------------------------------------
    has_audio = info.get("has_audio", True)
    if has_audio:
        a = ["[0:a]aresample=48000"]
        if cut:
            a += [f"aselect='{sel}'", "asetpts=N/SR/TB"]
        if settings["normalize_audio"]:
            a += ["loudnorm=I=-14:TP=-1.5:LRA=11", "aresample=48000"]
    else:
        idx = add_input("-f", "lavfi", "-t", f"{out_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo")
        a = [f"[{idx}:a]anull"]
    chains.append(",".join(a) + "[a0]")
    mix = ["[a0]"]

    events: list[tuple[float, str]] = []
    if settings["sfx"]:
        whoosh, pop = _make_sfx(job_dir)
        if (job_dir / whoosh).exists() and (job_dir / pop).exists():
            events += [(max(0.0, a_ - 0.12), whoosh) for a_, _ in windows]
            events += [(a_, pop) for a_, _, _ in cues]
            if hook_text:
                events.append((0.05, pop))
            events = sorted(events)[:16]
    vol = float(settings["sfx_volume"])
    for name_ in sorted({e[1] for e in events}):
        times = [t for t, n in events if n == name_]
        idx = add_input("-i", name_)
        tag = name_.split(".")[0]
        chains.append(f"[{idx}:a]aresample=48000,volume={vol:.3f},asplit={len(times)}"
                      + "".join(f"[{tag}s{k}]" for k in range(len(times))))
        for k, t in enumerate(times):
            chains.append(f"[{tag}s{k}]adelay={int(t * 1000)}:all=1[{tag}d{k}]")
            mix.append(f"[{tag}d{k}]")
    if music:
        idx = add_input("-stream_loop", "-1", "-i", music)
        chains.append(f"[{idx}:a]aresample=48000,volume={float(settings['music_volume']):.3f},"
                      f"afade=t=out:st={max(0.0, out_dur - 0.6):.3f}:d=0.6[mus]")
        mix.append("[mus]")
    if len(mix) > 1:
        chains.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:dropout_transition=0:normalize=0,"
                      "alimiter=limit=0.95[a]")
    else:
        chains.append("[a0]anull[a]")

    preset, crf = QUALITY.get(settings["quality"], QUALITY["balanced"])
    out_name = f"{name}.mp4"
    cmd = [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", ";".join(chains), "-map", "[v]", "-map", "[a]",
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
    log(f"Clip {index} rendered: {out_dur:.1f}s, style {style_key or 'no captions'}, {len(segs) - 1} cuts "
        f"({fillers} filler words), {plan['pieces']} camera positions, {len(cues)} emojis, {len(events)} sound effects.")
    return {
        "index": index, "file": out_name, "thumb": thumb, "duration": round(out_dur, 2),
        "source_start": round(start, 2), "source_end": round(end, 2), "style": style_key,
        "width": width, "height": height, "w0": w0, "w1": w1, "version": version,
        "title": clip["title"], "hook": clip.get("hook", ""), "description": clip.get("description", ""),
        "hashtags": clip.get("hashtags", []), "score": clip.get("score", 0), "reason": clip.get("reason", ""),
        "scores": clip.get("scores") or {},
        "engine": clip.get("engine", "offline"),
        "transcript": " ".join(w["w"] for w in all_words[w0:w1 + 1]),
    }
