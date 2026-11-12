"""Compilation videos: find moments across several source videos and join them into
one video with an intro, number labels and optional music."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config
from . import captions, reframe, render

THEMES = {
    "best": ("Best moments", "the strongest, most memorable moments"),
    "funny": ("Funniest moments", "the moments most likely to make a viewer laugh: jokes, absurd situations, "
              "comic timing, people cracking up"),
    "surprising": ("Most surprising moments", "reveals, twists, shocked reactions and things nobody saw coming"),
    "useful": ("Most useful tips", "the most practical, immediately usable advice, each one a complete tip"),
    "emotional": ("Most emotional moments", "the moments with the strongest feeling: moving, inspiring or heartfelt"),
    "custom": ("Custom theme", ""),
}
STYLES = {"countdown": "Numbered countdown (Top 5, best last)", "reel": "Plain reel (no numbers)"}


def theme_direction(settings: dict) -> tuple[str, str]:
    """(short name for titles, instruction for the AI)."""
    key = settings.get("comp_theme", "best")
    name, what = THEMES.get(key, THEMES["best"])
    custom = str(settings.get("comp_theme_text", "")).strip()
    if key == "custom" and custom:
        name, what = custom, custom
    elif key == "custom":
        name, what = THEMES["best"]
    direction = (
        f"These clips are for a compilation video on the theme: {what}. Pick moments that fit this theme above "
        "all else. A moment for a compilation can be short and does not need a full story arc, but it must be "
        "enjoyable on its own with no context, start cleanly and end on its high point. Where the transcript marks "
        "a loud reaction, the audience or the people on screen laughed, cheered or shouted there, which is strong "
        "evidence that something worth showing happened in that sentence or just before it.")
    return name, direction


def loud_sentences(path: str, sentences: list[dict]) -> set[int]:
    """Indices of sentences during which the audio has a loud burst (laughter, cheering,
    shouting), judged against the video's own typical level."""
    proc = subprocess.run(
        [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000",
         "-f", "s16le", "-"], capture_output=True)
    audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32)
    win = 4000  # half a second
    if len(audio) < win * 20:
        return set()
    frames = audio[:len(audio) // win * win].reshape(-1, win)
    level = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1.0)
    active = level[level > np.percentile(level, 30)]
    if len(active) < 10:
        return set()
    threshold = float(np.median(active) + max(6.0, 1.8 * active.std()))
    loud = level > threshold
    marks: set[int] = set()
    for s in sentences:
        a, b = int(s["s"] * 2), int(s["e"] * 2) + 2
        if loud[a:b].any():
            marks.add(s["i"])
    # A reaction marked on more than a fifth of the video is just a loud video, not a signal.
    return marks if len(marks) <= len(sentences) * 0.2 else set()


def default_title(settings: dict, count: int) -> str:
    name, _ = theme_direction(settings)
    if settings.get("comp_style") == "countdown":
        base = re.sub(r"\s*moments$", "", name, flags=re.I)
        return f"Top {count} {base} moments" if base.lower() != name.lower() else f"Top {count}: {name}"
    return name


def _ass(width: int, height: int, total: float, title: str, intro: float, labels: list[tuple[float, str, str]],
         platform: str) -> str:
    """Title over the intro and a label at the start of every moment."""
    portrait = height >= width
    zone = captions.SAFE_ZONES.get(platform, captions.SAFE_ZONES["universal"]) if portrait else {"top": 0.05, "right": 0.05}
    ref = min(width, height * 0.8) if portrait else height * 0.95
    ml = int(width * 0.06)
    mr = int(width * max(0.06, zone["right"]))
    top = int(height * zone["top"])
    c = captions._ass_color
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Title,Anton,{ref * 0.13:.0f},{c('#FFFFFF')},{c('#FFFFFF')},{c('#000000')},{c('#000000', 0x60)},"
        f"0,0,0,0,100,100,0,0,1,{ref * 0.012:.1f},{ref * 0.006:.1f},5,{ml},{mr},0,1",
        f"Style: Num,Anton,{ref * 0.2:.0f},{c('#FFE100')},{c('#FFE100')},{c('#000000')},{c('#000000', 0x60)},"
        f"0,0,0,0,100,100,0,0,1,{ref * 0.014:.1f},{ref * 0.007:.1f},7,{ml},{mr},{top},1",
        f"Style: Label,Poppins ExtraBold,{ref * 0.052:.0f},{c('#FFFFFF')},{c('#FFFFFF')},{c('#000000')},"
        f"{c('#000000', 0x60)},0,0,0,0,100,100,0,0,1,{ref * 0.006:.1f},{ref * 0.003:.1f},7,{ml},{mr},"
        f"{top + (int(ref * 0.23) if any(n for _, n, _ in labels) else 0)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    clean = lambda t: str(t).replace("\\", "").replace("{", "").replace("}", "").strip()
    ts = captions._ts
    events = []
    if title and intro > 0.3:
        import textwrap

        text = "\\N".join(textwrap.wrap(clean(title).upper(), 16 if portrait else 30))
        events.append(f"Dialogue: 3,{ts(0)},{ts(intro)},Title,,0,0,0,,{{\\fad(120,200)\\fscx85\\fscy85"
                      f"\\t(0,160,\\fscx100\\fscy100)}}{text}")
    for start, number, label in labels:
        end = min(total, start + 3.0)
        if number:
            events.append(f"Dialogue: 3,{ts(start)},{ts(end)},Num,,0,0,0,,{{\\fad(100,250)\\fscx70\\fscy70"
                          f"\\t(0,140,\\fscx100\\fscy100)}}{clean(number)}")
        if label:
            events.append(f"Dialogue: 3,{ts(start)},{ts(end)},Label,,0,0,0,,{{\\fad(100,250)}}{clean(label)[:60]}")
    return "\n".join(head + events) + "\n"


def assemble(job_dir: Path, moments: list[dict], teaser: str | None, title: str, settings: dict, music: str | None,
             fps_s: str, progress: Callable[[float, str], None], cancelled: Callable[[], bool],
             log: Callable[[str], None]) -> dict:
    """Join rendered moments (already cropped, captioned and levelled) into the final video.
    moments: [{"file", "title", "number"}] in playing order."""
    width, height = render.output_size(settings["aspect"], settings["resolution"])
    files = ([teaser] if teaser else []) + [m["file"] for m in moments]
    durations = [reframe.probe(str(job_dir / f))["duration"] for f in files]
    starts = [sum(durations[:i]) for i in range(len(files))]
    total = sum(durations)
    intro = durations[0] if teaser else min(2.6, durations[0])
    offset = 1 if teaser else 0
    labels = []
    for k, m in enumerate(moments):
        start = starts[k + offset]
        # With no teaser the title sits on the first moment, so its own label waits for the title to clear.
        labels.append((start + (intro if not teaser and k == 0 and title else 0.15), m.get("number", ""), m.get("title", "")))

    name = "compilation_" + render.slugify(title or "video")
    ass_name = f"{name}.ass"
    (job_dir / ass_name).write_text(
        _ass(width, height, total, title, intro, labels, settings.get("platform", "universal")), encoding="utf-8")

    inputs: list[str] = []
    n_inputs = 0

    def add_input(*args: str) -> int:
        nonlocal n_inputs
        inputs.extend(args)
        n_inputs += 1
        return n_inputs - 1

    chains: list[str] = []
    for f in files:
        i = add_input("-i", f)
        chains.append(f"[{i}:v]fps={fps_s},scale={width}:{height},setsar=1,format=yuv420p[cv{i}]")
        chains.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[ca{i}]")
    n = len(files)
    chains.append("".join(f"[cv{i}][ca{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[joined][speech]")
    chains.append(f"[joined]fade=t=in:d=0.3,fade=t=out:st={max(0.0, total - 0.6):.3f}:d=0.6[base]")
    brand = {"color": settings.get("brand_color", ""), "font": settings.get("brand_font", ""),
             "handle": settings.get("handle_text", "")}
    over = dict(settings, emojis=False)
    cur, _ = render.add_overlays(chains, "[base]", add_input, over, None, {}, [], total, width, height, fps_s,
                                 ass_name, brand)
    chains.append(f"{cur}null[v]")

    # Audio: speech, a whoosh on every join, optional music lowered under the speech.
    mix = ["[speech]"]
    whoosh, _ = render._make_sfx(job_dir)
    joins = [t for t in starts[1:] if t > 0.3]
    if joins and (job_dir / whoosh).exists():
        i = add_input("-i", whoosh)
        chains.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.55,"
                      f"asplit={len(joins)}" + "".join(f"[w{k}]" for k in range(len(joins))))
        for k, t in enumerate(joins):
            chains.append(f"[w{k}]adelay={int(max(0.0, t - 0.18) * 1000)}:all=1[wd{k}]")
            mix.append(f"[wd{k}]")
    if music:
        i = add_input("-stream_loop", "-1", "-i", music)
        vol = float(settings["music_volume"]) * 1.6
        mix[0] = "[sp]"
        chains.append("[speech]asplit=2[sp][key]")
        chains.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                      f"atrim=duration={total:.3f},volume={vol:.3f},afade=t=in:d=0.8,"
                      f"afade=t=out:st={max(0.0, total - 1.5):.3f}:d=1.5[bed]")
        chains.append("[bed][key]sidechaincompress=threshold=0.03:ratio=5:attack=40:release=500[duck]")
        mix.append("[duck]")
    if len(mix) > 1:
        chains.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:dropout_transition=0:normalize=0,"
                      "alimiter=limit=0.95[a]")
    else:
        chains.append("[speech]anull[a]")

    out_name = f"{name}.mp4"

    def build(codec: list[str]) -> list[str]:
        return [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
                "-filter_complex", ";".join(c for c in chains if c), "-map", "[v]", "-map", "[a]",
                *codec, "-pix_fmt", "yuv420p", "-r", fps_s,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
                "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_name]

    try:
        render.run_encode(build, settings, job_dir, total, lambda p: progress(p, "Joining the moments"), cancelled, log)
    finally:
        (job_dir / ass_name).unlink(missing_ok=True)
    if not (job_dir / out_name).exists():
        raise RuntimeError("Joining the compilation produced no file.")
    thumb = f"{name}.jpg"
    subprocess.run([config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{min(2.0, total / 2):.2f}",
                    "-i", out_name, "-frames:v", "1", "-vf", "scale=480:-2", thumb], cwd=str(job_dir), capture_output=True)
    stamp = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"
    chapters = ([f"0:00 Intro"] if teaser else []) + [
        f"{stamp(starts[k + offset])} {(m.get('number') + ' ') if m.get('number') else ''}{m.get('title', '')}".strip()
        for k, m in enumerate(moments)]
    if chapters and not chapters[0].startswith("0:00"):
        chapters[0] = "0:00 " + chapters[0].split(" ", 1)[1]
    return {"file": out_name, "thumb": thumb, "duration": round(total, 2), "width": width, "height": height,
            "chapters": chapters}
