"""Animated word-by-word captions and hook overlays as an ASS subtitle file."""
from __future__ import annotations

import re
import textwrap

# Sizes are fractions of a reference width so every resolution looks the same.
STYLES: dict[str, dict] = {
    "bold_pop": {
        "name": "Bold Pop", "desc": "Uppercase, yellow active word, pop-in",
        "font": "Poppins ExtraBold", "size": 0.082, "upper": True, "words": 3, "chars": 20,
        "primary": "#FFFFFF", "active": "#FFE100", "emphasis": "#2BFF6A",
        "outline_color": "#000000", "outline": 0.075, "shadow": 0.03, "anim": "pop",
    },
    "beast": {
        "name": "Comic Punch", "desc": "Chunky comic font, green active word that grows",
        "font": "Bangers", "size": 0.108, "upper": True, "words": 2, "chars": 16,
        "primary": "#FFFFFF", "active": "#39FF14", "emphasis": "#FF4D4D",
        "outline_color": "#000000", "outline": 0.085, "shadow": 0.05, "anim": "grow",
    },
    "karaoke": {
        "name": "Karaoke", "desc": "Color sweeps across each word as it is spoken",
        "font": "Poppins ExtraBold", "size": 0.07, "upper": False, "words": 4, "chars": 24,
        "primary": "#FFFFFF", "active": "#00E5FF", "emphasis": "#00E5FF",
        "outline_color": "#000000", "outline": 0.07, "shadow": 0.03, "anim": "karaoke",
    },
    "one_word": {
        "name": "One Word", "desc": "A single huge word at a time",
        "font": "Anton", "size": 0.15, "upper": True, "words": 1, "chars": 14,
        "primary": "#FFFFFF", "active": "#FFFFFF", "emphasis": "#FFE100",
        "outline_color": "#000000", "outline": 0.06, "shadow": 0.04, "anim": "pop",
    },
    "minimal": {
        "name": "Clean Minimal", "desc": "Small, calm, upcoming words dimmed",
        "font": "Poppins SemiBold", "size": 0.056, "upper": False, "words": 4, "chars": 26,
        "primary": "#FFFFFF", "active": "#FFFFFF", "emphasis": "#FFFFFF",
        "outline_color": "#000000", "outline": 0.045, "shadow": 0.05, "anim": "dim",
    },
    "neon": {
        "name": "Neon Glow", "desc": "Tall condensed font with a magenta glow",
        "font": "Bebas Neue", "size": 0.115, "upper": True, "words": 3, "chars": 20,
        "primary": "#FFFFFF", "active": "#00F0FF", "emphasis": "#FFE100",
        "outline_color": "#FF00C8", "outline": 0.035, "shadow": 0.0, "anim": "pop", "blur": 0.03,
    },
}
# How much of the screen each app covers with its own buttons and text, as fractions of the
# frame (portrait video). Text is kept out of these areas. "universal" is safe everywhere.
SAFE_ZONES = {
    "tiktok": {"top": 0.08, "bottom": 0.24, "right": 0.14},
    "reels": {"top": 0.13, "bottom": 0.22, "right": 0.12},
    "shorts": {"top": 0.09, "bottom": 0.20, "right": 0.16},
}
SAFE_ZONES["universal"] = {k: max(z[k] for z in SAFE_ZONES.values()) for k in ("top", "bottom", "right")}
PLATFORMS = {"universal": "All platforms (safest)", "tiktok": "TikTok", "reels": "Instagram Reels", "shorts": "YouTube Shorts"}

FONTS = ["Poppins ExtraBold", "Poppins SemiBold", "Anton", "Bangers", "Bebas Neue"]
MIX_ORDER = ["bold_pop", "beast", "karaoke", "neon", "one_word", "minimal"]

HOOK_STYLES = {
    "banner": {"name": "White banner", "text": "#000000", "box": "#FFFFFF", "boxed": True, "upper": False},
    "dark": {"name": "Dark banner", "text": "#FFE100", "box": "#000000", "boxed": True, "upper": True},
    "bold": {"name": "Bold outline", "text": "#FFE100", "box": "#000000", "boxed": False, "upper": True},
}


def _ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _tag_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _ts(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _clean(word: str, upper: bool) -> str:
    w = re.sub(r"^[\"'(\[{<]+|[\"')\]}>.,;:]+$", "", word.strip())
    w = w.replace("\\", "").replace("{", "").replace("}", "")
    return w.upper() if upper else w


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def group_words(words: list[dict], max_words: int, max_chars: int, gap: float = 0.45) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur_chars = sum(len(x["w"]) + 1 for x in cur)
        if cur and cur_chars + len(w["w"]) > max_chars:
            groups.append(cur)
            cur = []
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (len(cur) >= max_words or nxt is None or re.search(r"[.!?,;:]$", w["w"])
                or nxt["s"] - w["e"] > gap):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], duration: float, width: int, height: int, style_key: str | None,
              emphasis: list[str], hook_text: str, hook_style: str | None, hook_seconds: float,
              position: str = "lower", scale: float = 1.0, brand: dict | None = None,
              header: str = "", platform: str = "universal") -> str:
    """words: [{"w", "s", "e"}] with times relative to the start of the rendered clip."""
    portrait = height >= width
    zone = SAFE_ZONES.get(platform, SAFE_ZONES["universal"]) if portrait else {"top": 0.03, "bottom": 0.09, "right": 0.07}
    header = (header or "").replace("\\", "").replace("{", "").replace("}", "").strip()[:44]
    ref = min(width, height * 0.8)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
    ]
    events: list[str] = []
    margin_lr = int(width * 0.07)
    margin_r = int(width * max(0.07, zone["right"]))

    brand = brand or {}
    st = STYLES.get(style_key or "")
    if st:
        st = dict(st)
        if brand.get("color"):
            st["active"] = brand["color"]
            if st["words"] == 1:
                st["emphasis"] = brand["color"]
        if brand.get("font") in FONTS:
            st["font"] = brand["font"]
    if st and words:
        size = ref * st["size"] * scale * 1.22
        if position == "center":
            align, margin_v = 5, 0
        elif position == "top":
            align, margin_v = 8, int(height * (max(0.2, zone["top"] + 0.1) if portrait else 0.08))
        else:
            align, margin_v = 2, int(height * zone["bottom"])
        karaoke = st["anim"] == "karaoke"
        primary = _ass_color(st["active"] if karaoke else st["primary"])
        secondary = _ass_color(st["primary"])
        lines.append(
            f"Style: Cap,{st['font']},{size:.0f},{primary},{secondary},{_ass_color(st['outline_color'])},"
            f"{_ass_color('#000000', 0x60)},0,0,0,0,100,100,0,0,1,{size * st['outline']:.1f},"
            f"{size * st['shadow']:.1f},{align},{margin_lr},{margin_r},{margin_v},1")

        emph = {_norm(e) for e in emphasis if _norm(e)}
        blur = f"\\blur{size * st['blur']:.1f}" if st.get("blur") else ""
        groups = group_words(words, st["words"], st["chars"])
        for gi, group in enumerate(groups):
            next_start = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else duration
            group_end = max(group[-1]["s"] + 0.05, min(group[-1]["e"] + 0.3, next_start))
            texts = [_clean(w["w"], st["upper"]) for w in group]
            if not any(texts):
                continue
            if karaoke:
                parts = []
                for wi, w in enumerate(group):
                    end = group[wi + 1]["s"] if wi + 1 < len(group) else w["e"]
                    parts.append(f"{{\\kf{max(1, int(round((end - w['s']) * 100)))}}}{texts[wi]}")
                head = f"{{\\fad(60,0){blur}}}"
                events.append(f"Dialogue: 1,{_ts(group[0]['s'])},{_ts(group_end)},Cap,,0,0,0,,{head}{' '.join(parts)}")
                continue
            for wi, w in enumerate(group):
                start = w["s"]
                end = group[wi + 1]["s"] if wi + 1 < len(group) else group_end
                if end - start < 0.02:
                    continue
                head = blur
                if st["anim"] == "pop" and wi == 0:
                    head += "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)"
                if st["anim"] == "dim" and wi == 0:
                    head += "\\fad(90,0)"
                parts = []
                for xi, text in enumerate(texts):
                    is_emph = _norm(text) in emph
                    color = st["active"] if xi == wi else (st["emphasis"] if is_emph else st["primary"])
                    if st["words"] == 1 and is_emph:
                        color = st["emphasis"]
                    tags = f"\\c{_tag_color(color)}"
                    if st["anim"] == "grow":
                        tags += ("\\fscx100\\fscy100\\t(0,80,\\fscx116\\fscy116)" if xi == wi
                                 else "\\fscx100\\fscy100")
                    if st["anim"] == "dim":
                        tags += "\\alpha&H00&" if xi <= wi else "\\alpha&H90&"
                    parts.append(f"{{{tags}}}{text}")
                body = " ".join(parts)
                events.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{{{head}}}{body}" if head
                              else f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{body}")

    hk = HOOK_STYLES.get(hook_style or "")
    hook_text = (hook_text or "").strip()
    if hk and hook_text and hook_seconds > 0:
        size = ref * 0.08
        text = hook_text.replace("\\", "").replace("{", "").replace("}", "")
        text = text.upper() if hk["upper"] else text
        wrapped = "\\N".join(textwrap.wrap(text, 18 if portrait else 32)) or text
        margin_v = int(height * (zone["top"] + (0.055 if header else 0.0) + (0.0 if portrait else 0.02)))
        if hk["boxed"]:
            lines.append(
                f"Style: Hook,Poppins ExtraBold,{size:.0f},{_ass_color(hk['text'])},{_ass_color(hk['text'])},"
                f"{_ass_color(hk['box'])},{_ass_color(hk['box'])},0,0,0,0,100,100,0,0,3,{size * 0.22:.1f},0,8,"
                f"{margin_lr},{margin_r},{margin_v},1")
        else:
            lines.append(
                f"Style: Hook,Poppins ExtraBold,{size * 1.15:.0f},{_ass_color(hk['text'])},{_ass_color(hk['text'])},"
                f"{_ass_color(hk['box'])},{_ass_color('#000000', 0x60)},0,0,0,0,100,100,0,0,1,{size * 0.1:.1f},"
                f"{size * 0.04:.1f},8,{margin_lr},{margin_r},{margin_v},1")
        end = min(hook_seconds, duration)
        anim = "{\\fad(120,220)\\fscx88\\fscy88\\t(0,140,\\fscx100\\fscy100)}"
        events.append(f"Dialogue: 2,{_ts(0)},{_ts(end)},Hook,,0,0,0,,{anim}{wrapped}")

    if header:
        size = ref * 0.042
        lines.append(
            f"Style: Header,Poppins SemiBold,{size:.0f},{_ass_color('#FFFFFF')},{_ass_color('#FFFFFF')},"
            f"{_ass_color('#000000', 0x58)},{_ass_color('#000000', 0x58)},0,0,0,0,100,100,0,0,3,{size * 0.3:.1f},0,8,"
            f"{margin_lr},{margin_r},{int(height * zone['top'])},1")
        events.append(f"Dialogue: 0,{_ts(0)},{_ts(duration)},Header,,0,0,0,,{header}")

    handle = str(brand.get("handle") or "").replace("\\", "").replace("{", "").replace("}", "").strip()[:40]
    if handle:
        size = ref * 0.04
        lines.append(
            f"Style: Handle,Poppins SemiBold,{size:.0f},{_ass_color('#FFFFFF', 0x40)},{_ass_color('#FFFFFF', 0x40)},"
            f"{_ass_color('#000000', 0x70)},{_ass_color('#000000', 0x90)},0,0,0,0,100,100,0,0,1,{size * 0.06:.1f},0,2,"
            f"{margin_lr},{margin_r},{int(height * 0.035)},1")
        events.append(f"Dialogue: 0,{_ts(0)},{_ts(duration)},Handle,,0,0,0,,{handle}")

    lines += ["", "[Events]",
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    return "\n".join(lines + events) + "\n"
