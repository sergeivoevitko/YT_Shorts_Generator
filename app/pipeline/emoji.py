"""Emoji images for caption accents (Noto Emoji PNGs, cached on first use)."""
from __future__ import annotations

import re
import urllib.request

from .. import config

BASE_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/128/"

# Used when Claude is not available: spoken keyword -> emoji.
KEYWORDS = {
    "money": "💰", "dollars": "💰", "million": "💰", "billion": "💰", "rich": "💰", "pay": "💸", "cost": "💸",
    "fire": "🔥", "hot": "🔥", "crazy": "🤯", "insane": "🤯", "mind": "🤯", "love": "❤️", "heart": "❤️",
    "idea": "💡", "think": "🤔", "why": "🤔", "question": "🤔", "time": "⏰", "fast": "⚡", "quick": "⚡",
    "power": "⚡", "win": "🏆", "best": "🏆", "first": "🥇", "grow": "📈", "growth": "📈", "up": "📈",
    "down": "📉", "lose": "📉", "stop": "🛑", "never": "🚫", "wrong": "❌", "mistake": "❌", "right": "✅",
    "yes": "✅", "secret": "🤫", "world": "🌍", "work": "💼", "business": "💼", "brain": "🧠", "smart": "🧠",
    "learn": "📚", "book": "📚", "computer": "💻", "code": "💻", "ai": "🤖", "robot": "🤖", "model": "🤖",
    "phone": "📱", "people": "👥", "eyes": "👀", "look": "👀", "see": "👀", "laugh": "😂", "funny": "😂",
    "sad": "😢", "scary": "😱", "afraid": "😱", "strong": "💪", "hard": "💪", "goal": "🎯", "target": "🎯",
    "rocket": "🚀", "launch": "🚀", "future": "🚀", "food": "🍔", "sleep": "😴", "game": "🎮", "music": "🎵",
}


def _filename(char: str) -> str | None:
    points = [f"{ord(c):04x}" for c in char if ord(c) != 0xFE0F]
    if not points or any(ord(c) < 0x203C for c in char):
        return None
    return "emoji_u" + "_".join(points) + ".png"


def png_path(char: str) -> str | None:
    """Path to a PNG for this emoji, downloading it once if needed. None if unavailable."""
    name = _filename(char.strip())
    if not name:
        return None
    bundled = config.ASSETS / "emoji" / name
    if bundled.exists():
        return str(bundled)
    cache = config.WORKSPACE / "emoji"
    target = cache / name
    if target.exists():
        return str(target)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=8) as resp:
            data = resp.read()
        if data[:4] != b"\x89PNG":
            return None
        target.write_bytes(data)
        return str(target)
    except Exception:
        return None


def offline_cues(words: list[str], limit: int = 5) -> list[dict]:
    cues, used = [], set()
    for w in words:
        key = re.sub(r"[^a-z]", "", w.lower())
        emoji = KEYWORDS.get(key)
        if emoji and emoji not in used:
            used.add(emoji)
            cues.append({"word": w, "emoji": emoji})
        if len(cues) >= limit:
            break
    return cues
