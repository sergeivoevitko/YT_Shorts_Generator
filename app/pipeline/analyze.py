"""Pick the clip-worthy moments: Claude when an API key is set, heuristics otherwise."""
from __future__ import annotations

import re
from typing import Callable, List

from pydantic import BaseModel

from .. import config
from . import llm

SYSTEM_PROMPT = """You are a senior short-form video editor who has cut thousands of high-retention \
TikToks, Reels and YouTube Shorts out of long videos.

You receive the transcript of a long video as numbered sentences with start and end times in seconds. \
Choose the moments that will perform best as standalone vertical shorts.

What makes a strong clip:
- The first sentence stops the scroll on its own: a bold claim, a surprising fact, a question, a \
conflict, a number, or the start of a story. Never start on filler, a greeting, or a reference to \
something said earlier ("so as I said", "that's why").
- It is fully understandable with zero context from the rest of the video.
- It has an arc: setup, development, and a payoff (a punchline, a lesson, a reveal). End on the \
payoff sentence, never mid-thought.
- It triggers an emotion or a reaction: curiosity, surprise, humor, disagreement, usefulness.
- Dense delivery. Avoid rambling stretches, sponsor reads, intros, outros and calls to subscribe.

For each clip return:
- start_sentence and end_sentence: sentence numbers from the transcript (inclusive). The clip runs \
from the start of the first to the end of the last, so choose boundaries with care.
- title: a short, specific title for the post (under 70 characters), no clickbait that the clip \
does not deliver.
- hook: on-screen text for the first three seconds, at most 8 words, that creates an open question \
the clip answers. It must be different from the spoken first sentence and must not spoil the payoff.
- description: a one or two sentence post caption.
- hashtags: 3 to 6 relevant hashtags without the # sign.
- emphasis_words: 4 to 10 single words spoken in the clip that carry the meaning (numbers, strong \
nouns and verbs) to highlight in the captions. Copy them exactly as spoken.
- emojis: 2 to 5 cues, each a single spoken word from the clip (copied exactly) and one emoji that \
illustrates it. Spread them across the clip and only use them where they add something.
- hook_score, flow_score, value_score, trend_score: 0 to 100 each. Hook: how hard the first three \
seconds stop the scroll. Flow: how cleanly the clip runs from setup to payoff without dead spots. \
Value: how much the viewer gains (insight, laugh, emotion). Trend: how well the topic fits what \
short-form audiences currently engage with.
- score: 0 to 100, your honest overall estimate of the clip's short-form potential relative to \
typical content in its niche. Be critical: most clips should land between 50 and 85.
- reason: one sentence on why this moment works.

Rules: clips must not overlap, each must respect the duration limits given, and they should cover \
different ideas. Return them best first. If the video has fewer good moments than requested, return \
fewer clips rather than padding with weak ones."""


class EmojiCue(BaseModel):
    word: str
    emoji: str


class ClipPlan(BaseModel):
    start_sentence: int
    end_sentence: int
    title: str
    hook: str
    description: str
    hashtags: List[str]
    emphasis_words: List[str]
    emojis: List[EmojiCue]
    hook_score: int
    flow_score: int
    value_score: int
    trend_score: int
    score: int
    reason: str


class VideoPlan(BaseModel):
    clips: List[ClipPlan]


def _transcript_text(sentences: list[dict]) -> str:
    return "\n".join(f"[{s['i']}] {s['s']:.1f}-{s['e']:.1f} {s['text']}" for s in sentences)


def analyze_with_claude(sentences: list[dict], title: str, n: int, min_len: float, max_len: float,
                        instructions: str, log: Callable[[str], None]) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.get_api_key(), timeout=900.0)
    model = config.get_model()
    user = (
        f"Video title: {title}\n"
        f"Number of clips wanted: {n}\n"
        f"Each clip must last between {min_len:.0f} and {max_len:.0f} seconds.\n"
    )
    if instructions.strip():
        user += f"Extra direction from the creator: {instructions.strip()}\n"
    user += f"\nTranscript:\n{_transcript_text(sentences)}"

    kwargs = dict(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        output_format=VideoPlan,
    )
    response = None
    if model != "claude-haiku-5-5":
        # If a safety classifier declines the request, let the API retry it on a
        # fallback model inside the same call instead of failing the job.
        try:
            response = client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except (anthropic.BadRequestError, TypeError) as exc:
            log(f"Fallback-enabled request was rejected ({exc}); retrying without it.")
    if response is None:
        response = client.messages.parse(**kwargs)

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to analyze this transcript.")
    plan = response.parsed_output
    if plan is None:
        raise RuntimeError(f"Claude returned no usable plan (stop reason: {response.stop_reason}).")
    usage = response.usage
    log(f"Claude ({response.model}) proposed {len(plan.clips)} clips "
        f"[{usage.input_tokens} tokens in, {usage.output_tokens} out].")
    return [c.model_dump() for c in plan.clips]


CLIP_SHAPE = """{"clips": [{"start_sentence": 12, "end_sentence": 19, "title": "...", "hook": "...", \
"description": "...", "hashtags": ["...", "..."], "emphasis_words": ["...", "..."], \
"emojis": [{"word": "...", "emoji": "🔥"}], "hook_score": 80, "flow_score": 70, "value_score": 75, \
"trend_score": 60, "score": 74, "reason": "..."}]}"""


def analyze_with_provider(sentences: list[dict], title: str, n: int, min_len: float, max_len: float,
                          instructions: str, log: Callable[[str], None]) -> list[dict]:
    """Same task through a free or local provider. Long transcripts are sent in parts,
    because free tiers and local models accept much less text per request than Claude."""
    ai = llm.current()
    chunks: list[list[dict]] = [[]]
    size = 0
    for s in sentences:
        line = len(s["text"]) + 24
        if chunks[-1] and size + line > ai["chunk"]:
            # Overlap a little so a moment that straddles the boundary is not lost.
            chunks.append(chunks[-1][-6:])
            size = sum(len(x["text"]) + 24 for x in chunks[-1])
        chunks[-1].append(s)
        size += line
    per_chunk = n if len(chunks) == 1 else max(1, -(-n * 3 // (2 * len(chunks))))
    found: list[dict] = []
    for i, part in enumerate(chunks, start=1):
        user = (f"Video title: {title}\nNumber of clips wanted: {per_chunk}\n"
                f"Each clip must last between {min_len:.0f} and {max_len:.0f} seconds.\n")
        if len(chunks) > 1:
            user += f"This is part {i} of {len(chunks)} of the transcript. Only use sentence numbers that appear below.\n"
        if instructions.strip():
            user += f"Extra direction from the creator: {instructions.strip()}\n"
        user += f"\nTranscript:\n{_transcript_text(part)}"
        plan = llm.chat_json(SYSTEM_PROMPT, user, VideoPlan, CLIP_SHAPE, log)
        lo, hi = part[0]["i"], part[-1]["i"]
        found += [c.model_dump() for c in plan.clips if lo <= c.start_sentence <= c.end_sentence <= hi]
        if len(chunks) > 1:
            log(f"{ai['name']}: part {i} of {len(chunks)} analyzed, {len(plan.clips)} candidates.")
    found.sort(key=lambda c: -c["score"])
    log(f"{ai['name']} ({ai['model']}) proposed {len(found)} clips.")
    return found


# ---------------------------------------------------------------------------
# Offline fallback
# ---------------------------------------------------------------------------

_HOOK_WORDS = {
    "secret", "never", "always", "mistake", "truth", "nobody", "everyone", "crazy", "insane", "best",
    "worst", "biggest", "stop", "why", "how", "what", "imagine", "actually", "problem", "money",
    "million", "billion", "free", "fastest", "easiest", "hate", "love", "wrong", "lie", "lied",
    "shocking", "weird", "simple", "trick", "hack", "rule", "reason", "story", "first", "only",
}
_FILLER_START = {"so", "and", "but", "um", "uh", "yeah", "okay", "ok", "well", "because", "or", "also"}
_STOP = set("the a an and or but of to in on at for with is are was were be been it this that i you he she "
            "we they my your our their me him her them so if as not do does did have has had just really "
            "very can will would could should about from into than then there here what when where who how "
            "why which because like get got go going gonna know think thing things one out up all some".split())


def _sub_scores(sents: list[dict]) -> dict:
    first_words = re.findall(r"[a-z0-9']+", sents[0]["text"].lower())
    text = " ".join(s["text"] for s in sents).lower()
    words = re.findall(r"[a-z0-9']+", text)
    dur = max(sents[-1]["e"] - sents[0]["s"], 1.0)
    hook = 45 + 9 * len(_HOOK_WORDS.intersection(first_words)) + (12 if "?" in sents[0]["text"] else 0) \
        + (8 if re.search(r"\d", sents[0]["text"]) else 0) - (20 if first_words and first_words[0] in _FILLER_START else 0)
    flow = 55 + (len(words) / dur - 2.2) * 14 + (10 if sents[-1]["text"].rstrip().endswith((".", "!", "?")) else -10)
    value = 45 + min(len(_HOOK_WORDS.intersection(words)), 6) * 5 + min(len(set(words) - _STOP) / max(len(words), 1) * 40, 25)
    clamp = lambda x: int(max(5, min(95, x)))
    return {"hook": clamp(hook), "flow": clamp(flow), "value": clamp(value)}


def _score_window(sents: list[dict]) -> float:
    first = sents[0]["text"].lower()
    first_words = re.findall(r"[a-z0-9']+", first)
    text = " ".join(s["text"] for s in sents).lower()
    words = re.findall(r"[a-z0-9']+", text)
    dur = max(sents[-1]["e"] - sents[0]["s"], 1.0)
    score = 40.0
    score += 6 * len(_HOOK_WORDS.intersection(first_words))
    score += 8 if "?" in sents[0]["text"] else 0
    score += 5 if re.search(r"\d", first) else 0
    score += 4 if "you" in first_words else 0
    score -= 12 if first_words and first_words[0] in _FILLER_START else 0
    score += min(len(_HOOK_WORDS.intersection(words)), 6) * 1.5
    score += min(text.count("?"), 3) * 2 + min(text.count("!"), 3) * 1.5
    rate = len(words) / dur  # words per second: dense speech holds attention
    score += max(min((rate - 2.2) * 8, 10), -10)
    score += 5 if sents[-1]["text"].rstrip().endswith((".", "!", "?")) else -5
    if re.search(r"subscribe|sponsor|like and|link in the description|patreon", text):
        score -= 25
    return max(0.0, min(score, 95.0))


def _keywords(text: str, k: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z0-9']+", text):
        lw = w.lower()
        if lw in _STOP or len(lw) < 4 and not lw.isdigit():
            continue
        counts[lw] = counts.get(lw, 0) + (3 if lw in _HOOK_WORDS or lw[0].isdigit() else 1) + len(lw) // 6
    return [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:k]]


def _short_title(text: str, limit: int = 8) -> str:
    words = text.split()
    out = " ".join(words[:limit]).rstrip(",;:")
    return out + ("..." if len(words) > limit else "")


def analyze_heuristic(sentences: list[dict], n: int, min_len: float, max_len: float,
                      taken: list[tuple[int, int]] | None = None) -> list[dict]:
    target = (min_len + max_len) / 2
    candidates = []
    for i in range(len(sentences)):
        j = i
        while j + 1 < len(sentences) and sentences[j]["e"] - sentences[i]["s"] < target:
            j += 1
        while j > i and sentences[j]["e"] - sentences[i]["s"] > max_len:
            j -= 1
        dur = sentences[j]["e"] - sentences[i]["s"]
        if dur < min_len or dur > max_len:
            continue
        candidates.append((_score_window(sentences[i:j + 1]), i, j))
    candidates.sort(reverse=True)
    used = list(taken or [])
    clips = []
    for score, i, j in candidates:
        if len(clips) >= n:
            break
        if any(not (j < a or i > b) for a, b in used):
            continue
        used.append((i, j))
        text = " ".join(s["text"] for s in sentences[i:j + 1])
        kws = _keywords(text)
        clips.append({
            "start_sentence": i, "end_sentence": j,
            "title": _short_title(sentences[i]["text"], 9),
            "hook": _short_title(sentences[i]["text"], 7).upper(),
            "description": _short_title(text, 28),
            "hashtags": kws[:4] + ["shorts"],
            "emphasis_words": kws,
            "score": int(score),
            "scores": _sub_scores(sentences[i:j + 1]),
            "reason": "Picked by offline scoring (strong opening words, dense speech, clean ending).",
        })
    return clips


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _fit(clip: dict, sentences: list[dict], min_len: float, max_len: float) -> dict | None:
    last = len(sentences) - 1
    a = max(0, min(int(clip["start_sentence"]), last))
    b = max(a, min(int(clip["end_sentence"]), last))
    dur = lambda: sentences[b]["e"] - sentences[a]["s"]
    while dur() > max_len and b > a:
        b -= 1
    while dur() < min_len and b < last and sentences[b + 1]["e"] - sentences[a]["s"] <= max_len:
        b += 1
    if dur() < min(min_len * 0.6, 8) or dur() > max_len * 1.15:
        return None
    clip = dict(clip, start_sentence=a, end_sentence=b)
    clip["hashtags"] = [re.sub(r"[^A-Za-z0-9_]", "", h) for h in clip.get("hashtags", []) if h][:8]
    clip["hook"] = " ".join(str(clip.get("hook", "")).split()[:12])
    clip["score"] = max(0, min(int(clip.get("score", 50)), 100))
    if "hook_score" in clip:
        clip["scores"] = {k: max(0, min(int(clip.pop(f"{k}_score", 0)), 100)) for k in ("hook", "flow", "value", "trend")}
    clip["emojis"] = [e for e in clip.get("emojis") or [] if e.get("word") and e.get("emoji")][:6]
    return clip


def analyze(sentences: list[dict], title: str, n: int, min_len: float, max_len: float,
            use_ai: bool, instructions: str, log: Callable[[str], None]) -> tuple[list[dict], str]:
    """Return (clips, engine). Each clip has start/end in seconds plus its metadata."""
    if not sentences:
        raise RuntimeError("No speech was found in this video, so there is nothing to clip.")
    clips: list[dict] = []
    engine = "offline"
    ai = llm.current()
    if use_ai and ai["ready"]:
        try:
            if ai["provider"] == "claude":
                clips = analyze_with_claude(sentences, title, n, min_len, max_len, instructions, log)
            else:
                clips = analyze_with_provider(sentences, title, n, min_len, max_len, instructions, log)
            engine = ai["name"]
        except Exception as exc:
            log(f"{ai['name']} analysis failed ({type(exc).__name__}: {exc}). Using offline scoring instead.")
    elif use_ai:
        log("No AI is set up (see AI settings), so moments are picked by offline scoring.")

    fitted: list[dict] = []
    used: list[tuple[int, int]] = []
    for clip in clips:
        clip = _fit(clip, sentences, min_len, max_len)
        if clip is None:
            continue
        a, b = clip["start_sentence"], clip["end_sentence"]
        if any(not (b < x or a > y) for x, y in used):
            continue
        used.append((a, b))
        fitted.append(clip)
    fitted = fitted[:n]

    if engine == "offline":
        fitted = analyze_heuristic(sentences, n, min_len, max_len)
    if not fitted:
        raise RuntimeError("Could not find any stretch of speech that fits the clip length you chose. "
                           "Try a shorter minimum length.")

    for clip in fitted:
        clip["start"] = sentences[clip["start_sentence"]]["s"]
        clip["end"] = sentences[clip["end_sentence"]]["e"]
        clip["w0"] = sentences[clip["start_sentence"]]["w0"]
        clip["w1"] = sentences[clip["end_sentence"]]["w1"]
        clip["engine"] = engine
    fitted.sort(key=lambda c: -c["score"])
    return fitted, engine
