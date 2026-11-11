"""Narration scripts: written by Claude, pasted by the user, or picked from a built-in set."""
from __future__ import annotations

import difflib
import re
from typing import Callable, List

from pydantic import BaseModel

from .. import config

THEMES = {
    "motivation": "Motivation: getting started and keeping going",
    "discipline": "Discipline and consistency",
    "resilience": "Resilience: getting through hard times",
    "calm": "Calm, stillness and perspective",
    "adventure": "Adventure and curiosity",
    "gratitude": "Gratitude and noticing what you have",
    "nature": "Lessons from nature",
}
TONES = {
    "cinematic": "Cinematic and powerful",
    "calm": "Calm and reflective",
    "intense": "Intense, tough-love",
    "poetic": "Poetic and gentle",
}
WORDS_PER_SECOND = 2.5  # narration pace including pauses

SYSTEM_PROMPT = """You write short voiceover scripts for vertical videos made of nature, landscape and \
drone footage. The viewer hears a narrator and reads the words as captions.

Write for the ear:
- Short, plain sentences. One idea per sentence. Vary the rhythm, and let a few lines be very short.
- Open with a line that makes someone stop scrolling: a tension, a question, or a blunt truth. Never \
open with a greeting or "in a world".
- Build to one clear idea and end on a line worth remembering. No call to like or subscribe.
- Speak to "you". Use concrete images that fit nature footage (weather, rivers, mountains, light, \
seasons, trails) without describing specific shots, because you do not know what the footage shows.
- Original writing only. No quotes from real people, no invented statistics, no cliches such as \
"the journey of a thousand miles" or "rise and grind".
- Plain text that a text-to-speech voice can read: no emojis, no headings, no stage directions, no \
quotation marks, numbers written out as words. Separate paragraphs with a blank line where the \
narrator should take a longer pause.

For each story also return:
- title: a short post title (under 70 characters).
- hook: on-screen text for the first seconds, at most 7 words, different from the first spoken line.
- description: a one or two sentence post caption.
- hashtags: 4 to 6 hashtags without the # sign.
- emphasis_words: 5 to 10 single words from the script to highlight in the captions.
- emojis: 2 to 4 cues, each a single word from the script (copied exactly) and one fitting emoji.

When asked for several stories, make them clearly different in angle and opening."""

# Used when there is no API key and nothing was pasted. Original texts written for this app.
BUILTIN = {
    "motivation": ("Start Before You Feel Ready", "Still waiting to feel ready?",
                   "You keep waiting to feel ready. That feeling is not coming.\n\n"
                   "The river does not wait for a clear path. It starts moving, and the path appears around it. "
                   "Every stone it meets, it learns to go around.\n\n"
                   "You do not need the whole map. You need one honest step, taken today, while it still feels too early.\n\n"
                   "Start small. Start unsure. But start."),
    "discipline": ("Small Days Build Big Things", "Nobody sees the quiet days",
                   "Nobody claps for the quiet days. The early mornings. The work that nobody will ever see.\n\n"
                   "But look at a canyon. No single day of water carved it. It was the same river, showing up, "
                   "for longer than anyone was watching.\n\n"
                   "You are not behind. You are in the middle of the carving.\n\n"
                   "Do today's part. Then do it again tomorrow."),
    "resilience": ("Storms Do Not Last", "Read this on a hard day",
                   "Some seasons take more than they give. You get tired in a way that sleep does not fix.\n\n"
                   "But no storm has ever stayed. The same wind that bends the tree also makes its roots go deeper.\n\n"
                   "You have survived every hard day you have ever had. Every single one.\n\n"
                   "This one is not different. Hold on. The weather is already turning."),
    "calm": ("You Are Allowed To Slow Down", "Slow down for one minute",
             "You do not have to earn your rest.\n\n"
             "The mountain is not in a hurry. The tide does not apologize for going out. "
             "Nothing in nature blooms all year, and nothing is ashamed of winter.\n\n"
             "Take one slow breath. Let your shoulders drop.\n\n"
             "The world will still be here in a minute. And you will meet it better."),
    "adventure": ("Go See What Is Out There", "What are you waiting for?",
                  "There is a version of your life that only exists on the other side of leaving.\n\n"
                  "Past the last street you know. Past the plan. Out where the trail stops being marked.\n\n"
                  "You will not find it by thinking about it. You find it with tired legs and cold hands "
                  "and a view you cannot explain to anyone.\n\n"
                  "Go. The world is bigger than your routine."),
    "gratitude": ("Look At What Is Already Here", "You already have more than you think",
                  "You are standing inside a day you once hoped for.\n\n"
                  "The air in your lungs. Light on the water. Someone, somewhere, who would answer if you called.\n\n"
                  "It is easy to count what is missing. It takes practice to notice what stayed.\n\n"
                  "So notice. Just for a moment. This is already a lot."),
    "nature": ("What The Trees Know", "Nature never rushes. Why do you?",
               "A tree does not grow faster because you are watching it.\n\n"
               "It takes what the season gives. Light when there is light. Rest when there is none. "
               "And year after year, without any noise, it becomes enormous.\n\n"
               "You are allowed to grow like that. Slowly. Quietly. In your own season.\n\n"
               "Nothing in nature is late."),
}


class EmojiCue(BaseModel):
    word: str
    emoji: str


class Story(BaseModel):
    title: str
    hook: str
    script: str
    description: str
    hashtags: List[str]
    emphasis_words: List[str]
    emojis: List[EmojiCue]


class StoryBatch(BaseModel):
    stories: List[Story]


def _clean_script(text: str) -> str:
    text = text.replace("\r", "").replace("“", "").replace("”", "").replace('"', "")
    text = re.sub(r"[*_#>`]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _keywords(text: str, k: int = 8) -> list[str]:
    stop = set("the a an and or but of to in on at for with is are was were be been it this that you your not do does "
               "have has had just very can will would could should about from into than then there here what when "
               "where who how why which because like get got one out up all some every still while once".split())
    counts: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z']+", text.lower()):
        if len(w) > 3 and w not in stop:
            counts[w] = counts.get(w, 0) + 1 + len(w) // 5
    return [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:k]]


def write_with_claude(n: int, theme: str, tone: str, seconds: float, direction: str,
                      log: Callable[[str], None]) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.get_api_key(), timeout=600.0)
    model = config.get_model()
    words = int(seconds * WORDS_PER_SECOND)
    user = (f"Write {n} {'story' if n == 1 else 'different stories'}.\n"
            f"Theme: {THEMES.get(theme, theme)}\nTone: {TONES.get(tone, tone)}\n"
            f"Length: about {words} words each (roughly {seconds:.0f} seconds when spoken). "
            f"Stay within ten percent of that word count.\n")
    if direction.strip():
        user += f"Extra direction from the creator: {direction.strip()}\n"
    kwargs = dict(model=model, max_tokens=16000, system=SYSTEM_PROMPT,
                  messages=[{"role": "user", "content": user}], thinking={"type": "adaptive"},
                  output_config={"effort": "high"}, output_format=StoryBatch)
    response = None
    if model != "claude-haiku-5-5":
        try:
            response = client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except (anthropic.BadRequestError, TypeError) as exc:
            log(f"Fallback-enabled request was rejected ({exc}); retrying without it.")
    if response is None:
        response = client.messages.parse(**kwargs)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude returned no story (stop reason: {response.stop_reason}).")
    log(f"Claude ({response.model}) wrote {len(response.parsed_output.stories)} stories.")
    return [s.model_dump() for s in response.parsed_output.stories]


def get_stories(n: int, settings: dict, log: Callable[[str], None]) -> tuple[list[dict], str]:
    """Return (stories, source) where source is 'pasted', 'claude' or 'builtin'."""
    pasted = _clean_script(settings.get("story_text", ""))
    theme = settings["story_theme"]
    if pasted:
        first = re.split(r"(?<=[.!?])\s", pasted)[0]
        story = {"title": " ".join(first.split()[:9]).rstrip(".,!?"), "hook": "", "script": pasted,
                 "description": " ".join(pasted.split()[:28]) + ("..." if len(pasted.split()) > 28 else ""),
                 "hashtags": _keywords(pasted, 4) + ["motivation"], "emphasis_words": _keywords(pasted), "emojis": []}
        return [dict(story) for _ in range(n)], "pasted"

    if config.get_api_key() and settings.get("use_ai", True):
        try:
            stories = write_with_claude(n, theme, settings["story_tone"], float(settings["story_seconds"]),
                                        settings.get("instructions", ""), log)
            for s in stories:
                s["script"] = _clean_script(s["script"])
                s["hashtags"] = [re.sub(r"[^A-Za-z0-9_]", "", h) for h in s["hashtags"] if h][:8]
            stories = [s for s in stories if s["script"]]
            if stories:
                while len(stories) < n:
                    stories.append(dict(stories[len(stories) % len(stories)]))
                return stories[:n], "claude"
        except Exception as exc:
            log(f"Claude could not write the story ({type(exc).__name__}: {exc}). Using a built-in story instead.")
    else:
        log("No Anthropic API key is set and no text was pasted, so built-in stories are used.")

    order = ([theme] if theme in BUILTIN else []) + [k for k in BUILTIN if k != theme]
    stories = []
    for i in range(n):
        title, hook, script = BUILTIN[order[i % len(order)]]
        stories.append({"title": title, "hook": hook, "script": script, "description": title + ".",
                        "hashtags": _keywords(script, 3) + ["motivation", "nature"],
                        "emphasis_words": _keywords(script), "emojis": []})
    return stories, "builtin"


def align(script: str, heard: list[dict], duration: float) -> list[dict]:
    """Give every word of the script a start and end time, using the words the
    speech recognizer heard in the narration audio as timing anchors."""
    tokens = script.split()
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    if heard:
        matcher = difflib.SequenceMatcher(a=[norm(t) for t in tokens], b=[norm(h["w"]) for h in heard], autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                h = heard[block.b + k]
                times[block.a + k] = (h["s"], h["e"])
    # Fill words the recognizer missed by spreading them between their neighbours.
    i = 0
    while i < len(tokens):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(tokens) and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else 0.0
        right = times[j][0] if j < len(tokens) else duration
        weights = [len(tokens[k]) + 1 for k in range(i, j)]
        span, pos = max(right - left, 0.05 * (j - i)), left
        for k, wgt in zip(range(i, j), weights):
            step = span * wgt / sum(weights)
            times[k] = (pos, pos + step * 0.92)
            pos += step
        i = j
    words = []
    last = 0.0
    for tok, (s, e) in zip(tokens, times):
        s = max(s, last)
        e = max(e, s + 0.05)
        words.append({"w": tok, "s": round(s, 3), "e": round(e, 3)})
        last = s
    return words
