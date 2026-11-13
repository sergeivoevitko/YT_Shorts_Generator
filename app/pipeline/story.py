"""Narration scripts: written by Claude, pasted by the user, or picked from a built-in set."""
from __future__ import annotations

import difflib
import re
from typing import Callable, List

from pydantic import BaseModel

from .. import config
from . import llm

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
WORDS_PER_SECOND = 2.5

# How a script on any topic is shaped (used when the creator types a topic).
FORMATS = {
    "facts": "Surprising facts: a fast run of 3 to 5 little-known facts, each one sentence or two",
    "explainer": "Explainer: answer one question clearly, step by step, in plain words",
    "story": "Story: tell one true or plausible story with a setup, a turn and a payoff",
    "list": "Top list: a countdown of 3 to 5 items, best last",
    "motivation": "Motivational: a short speech that builds to one clear idea",
}

TOPIC_PROMPT = """You write voiceover scripts for short vertical videos about any topic. The viewer \
hears a narrator, reads the words as captions, and sees stock footage that matches what is said.

Write for the ear:
- Open with a line that stops the scroll: a surprising claim, a question, or a tension. Never open \
with a greeting, "did you know" or "in this video".
- Short, plain sentences. One idea per sentence. Vary the rhythm.
- Be specific and correct. If you are not sure a fact is true, leave it out. No invented numbers, no \
quotes attributed to real people.
- End on a line worth remembering. Where it fits, let the last line echo the first so the video \
loops cleanly. No call to like or subscribe.
- Plain text a text-to-speech voice can read: no emojis, headings, stage directions, quotation marks \
or lists with symbols; numbers written as words. Separate paragraphs with a blank line where the \
narrator should pause.

For each script also return:
- title: 4 to 6 words, 20 to 40 characters, key word near the front.
- hook: on-screen text for the first seconds, at most 7 words, different from the first spoken line.
- description: one or two sentences, the point in the first 125 characters.
- hashtags: 3 to 5 hashtags without the # sign.
- emphasis_words: 5 to 10 single words from the script to highlight in the captions.
- emojis: 2 to 4 cues, each a single word from the script (copied exactly) and one fitting emoji.
- search_terms: one stock-footage search per paragraph or every two sentences, in script order, \
5 to 10 in total. Each is 1 to 3 plain English words describing something a camera can film \
(for example "ocean waves", "city traffic night", "chef cooking"), never abstract ideas.

When asked for several scripts, give each a clearly different angle and opening."""  # narration pace including pauses

SYSTEM_PROMPT = """You write short voiceover scripts for vertical videos made of nature, landscape and \
drone footage. The viewer hears a narrator and reads the words as captions.

Write for the ear:
- Short, plain sentences. One idea per sentence. Vary the rhythm, and let a few lines be very short.
- Open with a line that makes someone stop scrolling: a tension, a question, or a blunt truth. Never \
open with a greeting or "in a world".
- Build to one clear idea and end on a line worth remembering. Where it fits, let the last line \
echo the first, so the video loops cleanly when it replays. No call to like or subscribe.
- Speak to "you". Use concrete images that fit nature footage (weather, rivers, mountains, light, \
seasons, trails) without describing specific shots, because you do not know what the footage shows.
- Original writing only. No quotes from real people, no invented statistics, no cliches such as \
"the journey of a thousand miles" or "rise and grind".
- Plain text that a text-to-speech voice can read: no emojis, no headings, no stage directions, no \
quotation marks, numbers written out as words. Separate paragraphs with a blank line where the \
narrator should take a longer pause.

For each story also return:
- title: a short post title of 4 to 6 words and 20 to 40 characters, key word near the front.
- hook: on-screen text for the first seconds, at most 7 words, different from the first spoken line.
- description: a one or two sentence post caption with the point in the first 125 characters.
- hashtags: 3 to 5 hashtags without the # sign.
- emphasis_words: 5 to 10 single words from the script to highlight in the captions.
- emojis: 2 to 4 cues, each a single word from the script (copied exactly) and one fitting emoji.
- search_terms: one stock-footage search per paragraph, in script order, 4 to 8 in total, each 1 to \
3 plain English words for something a camera can film in nature (for example "mountain sunrise").

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
    search_terms: List[str]


class StoryBatch(BaseModel):
    stories: List[Story]


class LooseStory(BaseModel):
    script: str
    title: str = "Untitled"
    hook: str = ""
    description: str = ""
    hashtags: List[str] = []
    emphasis_words: List[str] = []
    emojis: List[EmojiCue] = []
    search_terms: List[str] = []


class LooseStoryBatch(BaseModel):
    stories: List[LooseStory]


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


def _request(n: int, theme: str, tone: str, seconds: float, direction: str, topic: str, fmt: str) -> tuple[str, str]:
    """(system prompt, user message) for a batch of scripts."""
    words = int(seconds * WORDS_PER_SECOND)
    kind = "script" if topic else "story"
    user = f"Write {n} {kind if n == 1 else 'different ' + kind + 's'}.\n"
    if topic:
        user += f"Topic: {topic}\nFormat: {FORMATS.get(fmt, FORMATS['facts'])}\n"
    else:
        user += f"Theme: {THEMES.get(theme, theme)}\n"
    user += (f"Tone: {TONES.get(tone, tone)}\n"
             f"Length: about {words} words each (roughly {seconds:.0f} seconds when spoken). "
             f"Stay within ten percent of that word count.\n")
    if direction.strip():
        user += f"Extra direction from the creator: {direction.strip()}\n"
    return (TOPIC_PROMPT if topic else SYSTEM_PROMPT), user


def write_with_claude(n: int, theme: str, tone: str, seconds: float, direction: str,
                      log: Callable[[str], None], topic: str = "", fmt: str = "facts") -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.get_api_key(), timeout=600.0)
    model = config.get_model()
    system, user = _request(n, theme, tone, seconds, direction, topic, fmt)
    kwargs = dict(model=model, max_tokens=16000, system=system,
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


STORY_SHAPE = """{"stories": [{"title": "...", "hook": "...", "script": "First paragraph.\\n\\nSecond paragraph.", \
"description": "...", "hashtags": ["...", "..."], "emphasis_words": ["...", "..."], \
"emojis": [{"word": "...", "emoji": "🌄"}], "search_terms": ["...", "..."]}]}"""


def write_with_provider(n: int, theme: str, tone: str, seconds: float, direction: str,
                        log: Callable[[str], None], topic: str = "", fmt: str = "facts") -> list[dict]:
    """Story writing through a free or local provider."""
    system, user = _request(n, theme, tone, seconds, direction, topic, fmt)
    batch = llm.chat_json(system, user, LooseStoryBatch, STORY_SHAPE, log, strict=StoryBatch)
    ai = llm.current()
    log(f"{ai['name']} ({ai['model']}) wrote {len(batch.stories)} stories.")
    return [s.model_dump() for s in batch.stories]


def get_stories(n: int, settings: dict, log: Callable[[str], None]) -> tuple[list[dict], str]:
    """Return (stories, source) where source is 'pasted', 'claude' or 'builtin'."""
    pasted = _clean_script(settings.get("story_text", ""))
    theme = settings["story_theme"]
    if pasted:
        first = re.split(r"(?<=[.!?])\s", pasted)[0]
        story = {"title": " ".join(first.split()[:9]).rstrip(".,!?"), "hook": "", "script": pasted,
                 "description": " ".join(pasted.split()[:28]) + ("..." if len(pasted.split()) > 28 else ""),
                 "hashtags": _keywords(pasted, 4) + ["motivation"], "emphasis_words": _keywords(pasted), "emojis": [],
                 "search_terms": _keywords(pasted, 6)}
        return [dict(story) for _ in range(n)], "pasted"

    topic = str(settings.get("story_topic", "")).strip()
    ai = llm.current()
    if topic and not (ai["ready"] and settings.get("use_ai", True)):
        raise RuntimeError("Writing a script about your topic needs an AI. Open AI settings and connect one "
                           "(Groq and Gemini are free), or paste your own text instead.")
    if ai["ready"] and settings.get("use_ai", True):
        try:
            writer = write_with_claude if ai["provider"] == "claude" else write_with_provider
            stories = writer(n, theme, settings["story_tone"], float(settings["story_seconds"]),
                             settings.get("instructions", ""), log, topic, settings.get("story_format", "facts"))
            for s in stories:
                s["script"] = _clean_script(s["script"])
                s["hashtags"] = [re.sub(r"[^A-Za-z0-9_]", "", h) for h in s["hashtags"] if h][:8]
                s["search_terms"] = [" ".join(str(t).split()[:4]) for t in s.get("search_terms") or [] if str(t).strip()][:12] \
                    or _keywords(s["script"], 6)
            stories = [s for s in stories if s["script"]]
            if stories:
                while len(stories) < n:
                    stories.append(dict(stories[len(stories) % len(stories)]))
                return stories[:n], ai["name"]
        except Exception as exc:
            if topic:
                raise RuntimeError(f"{ai['name']} could not write a script about '{topic}': {exc}") from exc
            log(f"{ai['name']} could not write the story ({type(exc).__name__}: {exc}). Using a built-in story instead.")
    else:
        log("No AI is set up and no text was pasted, so built-in stories are used.")

    order = ([theme] if theme in BUILTIN else []) + [k for k in BUILTIN if k != theme]
    stories = []
    for i in range(n):
        title, hook, script = BUILTIN[order[i % len(order)]]
        stories.append({"title": title, "hook": hook, "script": script, "description": title + ".",
                        "hashtags": _keywords(script, 3) + ["motivation", "nature"],
                        "emphasis_words": _keywords(script), "emojis": [],
                        "search_terms": ["mountain sunrise", "forest", "river", "ocean waves", "clouds sky", "hiking trail"]})
    return stories, "builtin"


# ---------------------------------------------------------------------------
# Idea generator
# ---------------------------------------------------------------------------

class Idea(BaseModel):
    topic: str
    format: str
    angle: str


class IdeaList(BaseModel):
    ideas: List[Idea]


class LooseIdea(BaseModel):
    topic: str
    format: str = "facts"
    angle: str = ""


class LooseIdeaList(BaseModel):
    ideas: List[LooseIdea]


IDEA_PROMPT = """You suggest ideas for short vertical videos (30 to 60 seconds) that are narrated by a \
voiceover over stock footage, with no presenter on camera. Good ideas are specific, make people curious \
in one line, and can be shown with ordinary footage. Avoid topics that need recent news, real people's \
private lives, medical or financial advice, or facts you are not sure of.

Return ideas as: topic (the video's subject in one specific line, under 90 characters), format (one of: \
facts, explainer, story, list, motivation) and angle (one sentence on what makes it interesting)."""
IDEA_SHAPE = '{"ideas": [{"topic": "...", "format": "facts", "angle": "..."}]}'


def ideas(niche: str, count: int, log: Callable[[str], None]) -> list[dict]:
    """Video ideas for a niche, from the AI chosen in AI settings."""
    ai = llm.current()
    if not ai["ready"]:
        raise RuntimeError("Generating ideas needs an AI. Open AI settings and connect one (Groq and Gemini are free).")
    user = f"Niche or audience: {niche.strip() or 'general curiosity'}\nGive {count} different ideas."
    if ai["provider"] == "claude":
        import anthropic

        client = anthropic.Anthropic(api_key=config.get_api_key(), timeout=300.0)
        resp = client.messages.parse(model=config.get_model(), max_tokens=8000, system=IDEA_PROMPT,
                                     messages=[{"role": "user", "content": user}],
                                     output_config={"effort": "medium"}, output_format=IdeaList)
        if resp.parsed_output is None:
            raise RuntimeError(f"Claude returned no ideas (stop reason: {resp.stop_reason}).")
        items = [i.model_dump() for i in resp.parsed_output.ideas]
    else:
        items = [i.model_dump() for i in llm.chat_json(IDEA_PROMPT, user, LooseIdeaList, IDEA_SHAPE, log, strict=IdeaList).ideas]
    out = []
    for item in items:
        if str(item.get("topic", "")).strip():
            fmt = str(item.get("format", "facts")).lower().strip()
            out.append({"topic": str(item["topic"]).strip()[:120], "format": fmt if fmt in FORMATS else "facts",
                        "angle": str(item.get("angle", "")).strip()[:200]})
    return out[:count]


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
