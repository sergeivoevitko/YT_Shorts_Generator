"""Speech to text with word-level timestamps (faster-whisper)."""
from __future__ import annotations

import re
from typing import Callable

MODELS = {
    "tiny.en": "Tiny (fastest, least accurate)",
    "base.en": "Base (fast)",
    "small.en": "Small (recommended)",
    "medium.en": "Medium (slow on CPU, most accurate)",
}

_model_cache: dict = {}


def _load(model_name: str):
    from faster_whisper import WhisperModel

    if model_name in _model_cache:
        return _model_cache[model_name]
    model = None
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
    except Exception:
        model = None
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _model_cache.clear()
    _model_cache[model_name] = model
    return model


def _load_audio(path: str):
    """Decode to 16 kHz mono float32 with ffmpeg, which reads any container."""
    import subprocess

    import numpy as np

    from .. import config

    proc = subprocess.run(
        [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("Could not read the audio track: " + proc.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(proc.stdout, np.float32)


def _run(model, audio, progress, cancelled, duration_hint: float) -> list[dict]:
    segments, info = model.transcribe(
        audio, language="en", word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400}, beam_size=5,
        condition_on_previous_text=False,
    )
    total = info.duration or duration_hint or 1.0
    words: list[dict] = []
    for seg in segments:
        if cancelled():
            raise RuntimeError("Cancelled")
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"w": text, "s": round(float(w.start), 3), "e": round(float(w.end), 3)})
        progress(min(seg.end / total, 1.0), f"Transcribing speech ({seg.end / 60:.1f} / {total / 60:.1f} min)")
    return words


def transcribe(path: str, model_name: str, progress: Callable[[float, str], None],
               cancelled: Callable[[], bool], duration_hint: float = 0.0) -> dict:
    progress(0.0, f"Loading speech model '{model_name}' (first use downloads it)")
    model = _load(model_name)
    progress(0.0, "Reading the audio track")
    audio = _load_audio(path)
    try:
        words = _run(model, audio, progress, cancelled, duration_hint)
    except RuntimeError as exc:
        # A GPU that lacks the CUDA runtime libraries fails here, not at load time.
        if "Cancelled" in str(exc) or getattr(model.model, "device", "cpu") == "cpu":
            raise
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _model_cache.clear()
        _model_cache[model_name] = model
        words = _run(model, audio, progress, cancelled, duration_hint)

    # Keep timestamps monotonic and give every word a visible duration.
    for i, w in enumerate(words):
        if i and w["s"] < words[i - 1]["s"]:
            w["s"] = words[i - 1]["s"]
        if w["e"] <= w["s"]:
            w["e"] = w["s"] + 0.05
    return {"words": words, "sentences": build_sentences(words)}


_END = re.compile(r"[.!?][\"')\]]*$")


def build_sentences(words: list[dict], max_len: float = 14.0, gap: float = 0.9) -> list[dict]:
    """Group words into sentence-like units that clips can start and end on."""
    sentences: list[dict] = []
    start = 0
    for i, w in enumerate(words):
        is_last = i == len(words) - 1
        next_gap = (words[i + 1]["s"] - w["e"]) if not is_last else 99
        too_long = w["e"] - words[start]["s"] >= max_len and w["w"].endswith(",")
        way_too_long = w["e"] - words[start]["s"] >= max_len * 1.6
        if is_last or _END.search(w["w"]) or next_gap >= gap or too_long or way_too_long:
            sentences.append({
                "i": len(sentences),
                "s": words[start]["s"],
                "e": w["e"],
                "w0": start,
                "w1": i,
                "text": " ".join(x["w"] for x in words[start:i + 1]),
            })
            start = i + 1
    return sentences
