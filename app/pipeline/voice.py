"""Local text-to-speech with Kokoro (runs on the CPU, no account needed)."""
from __future__ import annotations

import re
import urllib.request
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = {"kokoro-v1.0.onnx": 325_000_000, "voices-v1.0.bin": 28_000_000}

VOICES = {
    "am_michael": "Michael (US, warm male)",
    "am_fenrir": "Fenrir (US, deep male)",
    "am_onyx": "Onyx (US, low male)",
    "bm_george": "George (British, male narrator)",
    "bm_lewis": "Lewis (British, male)",
    "af_heart": "Heart (US, warm female)",
    "af_bella": "Bella (US, expressive female)",
    "af_nicole": "Nicole (US, soft female)",
    "bf_emma": "Emma (British, female)",
}
DEFAULT_VOICE = "am_michael"

_engine = None


def model_dir() -> Path:
    return config.WORKSPACE / "models"


def ensure_model(progress: Callable[[float, str], None]) -> None:
    """Download the voice model on first use (about 350 MB, one time)."""
    target = model_dir()
    target.mkdir(parents=True, exist_ok=True)
    total = sum(MODEL_FILES.values())
    done_before = 0
    for name, size in MODEL_FILES.items():
        path = target / name
        if path.exists() and path.stat().st_size > size * 0.9:
            done_before += size
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(MODEL_URL + name, timeout=60) as resp, tmp.open("wb") as fh:
                got = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    progress(min((done_before + got) / total, 1.0),
                             f"Downloading the voice model, first time only ({(done_before + got) / 1e6:.0f} of {total / 1e6:.0f} MB)")
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download the voice model ({exc}). Check your internet connection "
                               "and try again.") from exc
        tmp.replace(path)
        done_before += size


def _load():
    global _engine
    if _engine is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError("The voice engine is not installed. Run install.bat (or update.bat) again.") from exc
        _engine = Kokoro(str(model_dir() / "kokoro-v1.0.onnx"), str(model_dir() / "voices-v1.0.bin"))
    return _engine


def split_sentences(text: str) -> list[tuple[str, bool]]:
    """[(sentence, ends_paragraph)] so pauses can follow the writing's rhythm."""
    out: list[tuple[str, bool]] = []
    for para in re.split(r"\n\s*\n|\n", text.strip()):
        parts = [p.strip() for p in re.split(r"(?<=[.!?…])[\"')\]]*\s+", para.strip()) if p.strip()]
        for i, part in enumerate(parts):
            out.append((part, i == len(parts) - 1))
    return out


def synthesize(text: str, voice: str, speed: float, out_wav: Path,
               progress: Callable[[float, str], None]) -> float:
    """Write the narration to out_wav (mono 24 kHz) and return its length in seconds."""
    ensure_model(progress)
    progress(0.0, "Loading the voice")
    engine = _load()
    voice = voice if voice in VOICES else DEFAULT_VOICE
    sentences = split_sentences(text)
    if not sentences:
        raise RuntimeError("The story text is empty.")
    rate = 24000
    chunks: list[np.ndarray] = []
    for i, (sentence, para_end) in enumerate(sentences):
        samples, rate = engine.create(sentence, voice=voice, speed=speed, lang="en-us")
        samples = np.asarray(samples, dtype=np.float32)
        # Trim the engine's own leading/trailing silence so the pauses below are the only ones.
        loud = np.where(np.abs(samples) > 0.01)[0]
        if len(loud):
            samples = samples[max(0, loud[0] - rate // 50):min(len(samples), loud[-1] + rate // 12)]
        chunks.append(samples)
        if i < len(sentences) - 1:
            chunks.append(np.zeros(int(rate * (0.75 if para_end else 0.38)), np.float32))
        progress((i + 1) / len(sentences), f"Recording the voiceover ({i + 1} of {len(sentences)} sentences)")
    audio = np.concatenate(chunks)
    peak = float(np.abs(audio).max()) or 1.0
    pcm = (audio / peak * 0.89 * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())
    return len(pcm) / rate
