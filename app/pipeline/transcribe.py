"""Speech to text with word-level timestamps (faster-whisper)."""
from __future__ import annotations

import re
from typing import Callable

from .. import config

MODELS = {
    "tiny.en": "Tiny (fastest, least accurate)",
    "base.en": "Base (fast)",
    "small.en": "Small (recommended)",
    "medium.en": "Medium (slow on CPU, most accurate)",
}

def _load_audio(path: str):
    """Decode to 16 kHz mono float32 with ffmpeg, which reads any container."""
    import subprocess

    import numpy as np

    proc = subprocess.run(
        [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("Could not read the audio track: " + proc.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(proc.stdout, np.float32)


def _worker(path: str, model_name: str, device: str, out_path: str) -> None:
    """Runs in a separate process (see transcribe). Prints progress lines on stdout."""
    import faulthandler
    import json
    import sys

    faulthandler.enable()
    config.enable_cuda_libraries()
    from faster_whisper import WhisperModel

    def say(tag: str, text: str) -> None:
        print(f"{tag} {text}", flush=True)

    say("M", f"Loading speech model '{model_name}' (first use downloads it)")
    model, used = None, "cpu"
    if device == "auto":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                model, used = WhisperModel(model_name, device="cuda", compute_type="float16"), "cuda"
        except Exception as exc:
            say("L", f"graphics card not usable for speech recognition ({exc}); using the processor")
            model = None
    if model is None:
        model, used = WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu"
    say("D", used)
    say("M", "Reading the audio track")
    audio = _load_audio(path)

    def run(m) -> list[dict]:
        segments, info = m.transcribe(
            audio, language="en", word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400}, beam_size=5, condition_on_previous_text=False)
        total = info.duration or 1.0
        words: list[dict] = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append({"w": text, "s": round(float(w.start), 3), "e": round(float(w.end), 3)})
            say("P", f"{min(seg.end / total, 1.0):.4f} {seg.end:.1f} {total:.1f}")
        return words

    try:
        words = run(model)
    except RuntimeError as exc:
        if used != "cuda":
            raise
        say("L", f"graphics card failed during speech recognition ({exc}); using the processor")
        say("D", "cpu")
        words = run(WhisperModel(model_name, device="cpu", compute_type="int8"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(words, fh)
    sys.stdout.flush()


def _run_worker(path: str, model_name: str, device: str, progress, cancelled) -> tuple[list[dict] | None, str]:
    """Start the helper process. Returns (words, error text). words is None if it died."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    tmp = tempfile.mkdtemp(prefix="clipforge_stt_")
    out_path, err_path = os.path.join(tmp, "words.json"), os.path.join(tmp, "stderr.txt")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf-8")
    with open(err_path, "w", encoding="utf-8", errors="replace") as err_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.pipeline.transcribe", path, model_name, device, out_path],
            cwd=str(config.ROOT), env=env, stdout=subprocess.PIPE, stderr=err_file, text=True, errors="replace")
        used = "cpu"
        try:
            for line in proc.stdout:
                if cancelled():
                    proc.kill()
                    raise RuntimeError("Cancelled")
                tag, _, rest = line.strip().partition(" ")
                if tag == "M":
                    progress(0.0, rest)
                elif tag == "D":
                    used = rest
                    config.applog(f"[speech] model {model_name} running on the "
                                  + ("graphics card (GPU)" if used == "cuda" else "processor (CPU)"))
                elif tag == "L":
                    config.applog(f"[speech] {rest}")
                elif tag == "P":
                    try:
                        frac, done, total = (float(x) for x in rest.split())
                    except ValueError:
                        continue
                    where = " on the graphics card" if used == "cuda" else ""
                    progress(frac, f"Transcribing speech{where} ({done / 60:.1f} / {total / 60:.1f} min)")
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
    try:
        err = open(err_path, encoding="utf-8", errors="replace").read()
    except OSError:
        err = ""
    words = None
    if code == 0 and os.path.exists(out_path):
        try:
            words = json.load(open(out_path, encoding="utf-8"))
        except ValueError:
            words = None
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    lines = [l for l in err.splitlines() if l.strip() and "UserWarning" not in l and "warnings.warn" not in l
             and "symlink" not in l.lower() and "HF_TOKEN" not in l]
    return words, f"exit code {code}. " + " | ".join(lines[-6:])[-700:]


def transcribe(path: str, model_name: str, progress: Callable[[float, str], None],
               cancelled: Callable[[], bool], duration_hint: float = 0.0) -> dict:
    """Speech to text in a helper process. Some failures inside the speech library (a
    graphics card without its libraries, for one) end the whole process instead of
    raising an error, and that must never take the app down."""
    words, err = _run_worker(path, model_name, "auto", progress, cancelled)
    if words is None:
        config.applog(f"[speech] the helper process stopped unexpectedly ({err}). Trying again on the processor.")
        progress(0.0, "Speech recognition failed on the graphics card, retrying on the processor")
        words, err = _run_worker(path, model_name, "cpu", progress, cancelled)
    if words is None:
        config.applog(f"[speech] failed on the processor too ({err})")
        raise RuntimeError("Speech recognition stopped unexpectedly. Try a smaller speech model under Analysis. "
                           f"Details: {err}")

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


if __name__ == "__main__":
    import sys

    _worker(*sys.argv[1:5])
