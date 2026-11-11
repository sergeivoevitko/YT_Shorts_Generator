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

    import os

    faulthandler.enable()
    config.preload_runtime()
    if os.environ.get("CLIPFORGE_NO_CUDA_LIBS") != "1":
        config.enable_cuda_libraries()
    threads = int(os.environ.get("CLIPFORGE_CPU_THREADS", "0"))
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
        model, used = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads), "cpu"
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
        words = run(WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(words, fh)
    sys.stdout.flush()


def _run_worker(path: str, model_name: str, device: str, progress, cancelled,
                extra_env: dict | None = None) -> tuple[list[dict] | None, str]:
    """Start the helper process. Returns (words, error text). words is None if it died."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    tmp = tempfile.mkdtemp(prefix="clipforge_stt_")
    out_path, err_path = os.path.join(tmp, "words.json"), os.path.join(tmp, "stderr.txt")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf-8")
    env.update(extra_env or {})
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


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CHUNK_SECONDS = 600


def transcribe_cloud(path: str, progress: Callable[[float, str], None], cancelled: Callable[[], bool],
                     duration: float) -> list[dict]:
    """Speech to text through Groq's free Whisper service. The audio is sent in
    ten-minute pieces as small mono MP3 files."""
    import os
    import subprocess
    import tempfile
    import time

    import httpx

    key = config.groq_key()
    if not key:
        raise RuntimeError("Cloud speech recognition needs a Groq key. Open AI settings, choose Groq, paste a free "
                           "key and press Save. You can switch the AI provider back afterwards.")
    if duration <= 0:
        from . import reframe

        duration = reframe.probe(path)["duration"]
    tmp = tempfile.mkdtemp(prefix="clipforge_groq_")
    words: list[dict] = []
    count = max(1, int(-(-duration // CHUNK_SECONDS)))
    try:
        for k in range(count):
            if cancelled():
                raise RuntimeError("Cancelled")
            offset = k * CHUNK_SECONDS
            progress(k / count, f"Transcribing speech in the cloud (part {k + 1} of {count})")
            piece = os.path.join(tmp, f"part{k}.mp3")
            proc = subprocess.run(
                [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(offset),
                 "-t", str(CHUNK_SECONDS), "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", piece],
                capture_output=True)
            if proc.returncode != 0 or not os.path.exists(piece) or os.path.getsize(piece) < 2000:
                if k == 0:
                    raise RuntimeError("Could not read the audio track: " + proc.stderr.decode(errors="replace")[-300:])
                break
            data = None
            for attempt in range(4):
                started = time.time()
                with open(piece, "rb") as fh:
                    resp = httpx.post(
                        GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                        files={"file": (f"part{k}.mp3", fh, "audio/mpeg")},
                        data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json", "language": "en",
                              "timestamp_granularities[]": ["word", "segment"]},
                        timeout=httpx.Timeout(300.0, connect=15.0))
                config.applog(f"[speech] Groq part {k + 1}/{count}: status={resp.status_code} in {time.time() - started:.1f}s")
                if resp.status_code == 200:
                    data = resp.json()
                    break
                detail = resp.text[:300].replace(key, "***")
                if resp.status_code in (429, 500, 502, 503) and attempt < 3:
                    try:
                        wait = float(resp.headers.get("retry-after", ""))
                    except ValueError:
                        wait = 10.0 * (attempt + 1)
                    if wait > 120:
                        raise RuntimeError(f"Groq's free speech limit is used up for now (try again in {wait / 60:.0f} minutes).")
                    config.applog(f"[speech] Groq asked to slow down, waiting {wait:.0f}s: {detail}")
                    progress(k / count, f"Groq asked to slow down, waiting {wait:.0f}s")
                    time.sleep(min(max(wait, 2.0), 120.0))
                    continue
                raise RuntimeError(f"Groq speech recognition failed ({resp.status_code}: {detail}).")
            for w in (data or {}).get("words") or []:
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                start = float(w["start"]) + offset
                # Groq stretches a word's end to the next word; cap it so pauses stay visible.
                end = min(float(w["end"]) + offset, start + 0.18 + 0.075 * len(text))
                words.append({"w": text, "s": round(start, 3), "e": round(max(end, start + 0.05), 3)})
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    if not words:
        raise RuntimeError("Groq found no speech in this video.")
    progress(1.0, "Transcribing speech in the cloud")
    return words


def transcribe(path: str, model_name: str, progress: Callable[[float, str], None],
               cancelled: Callable[[], bool], duration_hint: float = 0.0, engine: str = "local") -> dict:
    """Speech to text. The local engine runs in a helper process: some failures inside
    the speech library end the whole process instead of raising an error, and that must
    never take the app down. If every local attempt fails and a Groq key is saved, the
    free cloud service is used instead."""
    words = None
    if engine == "groq":
        config.applog("[speech] using Groq cloud speech recognition")
        words = transcribe_cloud(path, progress, cancelled, duration_hint)
    else:
        attempts = [
            ("auto", {}, "first attempt"),
            ("cpu", {}, "on the processor"),
            ("cpu", {"CT2_FORCE_CPU_ISA": "AVX2", "CLIPFORGE_NO_CUDA_LIBS": "1", "CLIPFORGE_CPU_THREADS": "4",
                     "KMP_DUPLICATE_LIB_OK": "FALSE"}, "in safe mode"),
        ]
        err = ""
        for device, env, label in attempts:
            words, err = _run_worker(path, model_name, device, progress, cancelled, env)
            if words is not None:
                break
            config.applog(f"[speech] local speech recognition stopped unexpectedly {label} ({err})")
            progress(0.0, "Speech recognition failed, trying another way")
        if words is None:
            crashed = "3221225477" in err or "access violation" in err.lower()
            hint = ""
            if crashed:
                hint = (" This kind of crash is usually caused by an outdated Microsoft Visual C++ runtime: run "
                        "install.bat again (it updates it), then restart the app.")
            if config.groq_key():
                config.applog("[speech] falling back to Groq cloud speech recognition." + hint)
                progress(0.0, "Local speech recognition is not working on this PC, using the Groq cloud instead")
                words = transcribe_cloud(path, progress, cancelled, duration_hint)
            else:
                raise RuntimeError("Speech recognition could not start on this PC." + hint + " Or choose 'Groq cloud' "
                                   "as the speech engine under Analysis (free, needs a Groq key in AI settings). "
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
