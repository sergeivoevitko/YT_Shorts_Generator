"""Video probing and face-aware crop planning for vertical reframing."""
from __future__ import annotations

import re
import subprocess
from statistics import median
from typing import Callable

import numpy as np

from .. import config

SAMPLE_FPS = 5
ANALYSIS_W = 480


def probe(path: str) -> dict:
    """Read display size, fps and duration by letting ffmpeg decode one frame."""
    proc = subprocess.run(
        [config.ffmpeg_path(), "-hide_banner", "-i", path, "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    err = proc.stderr
    info = {"width": 0, "height": 0, "fps": 30.0, "duration": 0.0, "has_audio": "Audio:" in err}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    # The output stream line reflects rotation metadata, the input line may not.
    out_part = err.split("Output #0", 1)[-1]
    m = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", out_part) or re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", err)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    if m:
        info["fps"] = float(m.group(1))
    if not info["width"]:
        raise RuntimeError("Could not read the video file. It may be corrupt or in an unsupported format.")
    return info


def crop_size(sw: int, sh: int, aw: int, ah: int) -> tuple[int, int]:
    """Largest window with aspect aw:ah that fits in the source (even numbers)."""
    target = aw / ah
    if sw / sh > target:
        ch, cw = sh, sh * target
    else:
        cw, ch = sw, sw / target
    return max(2, int(cw) // 2 * 2), max(2, int(ch) // 2 * 2)


class _FaceDetector:
    def __init__(self) -> None:
        import cv2

        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        self.cv2 = cv2
        self.yunet = None
        self.haar = None
        model = config.MODELS_DIR / "yunet.onnx"
        if model.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 50)
            except Exception:
                self.yunet = None
        if self.yunet is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def largest_face_x(self, frame: np.ndarray) -> float | None:
        """Horizontal center (0..1) of the most prominent face, or None."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            best = max(faces, key=lambda f: f[2] * f[3] * f[-1])
            if best[3] < h * 0.06:
                return None
            return float((best[0] + best[2] / 2) / w)
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.15, 6, minSize=(int(h * 0.08), int(h * 0.08)))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return float((x + fw / 2) / w)


def _sample_frames(path: str, start: float, dur: float, sw: int, sh: int):
    """Yield small BGR frames at SAMPLE_FPS, decoded by ffmpeg (any codec works)."""
    w = ANALYSIS_W
    h = max(2, int(sh * w / sw) // 2 * 2)
    cmd = [config.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", path, "-an", "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.kill()
        proc.wait()


def _pieces_for_shot(times: list[float], xs: list[float | None], end: float, deadzone: float) -> list[tuple]:
    """Turn noisy per-sample face positions into holds and short eased moves."""
    valid = [x for x in xs if x is not None]
    if not valid:
        return [(times[0], end, 0.5, 0.5)]
    # Fill gaps with the nearest detection, then median-filter the jitter away.
    filled: list[float] = []
    last = valid[0]
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    smooth = [median(filled[max(0, i - 2):i + 3]) for i in range(len(filled))]

    pieces = []
    cur = median(smooth[:max(1, SAMPLE_FPS * 3 // 2)])
    seg_start = times[0]
    streak = 0
    for i, x in enumerate(smooth):
        streak = streak + 1 if abs(x - cur) > deadzone else 0
        if streak >= SAMPLE_FPS * 3 // 5 + 1:  # the subject really moved, not a blip
            new = median(smooth[i:i + SAMPLE_FPS])
            first = times[i - streak + 1]
            if abs(new - cur) > 0.3:  # the subject jumped: treat it like a cut
                move_start = move_end = max(seg_start, first - 0.1)
            else:
                move_start = max(seg_start, first)
                move_end = min(end, move_start + 0.45)
            if move_start > seg_start:
                pieces.append((seg_start, move_start, cur, cur))
            if move_end > move_start:
                pieces.append((move_start, move_end, cur, new))
            seg_start, cur, streak = move_end, new, 0
    if end > seg_start:
        pieces.append((seg_start, end, cur, cur))
    return pieces


def plan_crop(path: str, start: float, dur: float, sw: int, sh: int, aw: int, ah: int, mode: str,
              progress: Callable[[float], None] | None = None) -> dict:
    """Return the crop window and an ffmpeg expression for its x position over time."""
    cw, ch = crop_size(sw, sh, aw, ah)
    y = (sh - ch) // 2
    max_x = sw - cw
    center = {"cw": cw, "ch": ch, "y": y, "x_expr": str(max_x // 2), "pieces": 1, "faces": 0.0}
    if mode != "smart" or max_x <= 0:
        return center

    try:
        detector = _FaceDetector()
    except Exception:
        return center
    import cv2

    shots: list[tuple[list[float], list[float | None]]] = [([], [])]
    prev_thumb = None
    found = 0
    total = max(1, int(dur * SAMPLE_FPS))
    n = 0
    for n, frame in enumerate(_sample_frames(path, start, dur, sw, sh)):
        t = n / SAMPLE_FPS
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)
        if prev_thumb is not None and float(np.abs(thumb - prev_thumb).mean()) > 30 and len(shots[-1][0]) >= 2:
            shots.append(([], []))  # hard cut in the source: reframe instantly there
        prev_thumb = thumb
        x = detector.largest_face_x(frame)
        found += x is not None
        shots[-1][0].append(t)
        shots[-1][1].append(x)
        if progress and n % 10 == 0:
            progress(min(n / total, 1.0))
    if not shots[0][0]:
        return center

    deadzone = max(0.06, (cw / sw) * 0.22)
    pieces: list[tuple] = []
    for i, (times, xs) in enumerate(shots):
        end = shots[i + 1][0][0] if i + 1 < len(shots) else dur + 1.0
        pieces.extend(_pieces_for_shot(times, xs, end, deadzone))

    def to_px(frac: float) -> int:
        return int(max(0, min(frac * sw - cw / 2, max_x)))

    terms = []
    for i, (a, b, x0, x1) in enumerate(pieces):
        p0, p1 = to_px(x0), to_px(x1)
        gate = f"gte(t,{a:.3f})*lt(t,{b:.3f})" if i < len(pieces) - 1 else f"gte(t,{a:.3f})"
        if i == 0:
            gate = f"lt(t,{b:.3f})" if len(pieces) > 1 else "1"
        if p0 == p1 or b - a < 0.01:
            terms.append(f"{gate}*{p1}")
        else:  # smoothstep ease between the two positions
            u = f"((t-{a:.3f})/{b - a:.3f})"
            terms.append(f"{gate}*({p0}+({p1 - p0})*{u}*{u}*(3-2*{u}))")
    return {"cw": cw, "ch": ch, "y": y, "x_expr": "+".join(terms), "pieces": len(pieces),
            "faces": found / max(1, n + 1)}
