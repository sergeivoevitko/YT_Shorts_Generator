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

    def faces(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """[(horizontal center 0..1, prominence)] for every face in the frame."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, found = self.yunet.detect(frame)
            if found is None:
                return []
            return [(float((f[0] + f[2] / 2) / w), float(f[2] * f[3] * f[-1])) for f in found if f[3] >= h * 0.06]
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.15, 6, minSize=(int(h * 0.08), int(h * 0.08)))
        return [(float((x + fw / 2) / w), float(fw * fh)) for x, y, fw, fh in found]

    def largest_face_x(self, frame: np.ndarray, prev: float | None = None) -> float | None:
        """Center of the most prominent face. The face followed so far gets a strong
        bonus, so the frame does not flip between two people of similar size."""
        found = self.faces(frame)
        if not found:
            return None
        return max(found, key=lambda f: f[1] * (2.0 if prev is not None and abs(f[0] - prev) < 0.12 else 1.0))[0]


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


def scene_cuts(path: str, start: float, dur: float) -> list[float]:
    """Exact times (seconds from the clip start) of hard cuts in the source, found by
    checking every frame. The value is the time of the first frame of the new shot."""
    cmd = [config.ffmpeg_path(), "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path, "-an",
           "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo", "-f", "null", "-"]
    try:
        err = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900).stderr
    except Exception:
        return []
    cuts = [float(t) for t in re.findall(r"pts_time:\s*([\d.]+)", err)]
    out: list[float] = []
    for t in sorted(cuts):
        if 0.05 < t < dur - 0.05 and (not out or t - out[-1] > 0.3):
            out.append(t)
    return out


def _pieces_for_shot(times: list[float], xs: list[float | None], end: float, deadzone: float,
                     begin: float | None = None) -> list[tuple]:
    """Turn noisy per-sample face positions into holds and short eased moves."""
    begin = times[0] if begin is None else begin
    valid = [x for x in xs if x is not None]
    if not valid:
        return [(begin, end, 0.5, 0.5)]
    # Fill gaps with the nearest detection, then median-filter the jitter away.
    filled: list[float] = []
    last = valid[0]
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    smooth = [median(filled[max(0, i - 3):i + 4]) for i in range(len(filled))]

    pieces = []
    cur = median(smooth[:max(1, SAMPLE_FPS * 3 // 2)])
    seg_start = begin
    streak = 0
    for i, x in enumerate(smooth):
        streak = streak + 1 if abs(x - cur) > deadzone else 0
        far = abs(x - cur) > 0.3
        # Only react to a change that lasts: about 1.2 s for a drift, 1.8 s for a jump to
        # another person. The plan is made in advance, so the move still starts on time.
        if streak >= (SAMPLE_FPS * 9 // 5 if far else SAMPLE_FPS * 6 // 5):
            new = median(smooth[i:i + SAMPLE_FPS * 2])
            first = times[i - streak + 1]
            if abs(new - cur) > 0.3:
                # A jump with no cut in the picture (hard cuts are split off before this point).
                # The exact frame is unknown, so glide quickly instead of snapping: a snap that
                # lands a few frames off shows as stray frames.
                move_start = max(seg_start, first - 0.15)
                move_end = min(end, move_start + 0.4)
            else:  # glide slowly; start a little early so the subject never touches the edge
                move_start = max(seg_start, first - 0.3)
                move_end = min(end, move_start + 1.0)
            if move_start > seg_start:
                pieces.append((seg_start, move_start, cur, cur))
            if move_end > move_start:
                pieces.append((move_start, move_end, cur, new))
            seg_start, cur, streak = move_end, new, 0
    if end > seg_start:
        pieces.append((seg_start, end, cur, cur))
    return pieces


def plan_crop(path: str, start: float, dur: float, sw: int, sh: int, aw: int, ah: int, mode: str,
              progress: Callable[[float], None] | None = None, src_fps: float = 30.0,
              out_fps: float = 30.0) -> dict:
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

    # Hard cuts are located to the exact frame, so the crop can switch on the same frame.
    cuts = scene_cuts(path, start, dur)
    bounds = [0.0] + cuts
    shots: list[tuple[list[float], list[float | None]]] = [([], []) for _ in bounds]
    follow: float | None = None
    last_shot = -1
    found = 0
    total = max(1, int(dur * SAMPLE_FPS))
    n = 0
    for n, frame in enumerate(_sample_frames(path, start, dur, sw, sh)):
        t = n / SAMPLE_FPS
        si = max(i for i, b in enumerate(bounds) if b <= t + 1e-6)
        if si != last_shot:
            follow, last_shot = None, si  # new shot: pick the most prominent face afresh
        x = detector.largest_face_x(frame, follow)
        follow = x if x is not None else follow
        found += x is not None
        shots[si][0].append(t)
        shots[si][1].append(x)
        if progress and n % 10 == 0:
            progress(min(n / total, 1.0))
    if not any(s[0] for s in shots):
        return center

    deadzone = max(0.07, (cw / sw) * 0.28)
    # Switch slightly before the new shot's first frame time so that frame is already reframed.
    edge = lambda t: max(0.0, t - 0.6 / max(src_fps, out_fps))
    pieces: list[tuple] = []
    for i, (times, xs) in enumerate(shots):
        begin = edge(bounds[i]) if i else 0.0
        end = edge(bounds[i + 1]) if i + 1 < len(shots) else dur + 1.0
        if not times:  # a shot shorter than the sampling step: keep the framing it came from
            prev_x = pieces[-1][3] if pieces else 0.5
            pieces.append((begin, end, prev_x, prev_x))
            continue
        pieces.extend(_pieces_for_shot(times, xs, end, deadzone, begin))

    def to_px(frac: float) -> int:
        return int(max(0, min(frac * sw - cw / 2, max_x)))

    terms = []
    for i, (a, b, x0, x1) in enumerate(pieces):
        p0, p1 = to_px(x0), to_px(x1)
        gate = f"gte(t,{a:.4f})*lt(t,{b:.4f})" if i < len(pieces) - 1 else f"gte(t,{a:.4f})"
        if i == 0:
            gate = f"lt(t,{b:.4f})" if len(pieces) > 1 else "1"
        if p0 == p1 or b - a < 0.01:
            terms.append(f"{gate}*{p1}")
        else:  # smoothstep ease between the two positions
            u = f"((t-{a:.4f})/{b - a:.4f})"
            terms.append(f"{gate}*({p0}+({p1 - p0})*{u}*{u}*(3-2*{u}))")
    return {"cw": cw, "ch": ch, "y": y, "x_expr": "+".join(terms), "pieces": len(pieces), "cuts": cuts,
            "faces": found / max(1, n + 1)}
