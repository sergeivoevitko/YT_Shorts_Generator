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

    def boxes(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """[(center x, center y, width, height, prominence)] of every face, as fractions of the frame."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, found = self.yunet.detect(frame)
            if found is None:
                return []
            return [(float((f[0] + f[2] / 2) / w), float((f[1] + f[3] / 2) / h), float(f[2] / w), float(f[3] / h),
                     float(f[2] * f[3] * f[-1])) for f in found if f[3] >= h * 0.06]
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.15, 6, minSize=(int(h * 0.08), int(h * 0.08)))
        return [(float((x + fw / 2) / w), float((y + fh / 2) / h), float(fw / w), float(fh / h), float(fw * fh))
                for x, y, fw, fh in found]

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


def _mouth_patch(gray: np.ndarray, box: tuple) -> np.ndarray:
    """Small normalised picture of the lower face, used to tell who is talking."""
    import cv2

    fh, fw = gray.shape[:2]
    cx, cy, w, h = box[:4]
    x0, x1 = int((cx - w * 0.5) * fw), int((cx + w * 0.5) * fw)
    y0, y1 = int(cy * fh), int((cy + h * 0.65) * fh)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    if x1 - x0 < 4 or y1 - y0 < 3:
        return np.zeros((12, 24), np.float32)
    return cv2.resize(gray[y0:y1, x0:x1], (24, 12)).astype(np.float32)


def _two_people(samples: list[list[tuple]]) -> tuple[dict, dict] | None:
    """If two faces stay on screen side by side through this shot, describe both.
    samples: for each sampled frame, [(cx, cy, w, h, weight, mouth patch)]."""
    if len(samples) < SAMPLE_FPS * 3 // 2:
        return None
    flat = sorted(((b[0], i, b) for i, boxes in enumerate(samples) for b in boxes), key=lambda t: t[0])
    if not flat:
        return None
    clusters: list[list[tuple]] = [[flat[0]]]
    for item in flat[1:]:
        if item[0] - clusters[-1][-1][0] > 0.10:
            clusters.append([])
        clusters[-1].append(item)
    people = []
    for cl in clusters:
        seen = {i for _, i, _ in cl}
        if len(seen) / len(samples) < 0.35:
            continue  # someone passing, or a false detection
        by_sample: dict[int, tuple] = {}
        for _, i, b in cl:
            if i not in by_sample or b[4] > by_sample[i][4]:
                by_sample[i] = b
        # How much the lower face changes from one sample to the next: talking moves it.
        activity = [0.0] * len(samples)
        for i in range(1, len(samples)):
            if i in by_sample and i - 1 in by_sample:
                activity[i] = float(np.abs(by_sample[i][5] - by_sample[i - 1][5]).mean())
        boxes = list(by_sample.values())
        people.append({"cx": median(b[0] for b in boxes), "cy": median(b[1] for b in boxes),
                       "w": median(b[2] for b in boxes), "h": median(b[3] for b in boxes),
                       "presence": len(seen) / len(samples), "activity": activity})
    if len(people) < 2:
        return None
    people.sort(key=lambda p: -p["presence"] * p["h"])
    a, b = sorted(people[:2], key=lambda p: p["cx"])
    if min(a["h"], b["h"]) < 0.5 * max(a["h"], b["h"]) or b["cx"] - a["cx"] < 0.18:
        return None  # one is far in the background, or the "two" are one person
    return a, b


def _speaker_pieces(times: list[float], a: dict, b: dict, begin: float, end: float) -> list[tuple]:
    """Cut between two people according to who is talking. Returns crop holds (t0, t1, x, x)."""
    def smooth(v: list[float]) -> list[float]:
        # A median first: one jumpy face detection makes a single large spike, talking does not.
        med = [median(v[max(0, i - 2):i + 3]) for i in range(len(v))]
        return [sum(med[max(0, i - 3):i + 4]) / len(med[max(0, i - 3):i + 4]) for i in range(len(med))]

    def relative(v: list[float]) -> list[float]:
        # Each person is measured against their own busiest moments, because a bigger or
        # better lit face always shows more raw change than a smaller one.
        top = float(np.percentile(v, 85)) if v else 0.0
        return [min(x / top, 2.0) if top > 1e-3 else 0.0 for x in v]

    act = [relative(smooth(a["activity"])), relative(smooth(b["activity"]))]
    xs = [a["cx"], b["cx"]]
    lead = max(1, min(len(times), SAMPLE_FPS * 2))
    cur = 0 if sum(act[0][:lead]) >= sum(act[1][:lead]) else 1
    pieces, seg_start, streak, last_switch = [], begin, 0, begin
    for i, t in enumerate(times):
        other = 1 - cur
        stronger = act[other][i] - act[cur][i] > 0.3
        streak = streak + 1 if stronger else 0
        if streak >= SAMPLE_FPS and t - last_switch >= 2.0:  # a full second of the other person talking
            switch_at = max(seg_start + 0.2, times[i - streak + 1])
            pieces.append((seg_start, switch_at, xs[cur], xs[cur]))
            seg_start, last_switch, cur, streak = switch_at, switch_at, other, 0
    pieces.append((seg_start, end, xs[cur], xs[cur]))
    return pieces


def plan_crop(path: str, start: float, dur: float, sw: int, sh: int, aw: int, ah: int, mode: str,
              progress: Callable[[float], None] | None = None, src_fps: float = 30.0,
              out_fps: float = 30.0, two_people: str = "split") -> dict:
    """Return the crop window and an ffmpeg expression for its x position over time."""
    cw, ch = crop_size(sw, sh, aw, ah)
    y = (sh - ch) // 2
    max_x = sw - cw
    center = {"cw": cw, "ch": ch, "y": y, "x_expr": str(max_x // 2), "pieces": 1, "faces": 0.0, "split": None,
              "duo_shots": 0, "speaker_cuts": 0}
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
    shot_boxes: list[list[list[tuple]]] = [[] for _ in bounds]
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
        boxes = detector.boxes(frame)
        if boxes:
            x = max(boxes, key=lambda f: f[4] * (2.0 if follow is not None and abs(f[0] - follow) < 0.12 else 1.0))[0]
        else:
            x = None
        follow = x if x is not None else follow
        found += x is not None
        shots[si][0].append(t)
        shots[si][1].append(x)
        if two_people != "one" and len(boxes) >= 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shot_boxes[si].append([b + (_mouth_patch(gray, b),) for b in boxes])
        else:
            shot_boxes[si].append([b + (None,) for b in boxes])
        if progress and n % 10 == 0:
            progress(min(n / total, 1.0))
    if not any(s[0] for s in shots):
        return center

    deadzone = max(0.07, (cw / sw) * 0.28)
    # Switch slightly before the new shot's first frame time so that frame is already reframed.
    edge = lambda t: max(0.0, t - 0.6 / max(src_fps, out_fps))
    pieces: list[tuple] = []
    duos: list[tuple[float, float, dict, dict]] = []   # shots that show two people: (begin, end, left, right)
    speaker_cuts = 0
    for i, (times, xs) in enumerate(shots):
        begin = edge(bounds[i]) if i else 0.0
        end = edge(bounds[i + 1]) if i + 1 < len(shots) else dur + 1.0
        if not times:  # a shot shorter than the sampling step: keep the framing it came from
            prev_x = pieces[-1][3] if pieces else 0.5
            pieces.append((begin, end, prev_x, prev_x))
            continue
        duo = None
        if two_people != "one":
            # Frames with a single face carry no mouth patch; they count as "that person only".
            usable = [[b if b[5] is not None else b[:5] + (np.zeros((12, 24), np.float32),) for b in boxes]
                      for boxes in shot_boxes[i]]
            duo = _two_people(usable)
        if duo:
            duos.append((begin, end, duo[0], duo[1]))
            if two_people in ("speaker", "mix"):
                cut_pieces = _speaker_pieces(times, duo[0], duo[1], begin, end)
                speaker_cuts += len(cut_pieces) - 1
                pieces.extend(cut_pieces)
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
    # Split screen: both people, one above the other, for the shots where two are on screen.
    split = None
    if two_people in ("split", "mix") and duos and ah > aw:
        ratio = (aw * 2) / ah                       # each panel is the full width and half the height
        gap_px = min((b["cx"] - a["cx"]) * sw for _, _, a, b in duos)
        face_px = max(max(a["w"], b["w"]) * sw for _, _, a, b in duos)
        # Zoom in on each person: the face takes about a third of the panel width. Never wider
        # than the gap between the two (the other person would show) or tighter than the face.
        pw = min(max(face_px * 3.2, min(480.0, sw * 0.3)), gap_px * 1.12, sh * ratio, sw * 0.62)
        if pw >= face_px * 1.5:
            pw = int(pw) // 2 * 2
            ph = min(sh, int(pw / ratio)) // 2 * 2
            pw = min(pw, int(ph * ratio) // 2 * 2)

            def exprs(which: int) -> tuple[str, str]:
                tx, ty = [], []
                for begin, end, a, b in duos:
                    p = (a, b)[which]
                    px = int(max(0, min(p["cx"] * sw - pw / 2, sw - pw)))
                    # The top person sits mid-panel, clear of the hook text; the bottom person sits
                    # high in their panel so the captions land on their chest, not their face.
                    py = int(max(0, min(p["cy"] * sh - ph * (0.56 if which == 0 else 0.30), sh - ph)))
                    gate = f"gte(t,{begin:.4f})*lt(t,{end:.4f})"
                    tx.append(f"{gate}*{px}")
                    ty.append(f"{gate}*{py}")
                return "+".join(tx), "+".join(ty)

            (ax, ay), (bx, by) = exprs(0), exprs(1)
            if two_people == "mix":
                # Variety: alternate about every five seconds between the split screen and a
                # close-up of whoever is talking.
                windows = []
                for begin, end, _, _ in duos:
                    t, on = begin, True
                    while t < min(end, dur):
                        nxt = min(t + 5.0, end)
                        if min(end, dur) - nxt < 2.0:
                            nxt = end
                        if on:
                            windows.append((t, nxt))
                        t, on = nxt, not on
            else:
                windows = [(begin, end) for begin, end, _, _ in duos]
            split = {"pw": pw, "ph": ph, "ax": ax, "ay": ay, "bx": bx, "by": by, "windows": len(windows),
                     "enable": "+".join(f"gte(t,{a_:.4f})*lt(t,{b_:.4f})" for a_, b_ in windows)}
    return {"cw": cw, "ch": ch, "y": y, "x_expr": "+".join(terms), "pieces": len(pieces), "cuts": cuts,
            "split": split, "duo_shots": len(duos), "speaker_cuts": speaker_cuts,
            "faces": found / max(1, n + 1)}
