# main_live.py – Live Exercise Analysis System
# Run: pip install ultralytics opencv-python
#      python main_live.py
# Keys: P = Push-up mode | S = Squat mode | Q = Quit

import cv2
import json
import math
import time
import uuid

# ── Keypoint indices (COCO) ───────────────────────────────────────────────────

LEFT_SHOULDER,  RIGHT_SHOULDER  = 5,  6
LEFT_ELBOW,     RIGHT_ELBOW     = 7,  8
LEFT_WRIST,     RIGHT_WRIST     = 9,  10
LEFT_HIP,       RIGHT_HIP       = 11, 12
LEFT_KNEE,      RIGHT_KNEE      = 13, 14
LEFT_ANKLE,     RIGHT_ANKLE     = 15, 16

# ── Push-up thresholds ────────────────────────────────────────────────────────

PUSHUP_EMA_ALPHA       = 0.4
ELBOW_DOWN_THRESHOLD   = 90    # elbow angle → DOWN state
ELBOW_UP_THRESHOLD     = 160   # elbow angle → rep complete
BODY_LINE_MIN_ANGLE    = 160   # below this → bad form
KEYPOINT_CONF_MIN      = 0.3

SQUAT_EMA_ALPHA        = PUSHUP_EMA_ALPHA
KNEE_DOWN_THRESHOLD    = 100
KNEE_UP_THRESHOLD      = 160
KNEE_START_THRESHOLD   = 140

# ── Push-up state ─────────────────────────────────────────────────────────────

_pu_ema            = {}
_pu_ema_init       = False
_pu_is_down        = False
_pu_total_reps     = 0
_pu_good_reps      = 0
_pu_cur_rep_good   = True

_PU_KP_MAP = {
    "ls": LEFT_SHOULDER,  "rs": RIGHT_SHOULDER,
    "le": LEFT_ELBOW,     "re": RIGHT_ELBOW,
    "lw": LEFT_WRIST,     "rw": RIGHT_WRIST,
    "lh": LEFT_HIP,       "rh": RIGHT_HIP,
    "la": LEFT_ANKLE,     "ra": RIGHT_ANKLE,
}

class EMAKeypointSmoother:
    def __init__(self, alpha=PUSHUP_EMA_ALPHA):
        self.alpha = alpha
        self._values = {}
        self._initialized = False

    def update(self, keypoints):
        if not self._initialized:
            self._values = {
                name: dict(point)
                for name, point in keypoints.items()
            }
            self._initialized = True
            return {
                name: dict(point)
                for name, point in self._values.items()
            }

        smoothed = {}
        for name, point in keypoints.items():
            previous = self._values.get(name, {})
            next_point = dict(point)
            if "x" in point and "y" in point:
                next_point["x"] = _ema(float(previous.get("x", point["x"])), float(point["x"]), self.alpha)
                next_point["y"] = _ema(float(previous.get("y", point["y"])), float(point["y"]), self.alpha)
            self._values[name] = next_point
            smoothed[name] = dict(next_point)
        return smoothed

def build_results_payload(squat_result, pushup_result=None, *, timestamp=None):
    pushup_result = pushup_result or {}
    total_squat_reps = squat_result.get("total_squats", squat_result.get("total_reps", 0))
    good_form_squat_reps = squat_result.get("good_squats", squat_result.get("good_form_reps", 0))

    return {
        "video_id": str(uuid.uuid4()),
        "timestamp": time.time() if timestamp is None else timestamp,
        "MAZEN_TOTAL_SQUATS_VAR": total_squat_reps,
        "MAZEN_GOOD_SQUATS_VAR": good_form_squat_reps,
        "summary": {
            "squats": {
                "total_reps": total_squat_reps,
                "good_form_reps": good_form_squat_reps,
            },
            "pushups": {
                "total_reps": pushup_result.get("total_reps", 0),
                "good_form_reps": pushup_result.get("good_form_reps", 0),
            },
        },
    }

def save_results_json(path="results.json"):
    payload = build_results_payload(
        {
            "total_reps": _sq_total_reps,
            "good_form_reps": _sq_good_reps,
        },
        {
            "total_reps": _pu_total_reps,
            "good_form_reps": _pu_good_reps,
        },
    )
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload

def _kp(kps, index):
    x, y, conf = float(kps[index][0]), float(kps[index][1]), float(kps[index][2])
    return (x, y) if conf >= KEYPOINT_CONF_MIN else (0.0, 0.0)

def _ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

def calculate_angle(a, b, c):
    if (0.0, 0.0) in (a, b, c):
        return 0.0
    ang = abs(math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]) -
                            math.atan2(c[1]-b[1], c[0]-b[0])))
    return 360.0 - ang if ang > 180.0 else ang

def _pt(name):
    return (_pu_ema[name+"_x"], _pu_ema[name+"_y"])

def reset_pushup_state():
    global _pu_ema, _pu_ema_init, _pu_is_down, _pu_total_reps, _pu_good_reps, _pu_cur_rep_good
    _pu_ema          = {}
    _pu_ema_init     = False
    _pu_is_down      = False
    _pu_total_reps   = 0
    _pu_good_reps    = 0
    _pu_cur_rep_good = True

def process_pushup_frame(keypoints):
    global _pu_ema, _pu_ema_init, _pu_is_down, _pu_total_reps, _pu_good_reps, _pu_cur_rep_good

    if keypoints is None:
        return (_pu_total_reps, _pu_good_reps, True)

    raw = {k: _kp(keypoints, idx) for k, idx in _PU_KP_MAP.items()}

    if not _pu_ema_init:
        for k, (x, y) in raw.items():
            _pu_ema[k+"_x"], _pu_ema[k+"_y"] = x, y
        _pu_ema_init = True
    else:
        for k, (x, y) in raw.items():
            _pu_ema[k+"_x"] = _ema(_pu_ema[k+"_x"], x, PUSHUP_EMA_ALPHA)
            _pu_ema[k+"_y"] = _ema(_pu_ema[k+"_y"], y, PUSHUP_EMA_ALPHA)

    la = calculate_angle(_pt("ls"), _pt("le"), _pt("lw"))
    ra = calculate_angle(_pt("rs"), _pt("re"), _pt("rw"))
    elbow_angle = (la + ra) / 2.0 if (la > 0 and ra > 0) else (la or ra)

    lb = calculate_angle(_pt("ls"), _pt("lh"), _pt("la"))
    rb = calculate_angle(_pt("rs"), _pt("rh"), _pt("ra"))
    if lb > 0 and rb > 0:
        body_line = min(lb, rb)
    elif lb > 0 or rb > 0:
        body_line = lb or rb
    else:
        body_line = 180.0

    if 0.0 < body_line < BODY_LINE_MIN_ANGLE:
        _pu_cur_rep_good = False

    if elbow_angle > 0 and elbow_angle < ELBOW_DOWN_THRESHOLD and not _pu_is_down:
        _pu_is_down      = True
        _pu_cur_rep_good = True

    elif elbow_angle > ELBOW_UP_THRESHOLD and _pu_is_down:
        _pu_is_down     = False
        _pu_total_reps += 1
        if _pu_cur_rep_good:
            _pu_good_reps += 1

    return (_pu_total_reps, _pu_good_reps, _pu_cur_rep_good)

_sq_ema              = {}
_sq_ema_init         = False
_sq_in_rep           = False
_sq_depth_reached    = False
_sq_total_reps       = 0
_sq_good_reps        = 0
_sq_cur_rep_good     = True

_SQ_KP_MAP = {
    "lh": LEFT_HIP,    "rh": RIGHT_HIP,
    "lk": LEFT_KNEE,   "rk": RIGHT_KNEE,
    "la": LEFT_ANKLE,  "ra": RIGHT_ANKLE,
}

def _sq_pt(name):
    return (_sq_ema[name+"_x"], _sq_ema[name+"_y"])

def reset_squat_state():
    global _sq_ema, _sq_ema_init, _sq_in_rep, _sq_depth_reached
    global _sq_total_reps, _sq_good_reps, _sq_cur_rep_good
    _sq_ema           = {}
    _sq_ema_init      = False
    _sq_in_rep        = False
    _sq_depth_reached = False
    _sq_total_reps    = 0
    _sq_good_reps     = 0
    _sq_cur_rep_good  = True

def process_squat_frame(keypoints):
    # Return: (total_reps: int, good_reps: int, is_good_form: bool)
    global _sq_ema, _sq_ema_init, _sq_in_rep, _sq_depth_reached
    global _sq_total_reps, _sq_good_reps, _sq_cur_rep_good

    if keypoints is None:
        return (_sq_total_reps, _sq_good_reps, True)

    raw = {k: _kp(keypoints, idx) for k, idx in _SQ_KP_MAP.items()}

    if not _sq_ema_init:
        for k, (x, y) in raw.items():
            _sq_ema[k+"_x"], _sq_ema[k+"_y"] = x, y
        _sq_ema_init = True
    else:
        for k, (x, y) in raw.items():
            _sq_ema[k+"_x"] = _ema(_sq_ema[k+"_x"], x, SQUAT_EMA_ALPHA)
            _sq_ema[k+"_y"] = _ema(_sq_ema[k+"_y"], y, SQUAT_EMA_ALPHA)

    left_knee_angle = calculate_angle(_sq_pt("lh"), _sq_pt("lk"), _sq_pt("la"))
    right_knee_angle = calculate_angle(_sq_pt("rh"), _sq_pt("rk"), _sq_pt("ra"))
    knee_angle = (
        (left_knee_angle + right_knee_angle) / 2.0
        if (left_knee_angle > 0 and right_knee_angle > 0)
        else (left_knee_angle or right_knee_angle)
    )

    if knee_angle > 0 and knee_angle < KNEE_START_THRESHOLD and not _sq_in_rep:
        _sq_in_rep = True
        _sq_depth_reached = False
        _sq_cur_rep_good = True

    if _sq_in_rep and knee_angle > 0 and knee_angle < KNEE_DOWN_THRESHOLD:
        _sq_depth_reached = True
        _sq_cur_rep_good = True

    elif _sq_in_rep and knee_angle > KNEE_UP_THRESHOLD:
        if _sq_depth_reached:
            _sq_total_reps += 1
            _sq_good_reps += 1
            _sq_cur_rep_good = True
        else:
            _sq_cur_rep_good = True
        _sq_in_rep = False
        _sq_depth_reached = False

    return (_sq_total_reps, _sq_good_reps, _sq_cur_rep_good)

# ── Config ────────────────────────────────────────────────────────────────────

INFER_SIZE   = 640          # inference resolution; reduce (e.g. 320) for more speed
MODE_LABELS  = {'pushup': 'Push-ups', 'squat': 'Squats'}
KEY_TO_MODE  = {ord('p'): 'pushup', ord('s'): 'squat'}

# ── Init model & capture ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt") if __name__ == "__main__" else None

cap = cv2.VideoCapture(0) if __name__ == "__main__" else None
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640) if cap is not None else None
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) if cap is not None else None
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1) if cap is not None else None

if cap is not None and not cap.isOpened():
    raise RuntimeError("Webcam not found – check device index in VideoCapture(0)")

# ── UI helper ─────────────────────────────────────────────────────────────────

def draw_overlay(frame, mode, total_reps, good_reps, good_form):
    h  = frame.shape[0]
    pw, ph = 265, 132

    # Semi-transparent dark panel
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    x, y0 = 18, 38
    dy    = 28

    cv2.putText(frame, f"Mode : {MODE_LABELS[mode]}",  (x, y0),      font, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Reps : {total_reps}",   (x, y0+dy),   font, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Good Reps  : {good_reps}",    (x, y0+dy*2), font, 0.65, (255,255,255), 2, cv2.LINE_AA)

    label = "GOOD FORM" if good_form else "BAD FORM"
    color = (0, 220, 0)  if good_form else (0, 0, 220)
    cv2.putText(frame, label, (x, y0+dy*3), font, 0.65, color, 2, cv2.LINE_AA)

    cv2.putText(frame, "P=Push-ups  S=Squats  Q=Quit",
                (10, h - 12), font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────

mode       = 'pushup'
total_reps = good_reps = 0
good_form  = True

print("Live analysis running.  P = Push-ups  |  S = Squats  |  Q = Quit") if __name__ == "__main__" else None

while __name__ == "__main__":
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=INFER_SIZE, verbose=False)

    # Extract keypoints for the first detected person, or None
    kd        = results[0].keypoints
    keypoints = kd.data[0].cpu().numpy() if (kd is not None and len(kd.data)) else None

    if mode == 'pushup':
        total_reps, good_reps, good_form = process_pushup_frame(keypoints)
    else:
        total_reps, good_reps, good_form = process_squat_frame(keypoints)

    annotated = results[0].plot()           # draws YOLO skeleton + keypoints
    draw_overlay(annotated, mode, total_reps, good_reps, good_form)

    cv2.imshow("Live Exercise Analysis", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key in KEY_TO_MODE and KEY_TO_MODE[key] != mode:
        mode = KEY_TO_MODE[key]
        if mode == 'pushup':
            total_reps, good_reps = _pu_total_reps, _pu_good_reps
        else:
            total_reps, good_reps = _sq_total_reps, _sq_good_reps
        good_form  = True
        print(f"→ Switched to {MODE_LABELS[mode]} mode")

if __name__ == "__main__":
    payload = save_results_json()
    print("Session saved to results.json")
    print(json.dumps(payload["summary"], indent=2))

cap.release() if cap is not None else None
cv2.destroyAllWindows() if __name__ == "__main__" else None
