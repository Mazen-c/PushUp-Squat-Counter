# main_live.py – Live Exercise Analysis System
# Run: pip install ultralytics opencv-python
#      python main_live.py
# Keys: P = Push-up mode | S = Squat mode | Q = Quit

import cv2
import math
from ultralytics import YOLO

# ── Keypoint indices (COCO) ───────────────────────────────────────────────────

LEFT_SHOULDER,  RIGHT_SHOULDER  = 5,  6
LEFT_ELBOW,     RIGHT_ELBOW     = 7,  8
LEFT_WRIST,     RIGHT_WRIST     = 9,  10
LEFT_HIP,       RIGHT_HIP       = 11, 12
LEFT_ANKLE,     RIGHT_ANKLE     = 15, 16

# ── Push-up thresholds ────────────────────────────────────────────────────────

PUSHUP_EMA_ALPHA       = 0.4
ELBOW_DOWN_THRESHOLD   = 90    # elbow angle → DOWN state
ELBOW_UP_THRESHOLD     = 160   # elbow angle → rep complete
BODY_LINE_MIN_ANGLE    = 160   # below this → bad form
KEYPOINT_CONF_MIN      = 0.3

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

def _kp(kps, index):
    x, y, conf = float(kps[index][0]), float(kps[index][1]), float(kps[index][2])
    return (x, y) if conf >= KEYPOINT_CONF_MIN else (0.0, 0.0)

def _ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

def _angle(a, b, c):
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

    la = _angle(_pt("ls"), _pt("le"), _pt("lw"))
    ra = _angle(_pt("rs"), _pt("re"), _pt("rw"))
    elbow_angle = (la + ra) / 2.0 if (la > 0 and ra > 0) else (la or ra)

    lb = _angle(_pt("ls"), _pt("lh"), _pt("la"))
    rb = _angle(_pt("rs"), _pt("rh"), _pt("ra"))
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

def process_squat_frame(_keypoints):
    # Return: (total_reps: int, good_reps: int, is_good_form: bool)
    return (0, 0, True)

# ── Config ────────────────────────────────────────────────────────────────────

INFER_SIZE   = 640          # inference resolution; reduce (e.g. 320) for more speed
MODE_LABELS  = {'pushup': 'Push-ups', 'squat': 'Squats'}
KEY_TO_MODE  = {ord('p'): 'pushup', ord('s'): 'squat'}

# ── Init model & capture ──────────────────────────────────────────────────────

model = YOLO("yolov8s-pose.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # discard stale frames, minimizes lag

if not cap.isOpened():
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

print("Live analysis running.  P = Push-ups  |  S = Squats  |  Q = Quit")

while True:
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
        total_reps = good_reps = 0
        good_form  = True
        reset_pushup_state()
        print(f"→ Switched to {MODE_LABELS[mode]} mode")

cap.release()
cv2.destroyAllWindows()
