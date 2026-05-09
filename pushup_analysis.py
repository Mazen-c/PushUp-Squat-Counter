# pushup_analysis.py  –  University Project: Exercise Analysis (Push-Up Module)
# Run:  pip install ultralytics opencv-python
#       python pushup_analysis.py

import math
import json
import uuid
import cv2
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_PATH = "pushup_video.mp4"

EMA_ALPHA            = 0.4   # higher = faster response, less smooth
ELBOW_DOWN_THRESHOLD = 90    # degrees – triggers DOWN state
ELBOW_UP_THRESHOLD   = 160   # degrees – triggers rep completion
BODY_LINE_MIN_ANGLE  = 160   # degrees – below this = bad form (hips sagging/piking)
KEYPOINT_CONF_MIN    = 0.3

# ── Squat placeholders (Mazen: replace these two values with your results) ────

MAZEN_TOTAL_SQUATS_VAR = 0
MAZEN_GOOD_SQUATS_VAR  = 0

# ── COCO keypoint indices ─────────────────────────────────────────────────────

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW,    RIGHT_ELBOW    = 7, 8
LEFT_WRIST,    RIGHT_WRIST    = 9, 10
LEFT_HIP,      RIGHT_HIP      = 11, 12
LEFT_ANKLE,    RIGHT_ANKLE    = 15, 16

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_keypoint(kps, index):
    x, y, conf = float(kps[index][0]), float(kps[index][1]), float(kps[index][2])
    return (x, y) if conf >= KEYPOINT_CONF_MIN else (0.0, 0.0)


def apply_ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev


def calculate_angle(a, b, c):
    # Returns the interior angle at B formed by A–B–C, in degrees [0, 180].
    if (0.0, 0.0) in (a, b, c):
        return 0.0
    angle = abs(math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]) -
                              math.atan2(c[1]-b[1], c[0]-b[0])))
    return 360.0 - angle if angle > 180.0 else angle

# ── Load model & video ────────────────────────────────────────────────────────

print("Loading YOLOv8 pose model...")
model = YOLO("yolov8s-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: cannot open {VIDEO_PATH}")
    exit(1)

print(f"Processing {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames...\n" + "-"*60)

# ── EMA state (one float per coordinate) ─────────────────────────────────────

ema = {}   # keyed by name, e.g. "ls_x", "ls_y", ...
ema_initialized = False

EMA_KEYS = ["ls", "rs", "le", "re", "lw", "rw", "lh", "rh", "la", "ra"]

# ── Rep-counting state ────────────────────────────────────────────────────────

is_down          = False
total_reps       = 0
good_form_reps   = 0
current_rep_good = True
frame_number     = 0

# ── Main loop ─────────────────────────────────────────────────────────────────

KP_MAP = {
    "ls": LEFT_SHOULDER,  "rs": RIGHT_SHOULDER,
    "le": LEFT_ELBOW,     "re": RIGHT_ELBOW,
    "lw": LEFT_WRIST,     "rw": RIGHT_WRIST,
    "lh": LEFT_HIP,       "rh": RIGHT_HIP,
    "la": LEFT_ANKLE,     "ra": RIGHT_ANKLE,
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    results        = model(frame, verbose=False)
    keypoints_data = results[0].keypoints
    if keypoints_data is None or len(keypoints_data.data) == 0:
        continue

    kps = keypoints_data.data[0]
    raw = {k: extract_keypoint(kps, idx) for k, idx in KP_MAP.items()}

    # Seed EMA on first valid frame; blend on every subsequent frame
    if not ema_initialized:
        for k, (x, y) in raw.items():
            ema[k+"_x"], ema[k+"_y"] = x, y
        ema_initialized = True
    else:
        for k, (x, y) in raw.items():
            ema[k+"_x"] = apply_ema(ema[k+"_x"], x, EMA_ALPHA)
            ema[k+"_y"] = apply_ema(ema[k+"_y"], y, EMA_ALPHA)

    def pt(k):
        return (ema[k+"_x"], ema[k+"_y"])

    # Elbow angle – average both sides, fall back to whichever is detected
    la = calculate_angle(pt("ls"), pt("le"), pt("lw"))
    ra = calculate_angle(pt("rs"), pt("re"), pt("rw"))
    if la > 0 and ra > 0:
        elbow_angle = (la + ra) / 2.0
    else:
        elbow_angle = la or ra

    # Body-line angle – worst of both sides flags bad form
    lb = calculate_angle(pt("ls"), pt("lh"), pt("la"))
    rb = calculate_angle(pt("rs"), pt("rh"), pt("ra"))
    if lb > 0 and rb > 0:
        body_line_angle = min(lb, rb)
    elif lb > 0 or rb > 0:
        body_line_angle = lb or rb
    else:
        body_line_angle = 180.0   # undetected – don't penalize

    # Form check: flag bad form if hips break alignment at any point in the rep
    if 0.0 < body_line_angle < BODY_LINE_MIN_ANGLE:
        current_rep_good = False

    # State machine
    if elbow_angle > 0 and elbow_angle < ELBOW_DOWN_THRESHOLD and not is_down:
        is_down          = True
        current_rep_good = True   # reset for this new rep

    elif elbow_angle > ELBOW_UP_THRESHOLD and is_down:
        is_down     = False
        total_reps += 1
        if current_rep_good:
            good_form_reps += 1
        print(f"  Frame {frame_number:5d} | Rep #{total_reps} – "
              f"{'GOOD' if current_rep_good else 'BAD'} FORM")

    if frame_number % 30 == 0:
        print(f"  Frame {frame_number:5d} | Elbow: {elbow_angle:6.1f}° | "
              f"Body: {body_line_angle:6.1f}° | is_down: {is_down} | Reps: {total_reps}")

cap.release()

# ── Results ───────────────────────────────────────────────────────────────────

print("-"*60)
print(f"\nFrames processed : {frame_number}")
print(f"Total reps       : {total_reps}")
print(f"Good-form reps   : {good_form_reps}\n")

output_data = {
    "video_id": str(uuid.uuid4()),
    "summary": {
        "squats": {
            "total_reps":     MAZEN_TOTAL_SQUATS_VAR,
            "good_form_reps": MAZEN_GOOD_SQUATS_VAR
        },
        "pushups": {
            "total_reps":     total_reps,
            "good_form_reps": good_form_reps
        }
    }
}

with open("results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("--- JSON output (saved to results.json) ---")
print(json.dumps(output_data, indent=2))
