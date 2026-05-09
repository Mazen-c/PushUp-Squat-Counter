# main_live.py – Live Exercise Analysis System
# Run: pip install ultralytics opencv-python
#      python main_live.py
# Keys: P = Push-up mode | S = Squat mode | Q = Quit

import cv2
from ultralytics import YOLO

# ── Placeholder processing functions ─────────────────────────────────────────
# Each function receives `keypoints`: a numpy array [17, 3] (x, y, conf),
# or None if no person was detected in the frame.
#
# Each function must maintain its own internal state (rep counters, EMA values,
# angle state-machine, etc.) and return the CURRENT totals every frame.
#
# To plug in your real logic later, replace these two functions with:
#   from pushup_analysis import process_pushup_frame
#   from squat_analysis  import process_squat_frame

def process_pushup_frame(keypoints):
    # TODO: move elbow-angle state machine + EMA from pushup_analysis.py here
    # Return: (total_reps: int, good_reps: int, is_good_form: bool)
    return (0, 0, True)

def process_squat_frame(keypoints):
    # TODO: move knee-angle state machine + EMA from squat_analysis.py here
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
        total_reps = good_reps = 0          # reset counters on mode switch
        good_form  = True
        print(f"→ Switched to {MODE_LABELS[mode]} mode")

cap.release()
cv2.destroyAllWindows()
