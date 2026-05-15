# Push-Up and Squat Counter

Live exercise analysis system that uses YOLOv8 pose estimation and OpenCV to count push-up and squat repetitions from a webcam feed. The app tracks total reps, good-form reps, and saves a JSON summary when the session ends.

## Features

- Live webcam exercise tracking.
- Push-up rep counting using shoulder, elbow, wrist, hip, and ankle keypoints.
- Squat rep counting using hip, knee, and ankle keypoints.
- EMA smoothing for keypoints to reduce jitter.
- Form validation for push-ups and squats.
- Mode switching during one session.
- Session summary saved to `results.json`.

## Project Files

| File | Purpose |
| --- | --- |
| `main_live.py` | Main live webcam app. Run this for the demo. |
| `pushup_analysis.py` | Offline push-up video analyzer. Requires an MP4 file. |
| `squat_analysis.py` | Reusable squat analysis module used by tests. |
| `pose_utils.py` | Shared pose/keypoint utility functions. |
| `test_analysis.py` | Unit tests for squat analysis and JSON payload helpers. |
| `results.json` | Output file generated after quitting the live session. |

## Setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have the `.venv` folder, only run the install command.

## Run the Live App

```powershell
.\.venv\Scripts\python.exe main_live.py
```

Controls:

- `P`: Push-up mode
- `S`: Squat mode
- `Q`: Quit and save `results.json`

The webcam window shows:

- Current mode
- Total reps
- Good-form reps
- Current form status

## Output

When you press `Q`, the live app saves `results.json` in the project folder.

Example:

```json
{
  "video_id": "b10ec861-318d-4c6f-9e1b-a26ed241efca",
  "timestamp": 1778867881.4506135,
  "MAZEN_TOTAL_SQUATS_VAR": 3,
  "MAZEN_GOOD_SQUATS_VAR": 3,
  "summary": {
    "squats": {
      "total_reps": 3,
      "good_form_reps": 3
    },
    "pushups": {
      "total_reps": 20,
      "good_form_reps": 1
    }
  }
}
```

## How Counting Works

### Push-Ups

- Elbow angle below `90` degrees means the user reached the down position.
- Elbow angle above `160` degrees completes a rep.
- Body-line angle is checked to detect bad form.

### Squats

- Knee angle below `100` degrees means sufficient squat depth was reached.
- Knee angle above `160` degrees completes a rep.
- Shallow knee bends and walking-like movement do not count as squat reps.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Notes

- `main_live.py` is the main file for the project demo.
- `pushup_analysis.py` is only useful if you have a saved video file such as `pushup_video.mp4`.
- Make sure your full body is visible to the camera for more reliable keypoint detection.
