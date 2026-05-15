"""Live webcam runner for squat counting.

The analysis modules are pure Python; this runner uses MediaPipe and OpenCV
when they are available. Keypoints are EMA-smoothed here before any exercise
logic sees them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pose_utils import normalize_keypoints
from squat_analysis import SquatAnalyzer


class EMAKeypointSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1]")
        self.alpha = alpha
        self._state: Dict[str, Dict[str, float]] = {}

    def update(self, raw_keypoints: Any) -> Dict[str, Dict[str, float]]:
        keypoints = normalize_keypoints(raw_keypoints)
        smoothed: Dict[str, Dict[str, float]] = {}

        for name, point in keypoints.items():
            previous = self._state.get(name)
            if previous is None:
                current = dict(point)
            else:
                current = {
                    "x": self._ema(previous["x"], point["x"]),
                    "y": self._ema(previous["y"], point["y"]),
                    "z": self._ema(previous.get("z", 0.0), point.get("z", 0.0)),
                    "visibility": point.get("visibility", 1.0),
                }
            smoothed[name] = current

        self._state.update(smoothed)
        return dict(self._state)

    def reset(self) -> None:
        self._state.clear()

    def _ema(self, previous: float, current: float) -> float:
        return self.alpha * current + (1.0 - self.alpha) * previous


def build_results_payload(
    squat_result: Mapping[str, Any],
    *,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    total_squats = int(squat_result.get("total_squats", 0))
    good_squats = int(squat_result.get("good_squats", 0))

    return {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "squats": dict(squat_result),
        "total_squats": total_squats,
        "good_squats": good_squats,
        "MAZEN_TOTAL_SQUATS_VAR": total_squats,
        "MAZEN_GOOD_SQUATS_VAR": good_squats,
    }


def write_results(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_live(args: argparse.Namespace) -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "main_live.py needs opencv-python and mediapipe for webcam mode. "
            "Install them, then run again."
        ) from exc

    mp_pose = None
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        mp_pose = mp.solutions.pose
    else:
        try:
            from mediapipe.python.solutions import pose as mp_pose  # type: ignore[assignment]
        except Exception:
            import sys

            raise SystemExit(
                "MediaPipe was imported, but the legacy Pose API is not available.\n"
                "Expected either `mediapipe.solutions.pose` or `mediapipe.python.solutions.pose`.\n\n"
                f"Detected Python: {sys.version.split()[0]}\n"
                f"Imported mediapipe from: {getattr(mp, '__file__', '<unknown>')}\n\n"
                "Fix (recommended):\n"
                "- Use Python 3.10–3.12\n"
                "- Pin MediaPipe to a build that still bundles `solutions`, e.g. `mediapipe==0.10.14`\n"
                "- Reinstall: pip install -r requirements.txt\n"
            )

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    smoother = EMAKeypointSmoother(alpha=args.ema_alpha)
    squat_analyzer = SquatAnalyzer()
    output_path = Path(args.output)

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            if result.pose_landmarks:
                raw_keypoints = result.pose_landmarks.landmark
                smoothed_keypoints = smoother.update(raw_keypoints)
                squat_result = squat_analyzer.update(smoothed_keypoints)
            else:
                squat_result = squat_analyzer.result(feedback="No pose detected")

            payload = build_results_payload(squat_result)
            write_results(output_path, payload)

            if args.display:
                _draw_overlay(cv2, frame, payload)
                cv2.imshow("Exercise counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    capture.release()
    if args.display:
        cv2.destroyAllWindows()


def _draw_overlay(cv2: Any, frame: Any, payload: Mapping[str, Any]) -> None:
    lines = [
        f"Squats: {payload['good_squats']}/{payload['total_squats']}",
        f"Squat: {payload['squats'].get('feedback', 'Ready')}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live squat counter")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--output", default="live_results.json", help="JSON output path")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA smoothing alpha")
    parser.add_argument("--display", action="store_true", help="Show annotated webcam window")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run_live(parse_args())
