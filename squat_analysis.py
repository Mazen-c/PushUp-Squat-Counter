"""Squat repetition and form analysis from pose keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pose_utils import average_sided_angle, torso_angle_from_vertical


@dataclass
class SquatAnalyzer:
    """Stateful squat counter.

    A rep starts once the knees leave the standing range, then completes when
    the user returns upright. The rep is good only if knee depth crossed the
    configured down threshold and the torso stayed inside the lean limit.
    """

    down_threshold: float = 90.0
    up_threshold: float = 160.0
    descent_start_threshold: float = 140.0
    max_torso_lean: float = 45.0
    min_visibility: float = 0.35

    state: str = "up"
    total_squats: int = 0
    good_squats: int = 0
    bad_squats: int = 0
    last_rep_good: Optional[bool] = None
    last_bad_reasons: List[str] = field(default_factory=list)
    rep_min_knee_angle: Optional[float] = None
    rep_max_torso_lean: Optional[float] = None
    current_knee_angle: Optional[float] = None
    current_torso_angle: Optional[float] = None

    def update(self, keypoints: Any) -> Dict[str, Any]:
        knee_angle = average_sided_angle(
            keypoints,
            ("hip", "knee", "ankle"),
            min_visibility=self.min_visibility,
        )
        torso_angle = torso_angle_from_vertical(
            keypoints,
            min_visibility=self.min_visibility,
        )

        self.current_knee_angle = knee_angle
        self.current_torso_angle = torso_angle
        if knee_angle is None:
            return self.result(feedback="Need hip, knee, and ankle keypoints")

        if self.state == "up":
            self.last_rep_good = None
            self.last_bad_reasons = []
            if knee_angle < self.descent_start_threshold:
                self.state = "descending"
                self.rep_min_knee_angle = knee_angle
                self.rep_max_torso_lean = torso_angle
        else:
            self.rep_min_knee_angle = min(
                self.rep_min_knee_angle if self.rep_min_knee_angle is not None else knee_angle,
                knee_angle,
            )
            if torso_angle is not None:
                self.rep_max_torso_lean = max(
                    self.rep_max_torso_lean
                    if self.rep_max_torso_lean is not None
                    else torso_angle,
                    torso_angle,
                )

            if knee_angle >= self.up_threshold:
                self._finish_rep()

        return self.result()

    def result(self, *, feedback: Optional[str] = None) -> Dict[str, Any]:
        if feedback is None:
            feedback = self._feedback()

        return {
            "total_squats": self.total_squats,
            "good_squats": self.good_squats,
            "bad_squats": self.bad_squats,
            "state": self.state,
            "current_knee_angle": _round_or_none(self.current_knee_angle),
            "current_torso_angle": _round_or_none(self.current_torso_angle),
            "rep_min_knee_angle": _round_or_none(self.rep_min_knee_angle),
            "rep_max_torso_lean": _round_or_none(self.rep_max_torso_lean),
            "last_rep_good": self.last_rep_good,
            "last_bad_reasons": list(self.last_bad_reasons),
            "feedback": feedback,
        }

    def _finish_rep(self) -> None:
        reasons: List[str] = []
        if self.rep_min_knee_angle is None or self.rep_min_knee_angle > self.down_threshold:
            reasons.append("Not deep enough")
        if (
            self.rep_max_torso_lean is not None
            and self.rep_max_torso_lean > self.max_torso_lean
        ):
            reasons.append("Torso leaning too far")

        self.total_squats += 1
        self.last_bad_reasons = reasons
        self.last_rep_good = not reasons
        if self.last_rep_good:
            self.good_squats += 1
        else:
            self.bad_squats += 1

        self.state = "up"
        self.rep_min_knee_angle = None
        self.rep_max_torso_lean = None

    def _feedback(self) -> str:
        if self.state == "descending":
            if (
                self.rep_min_knee_angle is not None
                and self.rep_min_knee_angle > self.down_threshold
            ):
                return "Go deeper"
            if (
                self.current_torso_angle is not None
                and self.current_torso_angle > self.max_torso_lean
            ):
                return "Keep your chest up"
            return "Drive up"
        if self.last_rep_good is True:
            return "Good squat"
        if self.last_rep_good is False:
            return ", ".join(self.last_bad_reasons)
        return "Ready"


def analyze_squat(keypoints: Any, analyzer: Optional[SquatAnalyzer] = None) -> Dict[str, Any]:
    """Compatibility helper for callers that prefer a function API."""
    analyzer = analyzer or SquatAnalyzer()
    return analyzer.update(keypoints)


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 2)
