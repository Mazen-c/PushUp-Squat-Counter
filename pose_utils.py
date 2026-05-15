"""Small pose helpers shared by live exercise analyzers."""

from __future__ import annotations

from math import acos, atan2, degrees, hypot
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

Point = Dict[str, float]
Keypoints = Dict[str, Point]

MEDIAPIPE_INDEX_TO_NAME = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

COCO_INDEX_TO_NAME = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def normalize_keypoints(keypoints: Any, schema: str = "auto") -> Keypoints:
    """Normalize common dict/list pose formats to named point dictionaries."""
    if keypoints is None:
        return {}

    if isinstance(keypoints, Mapping):
        normalized: Keypoints = {}
        for raw_name, raw_point in keypoints.items():
            name = str(raw_name).lower().replace(" ", "_")
            point = _point_from_any(raw_point)
            if point is not None:
                normalized[name] = point
        return normalized

    if isinstance(keypoints, Sequence):
        index_map = _index_map_for(keypoints, schema)
        normalized = {}
        for index, name in index_map.items():
            if index >= len(keypoints):
                continue
            point = _point_from_any(keypoints[index])
            if point is not None:
                normalized[name] = point
        return normalized

    return {}


def get_point(
    keypoints: Any,
    name: str,
    *,
    min_visibility: float = 0.35,
    schema: str = "auto",
) -> Optional[Point]:
    point = normalize_keypoints(keypoints, schema=schema).get(name)
    if point is None:
        return None
    if point.get("visibility", 1.0) < min_visibility:
        return None
    return point


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """Return angle ABC in degrees."""
    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])
    ab_len = hypot(ab[0], ab[1])
    cb_len = hypot(cb[0], cb[1])
    if ab_len == 0 or cb_len == 0:
        return 0.0

    cosine = (ab[0] * cb[0] + ab[1] * cb[1]) / (ab_len * cb_len)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def side_angle(
    keypoints: Any,
    side: str,
    joints: Tuple[str, str, str],
    *,
    min_visibility: float = 0.35,
    schema: str = "auto",
) -> Optional[float]:
    first = get_point(
        keypoints,
        f"{side}_{joints[0]}",
        min_visibility=min_visibility,
        schema=schema,
    )
    middle = get_point(
        keypoints,
        f"{side}_{joints[1]}",
        min_visibility=min_visibility,
        schema=schema,
    )
    last = get_point(
        keypoints,
        f"{side}_{joints[2]}",
        min_visibility=min_visibility,
        schema=schema,
    )
    if first is None or middle is None or last is None:
        return None
    return angle_degrees(first, middle, last)


def average_sided_angle(
    keypoints: Any,
    joints: Tuple[str, str, str],
    *,
    min_visibility: float = 0.35,
    schema: str = "auto",
) -> Optional[float]:
    angles = [
        angle
        for angle in (
            side_angle(
                keypoints,
                "left",
                joints,
                min_visibility=min_visibility,
                schema=schema,
            ),
            side_angle(
                keypoints,
                "right",
                joints,
                min_visibility=min_visibility,
                schema=schema,
            ),
        )
        if angle is not None
    ]
    if not angles:
        return None
    return sum(angles) / len(angles)


def midpoint(
    first: Optional[Point],
    second: Optional[Point],
) -> Optional[Point]:
    if first is None and second is None:
        return None
    if first is None:
        return second
    if second is None:
        return first
    return {
        "x": (first["x"] + second["x"]) / 2.0,
        "y": (first["y"] + second["y"]) / 2.0,
        "z": (first.get("z", 0.0) + second.get("z", 0.0)) / 2.0,
        "visibility": min(first.get("visibility", 1.0), second.get("visibility", 1.0)),
    }


def torso_angle_from_vertical(
    keypoints: Any,
    *,
    min_visibility: float = 0.35,
    schema: str = "auto",
) -> Optional[float]:
    """Return torso lean in degrees, where 0 is upright and 90 is horizontal."""
    left_shoulder = get_point(
        keypoints, "left_shoulder", min_visibility=min_visibility, schema=schema
    )
    right_shoulder = get_point(
        keypoints, "right_shoulder", min_visibility=min_visibility, schema=schema
    )
    left_hip = get_point(
        keypoints, "left_hip", min_visibility=min_visibility, schema=schema
    )
    right_hip = get_point(
        keypoints, "right_hip", min_visibility=min_visibility, schema=schema
    )

    shoulder = midpoint(left_shoulder, right_shoulder)
    hip = midpoint(left_hip, right_hip)
    if shoulder is None or hip is None:
        return None

    dx = abs(shoulder["x"] - hip["x"])
    dy = abs(shoulder["y"] - hip["y"])
    if dx == 0 and dy == 0:
        return 0.0
    return degrees(atan2(dx, dy))


def _index_map_for(keypoints: Sequence[Any], schema: str) -> Mapping[int, str]:
    if schema == "mediapipe":
        return MEDIAPIPE_INDEX_TO_NAME
    if schema == "coco":
        return COCO_INDEX_TO_NAME
    return MEDIAPIPE_INDEX_TO_NAME if len(keypoints) >= 29 else COCO_INDEX_TO_NAME


def _point_from_any(raw_point: Any) -> Optional[Point]:
    if raw_point is None:
        return None

    if isinstance(raw_point, Mapping):
        x = raw_point.get("x")
        y = raw_point.get("y")
        if x is None or y is None:
            return None
        return {
            "x": float(x),
            "y": float(y),
            "z": float(raw_point.get("z", 0.0)),
            "visibility": float(
                raw_point.get("visibility", raw_point.get("score", 1.0))
            ),
        }

    if hasattr(raw_point, "x") and hasattr(raw_point, "y"):
        return {
            "x": float(raw_point.x),
            "y": float(raw_point.y),
            "z": float(getattr(raw_point, "z", 0.0)),
            "visibility": float(getattr(raw_point, "visibility", 1.0)),
        }

    if isinstance(raw_point, Iterable):
        values = list(raw_point)
        if len(values) < 2:
            return None
        return {
            "x": float(values[0]),
            "y": float(values[1]),
            "z": float(values[2]) if len(values) > 2 else 0.0,
            "visibility": float(values[3]) if len(values) > 3 else 1.0,
        }

    return None
