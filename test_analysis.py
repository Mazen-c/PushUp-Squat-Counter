from main_live import EMAKeypointSmoother, build_results_payload
from squat_analysis import SquatAnalyzer


def pose(ankle_x, ankle_y, torso_x=0.0):
    return {
        "left_hip": {"x": 0.0, "y": 0.0, "visibility": 1.0},
        "left_knee": {"x": 0.0, "y": 1.0, "visibility": 1.0},
        "left_ankle": {"x": ankle_x, "y": ankle_y, "visibility": 1.0},
        "right_hip": {"x": 0.0, "y": 0.0, "visibility": 1.0},
        "right_knee": {"x": 0.0, "y": 1.0, "visibility": 1.0},
        "right_ankle": {"x": ankle_x, "y": ankle_y, "visibility": 1.0},
        "left_shoulder": {"x": torso_x, "y": -1.0, "visibility": 1.0},
        "right_shoulder": {"x": torso_x, "y": -1.0, "visibility": 1.0},
    }


def test_good_squat_counts_when_depth_crosses_threshold():
    analyzer = SquatAnalyzer()
    for keypoints in [
        pose(0.0, 2.0),
        pose(0.8, 1.6),
        pose(1.0, 1.0),
        pose(0.0, 2.0),
    ]:
        result = analyzer.update(keypoints)

    assert result["total_squats"] == 1
    assert result["good_squats"] == 1
    assert result["bad_squats"] == 0


def test_shallow_squat_counts_bad_rep():
    analyzer = SquatAnalyzer()
    for keypoints in [
        pose(0.0, 2.0),
        pose(0.8, 1.6),
        pose(0.0, 2.0),
    ]:
        result = analyzer.update(keypoints)

    assert result["total_squats"] == 1
    assert result["good_squats"] == 0
    assert result["bad_squats"] == 1
    assert "Not deep enough" in result["last_bad_reasons"]


def test_ema_smoother_runs_before_payload_values_are_built():
    smoother = EMAKeypointSmoother(alpha=0.5)
    first = smoother.update({"left_knee": {"x": 0.0, "y": 0.0}})
    second = smoother.update({"left_knee": {"x": 1.0, "y": 1.0}})

    assert first["left_knee"]["x"] == 0.0
    assert second["left_knee"]["x"] == 0.5

    payload = build_results_payload(
        {"total_squats": 3, "good_squats": 2},
        timestamp=1.0,
    )
    assert payload["MAZEN_TOTAL_SQUATS_VAR"] == 3
    assert payload["MAZEN_GOOD_SQUATS_VAR"] == 2
