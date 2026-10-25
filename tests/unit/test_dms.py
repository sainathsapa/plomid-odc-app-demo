"""DMS unit tests — pure geometry/tracker logic, no model weights."""

from __future__ import annotations

import math

from odc.dms import DmsTracker, FaceResult, ear, yaw_pitch_from_matrix


def _eye(height: float) -> list[tuple[float, float]]:
    return [(0, 0), (3, -height), (7, -height), (10, 0), (7, height), (3, height)]


def test_ear_separates_open_and_closed():
    assert ear(_eye(4.0)) > 0.5
    assert ear(_eye(0.5)) < 0.2


def test_yaw_pitch_identity_and_yawed():
    yaw, pitch = yaw_pitch_from_matrix(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    assert (yaw, pitch) == (0.0, 0.0)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    yaw, pitch = yaw_pitch_from_matrix(
        [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]
    )
    assert abs((yaw or 0) - 30.0) < 1.0 and abs(pitch or 0) < 1.0
    assert yaw_pitch_from_matrix([0.0] * 5) == (None, None)


def _face(**kw) -> FaceResult:
    base = {"bbox": (0.3, 0.2, 0.6, 0.7), "blink": 0.0, "jaw": 0.0,
            "ear_value": 0.6, "yaw": 0.0, "pitch": 0.0, "dots": {}, "closed": False}
    base.update(kw)
    return FaceResult(**base)  # type: ignore[arg-type]


def test_tracker_drowsy_needs_sustained_closure():
    tracker = DmsTracker()
    state = None
    for _ in range(5):
        state = tracker.update("A", True, _face())
    assert state is not None and state.drowsy is False
    for _ in range(10):
        state = tracker.update("A", True, _face(closed=True, blink=0.9))
    assert state is not None and state.drowsy is True
    assert tracker.update("B", False, None).face is False


def test_tracker_yawn_and_distraction():
    tracker = DmsTracker()
    state = None
    for _ in range(3):
        state = tracker.update("A", True, _face(jaw=0.8))
    assert state is not None and state.yawning is False
    state = tracker.update("A", True, _face(jaw=0.8))
    assert state is not None and state.yawning is True
    state = None
    for _ in range(5):
        state = tracker.update("B", True, _face(yaw=42.0))
    assert state is not None and state.distracted is True
    assert tracker.update("C", True, _face(yaw=5.0)).distracted is False


def test_face_mesh_loads_when_present():
    import pathlib

    import pytest

    from odc.dms import FaceMesh

    path = pathlib.Path("models/mediapipe-face/face_landmarker.task")
    if not path.exists():
        pytest.skip("face weights absent (download documented in ODC_MODEL_SELECTION)")
    mesh = FaceMesh(str(path))
    mesh.load()
    mesh.close()
