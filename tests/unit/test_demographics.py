"""Gender-appearance probe tests (EXPERIMENTAL, D30)."""

from __future__ import annotations

from odc.demographics import CONFIDENCE_FLOOR, MIN_BOX_AREA, classify_appearance
from odc.domain import GenderAppearance


def test_child_gate_absolute():
    label, conf = classify_appearance(0.01, {"man": 0.99, "woman": 0.01})
    assert (label, conf) == (GenderAppearance.UNKNOWN, 0.0)
    assert MIN_BOX_AREA == 0.04


def test_confidence_floor_abstains():
    label, conf = classify_appearance(0.5, {"man": 0.64, "woman": 0.36})
    assert (label, conf) == (GenderAppearance.UNKNOWN, 0.0)
    assert CONFIDENCE_FLOOR == 0.7


def test_confident_calls():
    assert classify_appearance(0.5, {"man": 0.07, "woman": 0.93})[0] == (
        GenderAppearance.APPARENT_WOMAN
    )
    assert classify_appearance(0.5, {"man": 0.98, "woman": 0.02})[0] == (
        GenderAppearance.APPARENT_MAN
    )


def test_disabled_is_default():
    from odc.config import AppConfig

    assert AppConfig().detection.enable_gender_appearance is False
    assert GenderAppearance.DISABLED.value == "DISABLED"


class _FakeProbe:
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def score(self, crop):
        self.calls += 1
        return self._scores


def test_gender_for_seats_skips_children_and_failures():
    from odc.demographics import gender_for_seats

    rows = [
        {"seat_id": "driver", "presence": "OCCUPIED", "child": "No"},
        {"seat_id": "baby", "presence": "OCCUPIED", "child": "Yes"},
        {"seat_id": "empty", "presence": "EMPTY_CANDIDATE", "child": "No"},
    ]
    crops = {"driver": (object(), 0.20), "baby": (object(), 0.20)}
    out = gender_for_seats(rows, crops, _FakeProbe({"man": 0.91, "woman": 0.09}))
    assert out == {"driver": ("APPARENT_MAN", 0.91)}
    # Child seat never probed even with a crop ready.
    assert "baby" not in out


def test_gender_for_seats_abstains_below_floor():
    from odc.demographics import gender_for_seats

    rows = [{"seat_id": "a", "presence": "OCCUPIED", "child": "Unknown"}]
    out = gender_for_seats(rows, {"a": (object(), 0.20)}, _FakeProbe({"man": 0.6, "woman": 0.4}))
    assert out == {"a": ("UNKNOWN", None)}


def test_gender_for_seats_survives_probe_errors():
    from odc.demographics import gender_for_seats

    class _Boom:
        def score(self, crop):
            raise RuntimeError("weights missing")

    rows = [{"seat_id": "a", "presence": "OCCUPIED", "child": "No"}]
    assert gender_for_seats(rows, {"a": (object(), 0.2)}, _Boom()) == {}
