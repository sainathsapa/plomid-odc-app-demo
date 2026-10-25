"""Calibration wizard tests — validation rules + save API."""

from __future__ import annotations

from odc.layouts import save_layout, validate_layout_payload


def _seat(sid: str = "seat-1", poly=None) -> dict:
    return {
        "seat_id": sid,
        "polygon": poly if poly is not None else [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
    }


def test_valid_payload_passes():
    assert validate_layout_payload("custom-van", [_seat("a"), _seat("b")]) == []


def test_bad_names_and_ids_rejected():
    assert validate_layout_payload("../evil", [_seat()]) != []
    assert validate_layout_payload("ok", [_seat("../x")]) != []
    assert validate_layout_payload("ok", [_seat("a"), _seat("a")]) != []
    assert validate_layout_payload("ok", []) != []
    assert validate_layout_payload("ok", [_seat("a")] * 13) != []


def test_degenerate_and_wild_polygons_rejected():
    assert validate_layout_payload("ok", [_seat("a", [[0.1, 0.1], [0.2, 0.2]])]) != []
    tiny = [[0.5, 0.5], [0.51, 0.5], [0.51, 0.51]]
    assert validate_layout_payload("ok", [_seat("a", tiny)]) != []
    wild = [[0.0, 0.0], [2.0, 0.0], [0.5, 1.0]]
    assert validate_layout_payload("ok", [_seat("a", wild)]) != []


def test_save_round_trips_through_loader(tmp_path):
    from odc.seats import load_layout

    target = save_layout("test-cal-van", [_seat("driver"), _seat("child-seat")], directory=tmp_path)
    assert target.exists()
    layout = load_layout(str(target))
    assert layout.version == "test-cal-van"
    assert [s.seat_id for s in layout.seats] == ["driver", "child-seat"]
    # A child-like label is a stable identifier only, never a restraint or
    # occupant classification rule.
    assert layout.seats[1].seat_id == "child-seat"


def test_save_refuses_overwrite_without_flag(tmp_path):
    import pytest

    save_layout("dup", [_seat()], directory=tmp_path)
    with pytest.raises(FileExistsError):
        save_layout("dup", [_seat()], directory=tmp_path)
    save_layout("dup", [_seat()], directory=tmp_path, overwrite=True)


def test_standard_layouts_have_one_based_multi_row_identity():
    from odc.seats import (
        apply_orientation,
        load_layout,
        order_seats_front_to_rear,
        seat_display_name,
    )

    five = load_layout("configs/seats/standard-5-seat-rhd.yaml")
    seven = load_layout("configs/seats/standard-7-seat-rhd.yaml")
    assert [len(row) for row in order_seats_front_to_rear(five)] == [2, 3]
    assert [len(row) for row in order_seats_front_to_rear(seven)] == [2, 3, 2]
    assert [s.seat_id for s in five.seats] == ["seat_1", "seat_2", "seat_3", "seat_4", "seat_5"]
    assert [seat_display_name(s) for s in five.seats] == [
        "Seat 1",
        "Seat 2",
        "Seat 3",
        "Seat 4",
        "Seat 5",
    ]
    assert all((s.row or 0) >= 1 and (s.display_number or 0) >= 1 for s in seven.seats)
    assert five.orientation == "RHD"
    assert five.seats[0].physical_role == "driver"
    lhd = apply_orientation(five, "LHD")
    assert lhd.orientation == "LHD"
    assert lhd.seats[0].physical_role == "front_passenger"
    assert lhd.seats[1].physical_role == "driver"
    assert [s.seat_id for s in lhd.seats] == [s.seat_id for s in five.seats]


def test_save_api_end_to_end():
    from pathlib import Path

    from fastapi.testclient import TestClient

    import tests.unit.test_console as tc
    from odc.config import AppConfig
    from odc.console.app import create_app
    from odc.seats import load_layout

    client = TestClient(
        create_app(AppConfig(), tc._StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"))
    )
    payload = {"name": "test-cal-api", "seats": [_seat("driver"), _seat("kid-2")]}
    r = client.post("/api/layouts/save", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["layout"] == "test-cal-api"
    try:
        assert client.post("/api/layouts/save", json=payload).status_code == 409
        bad = dict(payload, name="../evil")
        assert client.post("/api/layouts/save", json=bad).status_code == 400
        thin = dict(payload, name="test-cal-thin", seats=[_seat("a", [[0, 0], [1, 1]])])
        assert client.post("/api/layouts/save", json=thin).status_code == 400
        # /calibrate merged into the seat map: old URL redirects there.
        r = client.get("/calibrate", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"].endswith("/seats")
        assert "Calibrate seat zones" in client.get("/seats").text
    finally:
        Path("configs/seats/test-cal-api.yaml").unlink(missing_ok=True)
