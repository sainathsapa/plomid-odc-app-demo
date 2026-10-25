"""Seat geometry tests — overlap detection, ordering, validation. No weights."""

from __future__ import annotations

from odc.domain import SeatGeometry
from odc.seats import (
    SeatLayout,
    find_overlapping_seats,
    order_seats_front_to_rear,
    polygon_intersection_area,
    polygon_iou,
    validate_seat_layout,
)


def _seat(sid: str, poly: list, **kw) -> SeatGeometry:
    return SeatGeometry(
        seat_id=sid, polygon=[(float(x), float(y)) for x, y in poly],
        camera_profile="t", version="t", **kw,
    )


def _layout(seats: list[SeatGeometry]) -> SeatLayout:
    return SeatLayout(camera_profile="t", version="t", seats=seats)


def test_iou_identical_is_one_and_disjoint_is_zero():
    a = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    assert polygon_iou(a, a) == 1.0
    b = [(0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.6, 1.0)]
    assert polygon_iou(a, b) == 0.0
    assert polygon_intersection_area(a, b) == 0.0


def test_iou_partial_overlap_between_zero_and_one():
    a = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    b = [(0.25, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 1.0)]
    iou = polygon_iou(a, b)
    assert 0.2 < iou < 0.6


def test_f1_demo_layout_has_no_cross_seat_overlap():
    from odc.seats import load_layout

    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    pairs = find_overlapping_seats(layout)
    assert pairs == []


def test_n1_layout_covers_observed_driver_detection_center():
    from odc.domain import BBox
    from odc.seats import associate_track_first, load_layout

    layout = load_layout("configs/seats/demo-n1-v0.yaml")
    observed = BBox(x_min=0.3547, y_min=0.4411, x_max=0.6315, y_max=0.9973)

    association = associate_track_first([("night-driver", observed)], layout)

    assert association[0].seat_id == "driver"
    assert not association[0].ambiguous


def test_find_overlaps_clean_for_disjoint():
    layout = _layout([
        _seat("a", [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)]),
        _seat("b", [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)]),
    ])
    assert find_overlapping_seats(layout) == []


def test_order_front_to_rear_rows_and_columns():
    layout = _layout([
        _seat("rear-r", [(0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.6, 1.0)]),
        _seat("front-l", [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)]),
        _seat("rear-l", [(0.0, 0.6), (0.4, 0.6), (0.4, 1.0), (0.0, 1.0)]),
        _seat("front-r", [(0.6, 0.0), (1.0, 0.0), (1.0, 0.4), (0.6, 0.4)]),
    ])
    rows = order_seats_front_to_rear(layout)
    assert rows == [["front-l", "front-r"], ["rear-l", "rear-r"]]


def test_order_stored_row_hints_win():
    layout = _layout([
        _seat("b", [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)], row=1, col=0),
        _seat("a", [(0.0, 0.6), (0.4, 0.6), (0.4, 1.0), (0.0, 1.0)], row=0, col=0),
    ])
    assert order_seats_front_to_rear(layout) == [["a"], ["b"]]


def test_validate_warns_on_overlap_duplicate_and_empty():
    layout = _layout([
        _seat("a", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]),
        _seat("a", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]),
    ])
    warnings = validate_seat_layout(layout)
    assert any("duplicate" in w for w in warnings)
    assert any("overlap" in w for w in warnings)
    assert validate_seat_layout(_layout([])) == ["no seats: map is uncalibrated"]


def test_layout_geometry_warnings_pure_and_compatible():
    from odc.layouts import validate_layout_geometry, validate_layout_payload

    # Existing contract untouched: identical halves are not a hard error.
    seats = [
        {"seat_id": "a", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
        {"seat_id": "b", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
    ]
    assert validate_layout_payload("ok", seats) == []
    # ...but geometry warnings flag the overlap for the editors.
    warnings = validate_layout_geometry(seats)
    assert any("overlap" in w for w in warnings)
    clean = [
        {"seat_id": "a", "polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]},
        {"seat_id": "b", "polygon": [[0.6, 0], [1, 0], [1, 1], [0.6, 1]]},
    ]
    assert validate_layout_geometry(clean) == []


def test_standard_association_uses_geometry_not_detection_order():
    from odc.domain import BBox
    from odc.seats import associate_track_first, load_layout

    layout = load_layout("configs/seats/standard-5-seat-rhd.yaml")
    result = associate_track_first(
        [("track-rear-right", BBox(x_min=0.68, y_min=0.60, x_max=0.90, y_max=0.88)),
         ("track-front-left", BBox(x_min=0.16, y_min=0.10, x_max=0.44, y_max=0.40))],
        layout,
    )
    assert {item.track_id: item.seat_id for item in result} == {
        "track-rear-right": "seat_5",
        "track-front-left": "seat_1",
    }
