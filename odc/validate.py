"""Validation metrics core (Phase 9) — deterministic, ground-truth-based.

Detection: greedy IoU matching per class → P/R/F1.
Seats (3-way, the honest metric): OCCUPIED/LIKELY=positive, EMPTY=negative,
UNKNOWN/STALE=abstain. Abstains are reported, never hidden in accuracy.
"""

from __future__ import annotations

from typing import Any

from .domain import BBox


def box_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / (area_a + area_b - inter)


def match_boxes(
    gt: list[list[float]], pred: list[list[float]], iou_thresh: float = 0.5
) -> tuple[int, int, int, list[int], list[int]]:
    """Greedy matching by descending pred score order (pred sorted already).

    Returns (tp, fp, fn, matched_pred_indices, matched_gt_indices)."""
    used = set()
    tp = 0
    matched_pred: list[int] = []
    matched_gt: list[int] = []
    for pi, pbox in enumerate(pred):
        best, best_iou = -1, iou_thresh
        for gi, gbox in enumerate(gt):
            if gi in used:
                continue
            score = box_iou(gbox, pbox)
            if score > best_iou:
                best, best_iou = gi, score
        if best >= 0:
            used.add(best)
            matched_pred.append(pi)
            matched_gt.append(best)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp, matched_pred, matched_gt


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


POSITIVE = {"OCCUPIED", "LIKELY_OCCUPIED"}
NEGATIVE = {"EMPTY_CANDIDATE"}


def score_seat(truth: str, predicted: str) -> str:
    """3-way: 'correct' | 'wrong' | 'abstain' (UNKNOWN/STALE never wrong)."""
    if predicted in POSITIVE:
        return "correct" if truth == "occupied" else "wrong"
    if predicted in NEGATIVE:
        return "correct" if truth == "empty" else "wrong"
    return "abstain"


def run_dataset(
    dataset: dict[str, Any],
    annotations: dict[str, Any],
    split: str,
    detect: Any,
    threshold: float = 0.5,
    iou_thresh: float = 0.5,
    dense_frames: int = 60,
) -> dict[str, Any]:
    """Run one split. `detect(bgr) -> list[(label, score, xyxy)]` injected.

    Returns the machine-readable report (metrics + gates + gallery).
    Deterministic for identical inputs (no randomness, no wall clock in
    metrics; run_at is metadata only).
    """
    import time

    import cv2

    clips: dict[str, str] = dataset["clips"]
    acceptance: dict[str, float] = dataset.get("acceptance", {})
    wanted = set(annotations["splits"][split])
    frames = [f for f in annotations["frames"] if f["clip"] in wanted]

    det_tp = det_fp = det_fn = 0
    seat_correct = seat_wrong = seat_abstain = 0
    gallery: list[dict[str, Any]] = []
    latencies: list[float] = []
    for entry in frames:
        cap = cv2.VideoCapture(clips[entry["clip"]])
        cap.set(cv2.CAP_PROP_POS_FRAMES, entry["idx"])
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            continue
        start = time.monotonic()
        dets = detect(bgr)
        latencies.append((time.monotonic() - start) * 1000)
        persons = sorted([d for d in dets if d[0] == "person"], key=lambda d: -d[1])
        pred_boxes = [d[2] for d in persons if d[1] >= threshold]
        tp, fp, fn, matched_pred, matched_gt = match_boxes(entry["persons"], pred_boxes, iou_thresh)
        det_tp, det_fp, det_fn = det_tp + tp, det_fp + fp, det_fn + fn
        matched_set = set(matched_pred)
        for pi in range(len(pred_boxes)):
            if pi not in matched_set:
                gallery.append(
                    {
                        "clip": entry["clip"],
                        "idx": entry["idx"],
                        "kind": "fp",
                        "box": pred_boxes[pi],
                    }
                )
        hit_gt = set(matched_gt)
        for gi in range(len(entry["persons"])):
            if gi not in hit_gt:
                gallery.append(
                    {
                        "clip": entry["clip"],
                        "idx": entry["idx"],
                        "kind": "fn",
                        "box": entry["persons"][gi],
                    }
                )
        layout_name = annotations.get("layouts", {}).get(entry["clip"])
        dark = float(bgr.mean()) < 30.0
        for seat_id, truth in entry.get("seats", {}).items():
            voted = _vote_seat(entry, seat_id, persons, threshold, layout_name, dark)
            outcome = score_seat(truth, voted)
            if outcome == "correct":
                seat_correct += 1
            elif outcome == "wrong":
                seat_wrong += 1
                gallery.append(
                    {
                        "clip": entry["clip"],
                        "idx": entry["idx"],
                        "kind": f"seat-wrong-{seat_id}",
                        "box": [],
                    }
                )
            else:
                seat_abstain += 1

    det = prf(det_tp, det_fp, det_fn)
    seat_total = seat_correct + seat_wrong + seat_abstain
    seat_rate = seat_correct / seat_total if seat_total else 0.0
    unknown_rate = seat_abstain / seat_total if seat_total else 0.0
    latencies.sort()
    metrics = {
        "person_precision": round(det["precision"], 4),
        "person_recall": round(det["recall"], 4),
        "person_f1": round(det["f1"], 4),
        "person_tp": det_tp,
        "person_fp": det_fp,
        "person_fn": det_fn,
        "seat_correct_rate": round(seat_rate, 4),
        "seat_unknown_rate": round(unknown_rate, 4),
        "seat_correct": seat_correct,
        "seat_wrong": seat_wrong,
        "seat_abstain": seat_abstain,
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "frames": len(frames),
    }
    gates = [
        {
            "gate": "person_precision_dev",
            "passed": det["precision"] >= acceptance.get("person_precision_dev", 0),
            "value": round(det["precision"], 4),
        },
        {
            "gate": "person_recall_dev",
            "passed": det["recall"] >= acceptance.get("person_recall_dev", 0),
            "value": round(det["recall"], 4),
        },
        {
            "gate": "seat_correct_dev",
            "passed": seat_rate >= acceptance.get("seat_correct_dev", 0),
            "value": round(seat_rate, 4),
        },
        {
            "gate": "seat_unknown_rate_max",
            "passed": unknown_rate <= acceptance.get("seat_unknown_rate_max", 1),
            "value": round(unknown_rate, 4),
        },
    ]
    return {
        "kind": "VALIDATION — ground-truth-based, split " + split,
        "split": split,
        "metrics": metrics,
        "gates": gates,
        "gallery": gallery[:40],
        "holdout_policy": dataset.get("holdout_policy", ""),
    }


def _vote_seat(
    entry: dict[str, Any],
    seat_id: str,
    persons: list[Any],
    threshold: float,
    layout_name: str | None,
    dark: bool = False,
) -> str:
    """Single-frame seat vote: positive / EMPTY_CANDIDATE / UNKNOWN.

    Mirrors the engine rule (D12/D17): poor observations can only abstain,
    never vote empty."""
    if not layout_name or dark:
        return "UNKNOWN"
    from .seats import associate_track_first, load_layout

    try:
        layout = load_layout(f"configs/seats/{layout_name}.yaml")
    except Exception:
        return "UNKNOWN"
    pairs = [(f"p{i}", _xyxy_to_bbox(d[2])) for i, d in enumerate(persons) if d[1] >= threshold]
    assoc = associate_track_first(pairs, layout)
    mine = [a for a in assoc if a.seat_id == seat_id and not a.ambiguous]
    if mine:
        return "OCCUPIED"
    if any(a.ambiguous for a in assoc):
        return "UNKNOWN"
    return "EMPTY_CANDIDATE"


def _xyxy_to_bbox(box: list[float]) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in box)
    return BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def compare_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Regression diff between two validation reports."""
    deltas: dict[str, Any] = {"gates": [], "metric_deltas": {}}
    for key in ("person_f1", "seat_correct_rate", "seat_unknown_rate"):
        a = float(old.get("metrics", {}).get(key, 0.0))
        b = float(new.get("metrics", {}).get(key, 0.0))
        deltas["metric_deltas"][key] = round(b - a, 4)
    for gate in new.get("gates", []):
        deltas["gates"].append(
            {
                "gate": gate["gate"],
                "passed": gate["passed"],
                "was": next(
                    (g["passed"] for g in old.get("gates", []) if g["gate"] == gate["gate"]), None
                ),
            }
        )
    deltas["regressed"] = any(g["passed"] is False and g["was"] is True for g in deltas["gates"])
    return deltas


def write_html_report(
    report: dict[str, Any],
    clips: dict[str, str],
    out_dir: str,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Human-readable report + failure gallery thumbnails. Returns index path."""
    from pathlib import Path as _Path

    import cv2

    out = _Path(out_dir)
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    items = []
    for n, item in enumerate(report.get("gallery", [])):
        cap = cv2.VideoCapture(clips[item["clip"]])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["idx"])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[0], frame.shape[1]
        color = (0, 0, 255) if item["kind"] == "fp" else (0, 255, 0)
        box = item.get("box") or []
        if len(box) == 4:
            x0, y0, x1, y1 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
        small = cv2.resize(frame, (480, 270))
        name = f"{item['clip']}_{item['idx']}_{item['kind']}_{n}.jpg"
        cv2.imwrite(str(thumbs / name), small)
        items.append((item, name))
    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report["metrics"].items()
    )
    gate_rows = "".join(
        f"<tr><td>{g['gate']}</td><td>{'PASS' if g['passed'] else 'FAIL'}</td>"
        f"<td>{g['value']}</td></tr>"
        for g in report["gates"]
    )
    gal = "".join(
        f"<figure><img src='thumbs/{name}'><figcaption>{item['clip']} "
        f"f{item['idx']} {item['kind']}</figcaption></figure>"
        for item, name in items
    )
    compare_html = ""
    if comparison is not None:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:+.4f}</td></tr>"
            for k, v in comparison["metric_deltas"].items()
        )
        compare_html = (
            "<h2>Regression vs previous</h2>"
            f"<p>regressed: {comparison['regressed']}</p>"
            f"<table>{rows}</table>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>ODC validation</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
        "figure{display:inline-block;margin:8px}img{width:240px}</style></head><body>"
        f"<h1>ODC validation — {report['split']}</h1>"
        f"<p>{report['kind']}</p>"
        f"<p>{report.get('holdout_policy', '')}</p>"
        "<h2>Metrics</h2><table>" + metrics_rows + "</table>"
        "<h2>Gates</h2><table>"
        + gate_rows
        + "</table>"
        + compare_html
        + "<h2>Failure gallery</h2>"
        + (gal or "<p>no failures recorded</p>")
        + "</body></html>"
    )
    index = out / "index.html"
    index.write_text(html)
    return str(index)
