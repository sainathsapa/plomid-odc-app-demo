"""Validation integration — real model on the real dev split + DB record."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from test_detector_ort import WEIGHTS, _runtime

from odc.validate import run_dataset

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no detector weights")


def _dev():
    dataset = yaml.safe_load(Path("eval/datasets/v1.yaml").read_text())
    annotations = json.loads(Path("eval/annotations/v1.json").read_text())
    return dataset, annotations


def _detect_fn(runtime):
    from odc import contracts
    from odc.domain import FrameMeta, RunMode

    def detect(bgr):
        frame = contracts.Frame(
            meta=FrameMeta(
                source_id="eval",
                seq=0,
                captured_at=datetime.now(UTC),
                width=bgr.shape[1],
                height=bgr.shape[0],
                mode=RunMode.TEST,
            ),
            pixels=bgr,
        )
        result = runtime.infer(frame)
        return [
            (d.class_label, d.score, [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max])
            for d in result.detections
        ]

    return detect


def _stable(report):
    return {k: v for k, v in report["metrics"].items() if "latency" not in k}


@needs_weights
def test_dev_split_report_structure():

    dataset, annotations = _dev()
    runtime = _runtime(threshold=0.3)
    try:
        report = run_dataset(dataset, annotations, "dev", _detect_fn(runtime), threshold=0.3)
        assert report["split"] == "dev"
        assert report["metrics"]["frames"] == 16
        assert report["metrics"]["seat_wrong"] == 0
        assert len(report["gates"]) == 4
        assert all(g["passed"] for g in report["gates"])
    finally:
        runtime.close()


@needs_weights
def test_dev_metrics_reproduce():
    dataset, annotations = _dev()
    runtime = _runtime(threshold=0.3)
    try:
        first = run_dataset(dataset, annotations, "dev", _detect_fn(runtime), threshold=0.3)
        second = run_dataset(dataset, annotations, "dev", _detect_fn(runtime), threshold=0.3)
        stable = _stable(first) == _stable(second)
        assert stable
        assert first["metrics"]["seat_correct_rate"] >= 0.75
    finally:
        runtime.close()


@needs_weights
def test_record_to_db_roundtrip(plomid_dsn):
    from odc.persistence import PlomidRepository

    dataset, annotations = _dev()
    runtime = _runtime(threshold=0.3)
    repo = PlomidRepository(plomid_dsn)
    try:
        report = run_dataset(dataset, annotations, "dev", _detect_fn(runtime), threshold=0.3)
        now = datetime.now(UTC)
        repo.ensure_schema_version(2)
        repo.write_validation_run(
            {
                "run_id": "val-test",
                "build_version": "0.1.0",
                "dataset_id": "v1",
                "hardware_profile": "test",
                "model_manifest": "{}",
                "started_at": now,
                "completed_at": now,
                "status": "DONE",
            }
        )
        repo.write_validation_result(
            [
                {
                    "run_id": "val-test",
                    "task": "detection",
                    "label": "person",
                    "scenario": "dev",
                    "precision": report["metrics"]["person_precision"],
                    "recall": report["metrics"]["person_recall"],
                    "f1": report["metrics"]["person_f1"],
                    "fp_count": 0,
                    "fn_count": 0,
                    "latency_p50_ms": 0.0,
                    "latency_p95_ms": 0.0,
                    "notes": "t",
                }
            ]
        )
        runs = repo.read_validation_runs()
        assert [r["run_id"] for r in runs] == ["val-test"]
    finally:
        runtime.close()
        repo.close()
