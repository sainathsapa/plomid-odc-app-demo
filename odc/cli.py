"""`odc` CLI (Phase 1): version, config-check, health. Exits cleanly."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from typing import Any

from . import __version__
from .config import load_config
from .domain import HealthStatus
from .health import build_health
from .logging import get_logger, setup_logging

log = get_logger("cli")


def _probe_db(dsn: str, timeout_s: int) -> tuple[HealthStatus, str]:
    import psycopg  # local import: psycopg only needed when probing

    try:
        conn = psycopg.connect(dsn, connect_timeout=timeout_s, autocommit=True)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return HealthStatus.HEALTHY, "SELECT 1 ok"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"{type(e).__name__}: {e}"


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_config_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(json.dumps(cfg.model_dump(mode="json", exclude={"plomid": {"dsn"}}), indent=2))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.probe_db and cfg.plomid.dsn:
        status, detail = _probe_db(
            cfg.plomid.dsn.get_secret_value(), int(cfg.plomid.connect_timeout_s)
        )
    else:
        status, detail = HealthStatus.UNKNOWN, "not probed (pass --probe-db with ODC_DSN)"
    health = build_health(cfg, status, detail)
    print(health.model_dump_json(indent=2))
    return 0 if health.overall != HealthStatus.UNHEALTHY else 2


def cmd_replay(args: argparse.Namespace) -> int:
    from .capture import (
        FileReplaySource,
        LiveCameraSource,
        QualityConfig,
        assess_quality,
    )

    cfg = load_config(args.config)
    source: FileReplaySource | LiveCameraSource
    if cfg.camera.source == "live":
        source = LiveCameraSource(cfg.camera)
    else:
        source = FileReplaySource(cfg.camera)
    quality_cfg = QualityConfig(
        dark_thresh=cfg.camera.quality.dark_thresh,
        bright_thresh=cfg.camera.quality.bright_thresh,
        blur_var_thresh=cfg.camera.quality.blur_var_thresh,
        frozen_diff_thresh=cfg.camera.quality.frozen_diff_thresh,
    )
    try:
        first = source.open()
    except Exception as e:
        log.error("replay open failed: %s: %s", type(e).__name__, e)
        return 1
    prev: object = None
    n = 0
    dark = bright = blurry = frozen = 0
    try:
        while args.max_frames is None or n < args.max_frames:
            frame = source.next()
            if frame is None:
                break
            pixels = frame.pixels
            if pixels is None:
                continue
            quality, prev = assess_quality(pixels, prev, quality_cfg)  # type: ignore[arg-type]
            n += 1
            dark += quality.dark
            bright += quality.bright
            blurry += quality.blurry
            frozen += quality.frozen
    finally:
        health = source.health()
        source.close()
    print(
        json.dumps(
            {
                "source": source.kind,
                "path": cfg.camera.path,
                "frames": n,
                "first_seq": first.seq,
                "health": health.status.value,
                "failure": health.failure_code,
                "advisory_quality": {
                    "dark": dark,
                    "bright": bright,
                    "blurry": blurry,
                    "frozen": frozen,
                },
            },
            indent=2,
        )
    )
    return 0 if health.status.value != "UNHEALTHY" else 1


def cmd_detect(args: argparse.Namespace) -> int:
    import json as _json

    from .capture import FileReplaySource
    from .domain import ModelManifest
    from .eval import run_detector_eval
    from .runtime import OnnxDetectorRuntime

    cfg = load_config(args.config)
    manifest_path = (
        __import__("pathlib").Path("models/manifests") / f"{cfg.detection.model_id}.json"
    )
    if not manifest_path.exists():
        log.error("unknown model %s", cfg.detection.model_id)
        return 1
    info = _json.loads(manifest_path.read_text())
    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
        source_url=f"https://huggingface.co/{info['hf_repo']}",
        code_revision=info.get("hf_revision", "unset"),
        weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
        dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
        approved_for_commercial_use=bool(info.get("approved_for_commercial_use", False)),
        input_profile=info.get("input_profile", ""),
        runtime_profile=info.get("runtime_profile", ""),
    )
    runtime = OnnxDetectorRuntime(threshold=args.threshold or cfg.detection.threshold)
    try:
        runtime.load(manifest)
    except Exception as e:
        log.error("detector load failed: %s: %s", type(e).__name__, e)
        return 1
    source = FileReplaySource(cfg.camera)
    report = run_detector_eval(source, runtime, manifest, cfg, max_frames=args.max_frames)
    runtime.close()
    print(_json.dumps(report, indent=2))
    if args.out:
        __import__("pathlib").Path(args.out).write_text(_json.dumps(report, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .banner import print_banner
    from .console.app import create_app
    from .persistence import PlomidRepository
    from .config import DEFAULT_LOCAL_PLOMID_DSN
    from pydantic import SecretStr
    from .seats import load_layout

    cfg = load_config(args.config)
    if cfg.plomid.dsn is None:
        cfg.plomid.dsn = SecretStr(DEFAULT_LOCAL_PLOMID_DSN)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        log.warning(
            "binding %s without TLS: credentials cross the network in the clear; "
            "loopback-only unless the pilot network is trusted (D9)",
            args.host,
        )
    repo = PlomidRepository(cfg.plomid.dsn.get_secret_value(), int(cfg.plomid.connect_timeout_s))
    try:
        repo.ensure_schema_version(2)
    except Exception as exc:
        raise SystemExit(f"PLOMID startup provisioning failed: {type(exc).__name__}: {exc}") from exc
    try:
        layout = load_layout(f"configs/seats/demo-{cfg.camera.camera_id}-v0.yaml")
    except Exception:
        layout = load_layout("configs/seats/standard-5-seat-rhd.yaml")
    app = create_app(cfg, repo, layout, autostart=args.auto_start)
    print_banner()
    sys.stderr.write(f"console on http://{args.host}:{args.port}/live\n")
    sys.stderr.flush()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    import datetime as _dt
    import json as _json
    from pathlib import Path as _Path

    import yaml as _yaml

    from .runtime import OnnxDetectorRuntime
    from .validate import compare_reports, run_dataset, write_html_report

    cfg = load_config(args.config)
    dataset = _yaml.safe_load(_Path(args.dataset).read_text())
    annotations = _json.loads(_Path(dataset["annotations"]).read_text())
    manifest_path = _Path("models/manifests") / f"{cfg.detection.model_id}.json"
    if not manifest_path.exists():
        log.error("unknown model %s", cfg.detection.model_id)
        return 1
    info = _json.loads(manifest_path.read_text())
    from .domain import ModelManifest

    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
        source_url=f"https://huggingface.co/{info['hf_repo']}",
        code_revision=info.get("hf_revision", "unset"),
        weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
        dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
        approved_for_commercial_use=bool(info.get("approved_for_commercial_use", False)),
        input_profile=info.get("input_profile", ""),
        runtime_profile=info.get("runtime_profile", ""),
    )
    runtime = OnnxDetectorRuntime(
        threshold=args.threshold or dataset.get("eval_threshold") or cfg.detection.threshold
    )
    try:
        runtime.load(manifest)
    except Exception as e:
        log.error("detector load failed: %s: %s", type(e).__name__, e)
        return 1

    def detect(bgr: Any) -> list[tuple[str, float, list[float]]]:
        from . import contracts
        from .domain import FrameMeta, RunMode

        frame = contracts.Frame(
            meta=FrameMeta(
                source_id="eval",
                seq=0,
                captured_at=_dt.datetime.now(_dt.UTC),
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

    try:
        report = run_dataset(dataset, annotations, args.split, detect, threshold=runtime._threshold)
    finally:
        runtime.close()
    report["model"] = manifest.model_dump(mode="json")
    report["dataset"] = args.dataset
    out_dir = _Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(_json.dumps(report, indent=2))
    comparison = None
    if args.compare:
        old = _json.loads(_Path(args.compare).read_text())
        comparison = compare_reports(old, report)
        print(_json.dumps(comparison, indent=2))
    write_html_report(report, dataset["clips"], str(out_dir), comparison)
    print(_json.dumps({"metrics": report["metrics"], "gates": report["gates"]}, indent=2))
    if args.record_to_db:
        if not cfg.plomid.dsn:
            log.error("record-to-db needs ODC_DSN")
            return 1
        from .persistence import PlomidRepository

        repo = PlomidRepository(cfg.plomid.dsn.get_secret_value())
        try:
            import uuid as _uuid

            run_id = f"val-{_uuid.uuid4().hex[:8]}"
            now = _dt.datetime.now(_dt.UTC)
            repo.ensure_schema_version(2)
            repo.write_validation_run(
                {
                    "run_id": run_id,
                    "build_version": cfg.build_version,
                    "dataset_id": args.dataset,
                    "hardware_profile": "laptop-dev",
                    "model_manifest": manifest.model_dump_json(),
                    "started_at": now,
                    "completed_at": now,
                    "status": "DONE",
                }
            )
            metrics = report["metrics"]
            repo.write_validation_result(
                [
                    {
                        "run_id": run_id,
                        "task": "detection",
                        "label": "person",
                        "scenario": args.split,
                        "precision": metrics["person_precision"],
                        "recall": metrics["person_recall"],
                        "f1": metrics["person_f1"],
                        "fp_count": metrics["person_fp"],
                        "fn_count": metrics["person_fn"],
                        "latency_p50_ms": metrics["latency_p50_ms"],
                        "latency_p95_ms": metrics["latency_p95_ms"],
                        "notes": "odc validate",
                    }
                ]
            )
            repo.write_validation_result(
                [
                    {
                        "run_id": run_id,
                        "task": "seat-state",
                        "label": "all",
                        "scenario": args.split,
                        "precision": metrics["seat_correct_rate"],
                        "recall": metrics["seat_correct_rate"],
                        "f1": metrics["seat_correct_rate"],
                        "fp_count": metrics["seat_wrong"],
                        "fn_count": metrics["seat_abstain"],
                        "latency_p50_ms": None,
                        "latency_p95_ms": None,
                        "notes": "3-way scoring",
                    }
                ]
            )
            print(f"recorded {run_id}")
        finally:
            repo.close()
    failed = [g for g in report["gates"] if not g["passed"]]
    return 1 if failed else 0


def cmd_profile(args: argparse.Namespace) -> int:
    import json as _json

    from .profile import profile_clip

    def build_runtime() -> object:
        from .domain import ModelManifest
        from .runtime import OnnxDetectorRuntime

        info = _json.loads(
            __import__("pathlib").Path("models/manifests/rtdetrv2-s-coco.json").read_text()
        )
        manifest = ModelManifest(
            model_id=info["model_id"],
            family=info["family"],
            checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
            source_url=f"https://huggingface.co/{info['hf_repo']}",
            code_revision=info.get("hf_revision", "unset"),
            weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
            dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
            approved_for_commercial_use=bool(info.get("approved_for_commercial_use", False)),
            input_profile=info.get("input_profile", ""),
            runtime_profile=info.get("runtime_profile", ""),
        )
        runtime = OnnxDetectorRuntime(threshold=0.3)
        if args.threads:
            import onnxruntime as _ort  # type: ignore[import-untyped]

            opts = _ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            opts.inter_op_num_threads = 1
            runtime.load(manifest, sess_options=opts)
        else:
            runtime.load(manifest)
        return runtime

    report = profile_clip(args.clip, args.max_frames, build_runtime)
    print(_json.dumps(report, indent=2))
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _Path

    try:
        import onnxruntime as _ort

        available = _ort.get_available_providers()
    except Exception:
        available = []
    print(_json.dumps({"available_providers": available}, indent=2))
    for manifest_path in sorted(_Path("models/manifests").glob("*.json")):
        try:
            info = _json.loads(manifest_path.read_text())
        except Exception:
            continue
        print(f"== {info.get('model_id', manifest_path.stem)}")
        print(f"   preference: {info.get('providers_preference', ['CPUExecutionProvider'])}")
        for provider, reason in (info.get("providers_blocked", {}) or {}).items():
            print(f"   blocked {provider}: {reason[:100]}")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    """Delete generated artifacts older than N days (reports, thumbnails).

    Diagnostic media has no other home: footage/ is gitignored staging,
    the hub keeps one transient JPEG, PLOMID stores metadata only.
    """
    import time as _time
    from pathlib import Path as _Path

    root = _Path(args.dir)
    if not root.exists():
        print(f"nothing to purge: {args.dir}")
        return 0
    cutoff = _time.time() - args.older_than_days * 86400
    removed = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    print(f"purged {removed} files from {args.dir}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Readiness bundle: config, footage, camera, model, PLOMID, licenses.

    Prints JSON (no secrets, no frames), exit 0 only when every check that
    CAN pass in this environment passes; each check reports pass/fail/skip
    with reasons — a fail is never hidden as a skip.
    """
    import json as _json
    from pathlib import Path as _Path

    checks: list[dict[str, str]] = []

    def check(name: str, status: str, detail: str = "") -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    try:
        cfg = load_config(args.config)
        check("config", "pass", f"mode={cfg.mode.value}")
    except Exception as e:
        check("config", "fail", f"{type(e).__name__}: {e}")
        print(_json.dumps({"checks": checks}, indent=2))
        return 1

    footage = _Path(cfg.camera.path)
    check("footage", "pass" if footage.exists() else "fail", str(footage))

    try:
        from .capture import FileReplaySource

        source = FileReplaySource(cfg.camera)
        source.open()
        source.close()
        check("camera_open", "pass", cfg.camera.source)
    except Exception as e:
        check("camera_open", "fail", f"{type(e).__name__}: {e}")

    manifest_path = _Path("models/manifests") / f"{cfg.detection.model_id}.json"
    onnx_path = _Path("models") / cfg.detection.model_id / "onnx" / "model.onnx"
    if manifest_path.exists() and onnx_path.exists():
        try:
            from .domain import ModelManifest
            from .runtime import OnnxDetectorRuntime

            info = _json.loads(manifest_path.read_text())
            runtime = OnnxDetectorRuntime()
            runtime.load(
                ModelManifest(
                    model_id=info["model_id"],
                    family=info["family"],
                    checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
                    source_url=f"https://huggingface.co/{info['hf_repo']}",
                    code_revision=info.get("hf_revision", "unset"),
                    weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
                    dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
                    approved_for_commercial_use=bool(
                        info.get("approved_for_commercial_use", False)
                    ),
                    input_profile=info.get("input_profile", ""),
                    runtime_profile=info.get("runtime_profile", ""),
                )
            )
            runtime.close()
            check("model_load", "pass", cfg.detection.model_id)
        except Exception as e:
            check("model_load", "fail", f"{type(e).__name__}: {e}")
    else:
        check("model_load", "skip", "weights absent (fetch via scripts/fetch_models.py)")

    if cfg.plomid.dsn:
        try:
            from .persistence import PlomidRepository

            repo = PlomidRepository(
                cfg.plomid.dsn.get_secret_value(), int(cfg.plomid.connect_timeout_s)
            )
            try:
                version = repo.ensure_schema_version(2)
                check("plomid", "pass", f"schema v{version}")
            finally:
                repo.close()
        except Exception as e:
            check("plomid", "fail", f"{type(e).__name__}: {e}")
    else:
        check("plomid", "skip", "ODC_DSN unset")

    import shutil as _shutil

    _, _, free = _shutil.disk_usage(".")
    check("disk_gb_free", "pass" if free > 1_000_000_000 else "fail", f"{free / 1e9:.1f} GB")
    print(_json.dumps({"checks": checks}, indent=2))
    failed = [c for c in checks if c["status"] == "fail"]
    return 1 if failed else 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Provision a database: migrate to current schema, verify, report."""
    cfg = load_config(args.config)
    if not cfg.plomid.dsn:
        log.error("bootstrap needs ODC_DSN")
        return 1
    from .persistence import PlomidRepository

    repo = PlomidRepository(cfg.plomid.dsn.get_secret_value(), int(cfg.plomid.connect_timeout_s))
    try:
        version = repo.ensure_schema_version(2)
        healthy = repo.health_check()
    except Exception as e:
        log.error("bootstrap failed: %s: %s", type(e).__name__, e)
        return 1
    finally:
        repo.close()
    print(f"plomid ready: schema v{version}, health={healthy}")
    return 0 if healthy else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """One-shot full chain on ANY video file: detect → track → state.

    No database writes (transient probe). Unknown geometry → UNCALIBRATED:
    person evidence still reported, seat states stay UNKNOWN, never guessed.
    """
    import json as _json
    from pathlib import Path as _Path

    from .capture import FileReplaySource
    from .config import CameraConfig
    from .domain import ModelManifest
    from .runtime import OnnxDetectorRuntime
    from .seats import SeatLayout, load_layout
    from .state import SeatStateEngine, StateRules
    from .tracking import ByteTrackStyleTracker, TrackerConfig

    path = args.file
    if not _Path(path).exists():
        log.error("file missing: %s", path)
        return 1
    info = _json.loads(_Path("models/manifests/rtdetrv2-s-coco.json").read_text())
    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
        source_url=f"https://huggingface.co/{info['hf_repo']}",
        code_revision=info.get("hf_revision", "unset"),
        weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
        dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
        approved_for_commercial_use=bool(info.get("approved_for_commercial_use", False)),
        input_profile=info.get("input_profile", ""),
        runtime_profile=info.get("runtime_profile", ""),
    )
    if args.layout:
        try:
            layout = load_layout(f"configs/seats/{args.layout}.yaml")
            calibrated = True
        except Exception as e:
            log.error("layout load failed: %s: %s", type(e).__name__, e)
            return 1
    else:
        layout = SeatLayout(camera_profile="uncalibrated", version="none", seats=[])
        calibrated = False
    runtime = OnnxDetectorRuntime(threshold=args.threshold)
    try:
        runtime.load(manifest)
    except Exception as e:
        log.error("detector load failed: %s: %s", type(e).__name__, e)
        return 1
    tracker = ByteTrackStyleTracker(TrackerConfig(high_thresh=0.35, low_thresh=0.15))
    engine = SeatStateEngine("ses-analyze", layout, StateRules())
    source = FileReplaySource(CameraConfig(path=path, camera_id="probe"))
    person_frames = 0
    person_total = 0
    max_persons = 0
    classes: dict[str, int] = {}
    states_out: list[Any] = []
    events_out: list[Any] = []
    try:
        source.open()
        n = 0
        suppressed_total = 0
        duplicates_total = 0
        while args.max_frames is None or n < args.max_frames:
            frame = source.next()
            if frame is None or frame.pixels is None:
                break
            n += 1
            result = runtime.infer(frame)
            # Same live-pipeline post-processing (per-class gates + cabin
            # filter + NMS) so probe counts match the dashboard, not raw
            # duplicates/hallucinations.
            try:
                from .segmentation import PER_CLASS_THRESHOLDS, postprocess_detections

                gates = dict(PER_CLASS_THRESHOLDS)
                gates["person"] = args.threshold
                final, stats = postprocess_detections(result.detections, gates, 0.5)
                suppressed_total += int(stats.get("suppressed", 0))
                duplicates_total += int(stats.get("duplicates", 0))
            except Exception:
                final = result.detections
            for det in final:
                classes[det.class_label] = classes.get(det.class_label, 0) + 1
            persons = [d for d in final if d.class_label == "person"]
            person_total += len(persons)
            max_persons = max(max_persons, len(persons))
            if persons:
                person_frames += 1
            states = tracker.update(frame, persons)
            objects = [d for d in final if d.class_label != "person"]
            out = engine.update(frame, states, layout.seats, objects)
            states_out = out.seats
            events_out.extend(out.events)
    finally:
        source.close()
        runtime.close()
    print(
        _json.dumps(
            {
                "file": path,
                "frames": n,
                "calibrated": calibrated,
                "layout": layout.version,
                "person_frames": person_frames,
                "person_detections_total": person_total,
                "max_persons_one_frame": max_persons,
                "suppressed_hallucinations": suppressed_total,
                "duplicates_removed": duplicates_total,
                "other_classes": {k: v for k, v in classes.items() if k != "person"},
                "final_seats": [
                    {
                        "seat_id": s.seat_id,
                        "presence": s.presence.value,
                        "confidence": s.confidence,
                        "reasons": s.reason_codes,
                        "object": s.object_label,
                    }
                    for s in states_out
                ],
                "transitions": len(events_out),
            },
            indent=2,
        )
    )
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    """Experimental attribute mining on one clip (Phase 7 follow-up).

    Runs the gd-v1 miner; prints candidate boxes. Outputs are CANDIDATES —
    verdicts stay UNSUPPORTED until gated. Never persisted, never safety.
    """
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(".").resolve()))
    from scripts.eval_attributes import main as eval_main

    out = args.out or f"/tmp/mine-{args.clip}.json"
    return eval_main(["--clips", args.clip, "--frames", str(args.frames), "--out", out])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odc", description="PLOMID ODC")
    p.add_argument("--config", default=None, help="YAML config file (optional)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    sub.add_parser("config-check")
    h = sub.add_parser("health")
    h.add_argument("--probe-db", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--max-frames", type=int, default=None)
    d = sub.add_parser("detect")
    d.add_argument("--max-frames", type=int, default=None)
    d.add_argument("--threshold", type=float, default=None)
    d.add_argument("--out", default=None)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument(
        "--auto-start",
        action="store_true",
        help="start the default session immediately (default: idle console)",
    )
    v = sub.add_parser("validate")
    v.add_argument("--dataset", default="eval/datasets/v1.yaml")
    v.add_argument("--split", default="dev", choices=["dev", "holdout"])
    v.add_argument("--threshold", type=float, default=None)
    v.add_argument("--out", default="reports/v1-dev")
    v.add_argument("--compare", default=None)
    v.add_argument("--record-to-db", action="store_true")
    pu = sub.add_parser("purge")
    pu.add_argument("--dir", default="reports")
    pu.add_argument("--older-than-days", type=int, default=30)
    sub.add_parser("selftest")
    sub.add_parser("bootstrap")
    prof = sub.add_parser("profile")
    prof.add_argument("--clip", default="footage/F1_family_ride.mp4")
    prof.add_argument("--max-frames", type=int, default=30)
    prof.add_argument("--threads", type=int, default=None)
    sub.add_parser("providers")
    an = sub.add_parser("analyze")
    an.add_argument("file")
    an.add_argument("--layout", default=None)
    an.add_argument("--max-frames", type=int, default=60)
    an.add_argument("--threshold", type=float, default=0.3)
    mn = sub.add_parser("mine")
    mn.add_argument(
        "--clip", required=True, choices=["F1", "C1", "C2", "E1", "N1", "S1", "S2", "R1"]
    )
    mn.add_argument("--frames", type=int, default=3)
    mn.add_argument("--out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        if args.cmd == "version":
            return cmd_version(args)
        if args.cmd == "config-check":
            return cmd_config_check(args)
        if args.cmd == "health":
            return cmd_health(args)
        if args.cmd == "replay":
            return cmd_replay(args)
        if args.cmd == "detect":
            return cmd_detect(args)
        if args.cmd == "serve":
            return cmd_serve(args)
        if args.cmd == "validate":
            return cmd_validate(args)
        if args.cmd == "purge":
            return cmd_purge(args)
        if args.cmd == "selftest":
            return cmd_selftest(args)
        if args.cmd == "bootstrap":
            return cmd_bootstrap(args)
        if args.cmd == "selftest":
            return cmd_selftest(args)
        if args.cmd == "bootstrap":
            return cmd_bootstrap(args)
        if args.cmd == "profile":
            return cmd_profile(args)
        if args.cmd == "providers":
            return cmd_providers(args)
        if args.cmd == "analyze":
            return cmd_analyze(args)
        if args.cmd == "mine":
            return cmd_mine(args)
    except Exception as e:
        log.error("command failed: %s: %s", type(e).__name__, e)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
