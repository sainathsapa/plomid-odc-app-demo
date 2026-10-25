"""ONNX detector runtime (Phase 4) — deployment inference path.

Preprocessing MUST use PIL-BILINEAR (proven by parity: max|dbox| 3e-5 vs
MPS-native; cv2.INTER_LINEAR diverges ~1.0). Providers obey the manifest
allowlist — CoreML is rejected for rtdetrv2-s with recorded evidence.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import contracts
from .coco_labels import label_name
from .domain import BBox, Detection, ModelManifest
from .logging import get_logger

log = get_logger("runtime")

ROOT = Path(__file__).resolve().parent.parent
INPUT_SIZE = 640
_SESSION_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple[Any, list[str]]] = {}
_SESSION_CACHE_LOCK = threading.Lock()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → float32 NCHW. PIL path is parity-mandated (see above)."""
    rgb = bgr[:, :, ::-1]
    pil = Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def decode(
    logits: np.ndarray,
    boxes: np.ndarray,
    model_id: str,
    frame_seq: int,
    threshold: float,
) -> list[Detection]:
    """logits [300,80] + boxes [300,4] cxcywh-normalized → domain detections."""
    scores = sigmoid(logits)
    best = scores.argmax(axis=1)
    conf = scores[np.arange(scores.shape[0]), best]
    out: list[Detection] = []
    for cls, score, box in zip(best.tolist(), conf.tolist(), boxes.tolist(), strict=True):
        if score < threshold:
            continue
        cx, cy, w, h = (float(v) for v in box)
        x0, y0 = min(max(cx - w / 2, 0.0), 1.0), min(max(cy - h / 2, 0.0), 1.0)
        x1, y1 = min(max(cx + w / 2, 0.0), 1.0), min(max(cy + h / 2, 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(
            Detection(
                class_label=label_name(int(cls)),
                score=float(score),
                bbox=BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
                model_id=model_id,
                frame_seq=frame_seq,
            )
        )
    return out


class ProviderDegenerate(Exception):
    """A provider returned numerically degenerate output (e.g. all-equal
    logits — the measured CoreML failure signature). Fail closed."""


def select_providers(preference: list[str], blocked: set[str], available: list[str]) -> list[str]:
    """GPU-first order minus measured-corrupt providers. Pure, tested."""
    return [p for p in preference if p in available and p not in blocked]


def _reject_degenerate(logits: np.ndarray, provider: str) -> None:
    spread = float(logits.max() - logits.min())
    if not np.isfinite(spread) or spread < 1e-6:
        raise ProviderDegenerate(
            f"provider {provider} returned degenerate logits (spread {spread:.2e}); "
            "refusing to emit garbage detections"
        )


def load_manifest(model_id: str) -> ModelManifest:
    """Read a versioned model manifest (checksum + license + runtime pins).

    Shared by the live service and the per-video seat-mapping probe so both
    run the exact same weights and provider policy.
    """
    info = json.loads((ROOT / "models" / "manifests" / f"{model_id}.json").read_text())
    return ModelManifest(
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


class OnnxDetectorRuntime(contracts.DetectorRuntime):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._session: Any = None
        self._providers: list[str] = []
        self._model_id = ""
        self._input_name = "images"

    @property
    def providers_active(self) -> list[str]:
        return list(self._providers)

    def load(self, manifest: ModelManifest, sess_options: Any = None) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        model_dir = ROOT / "models" / manifest.model_id
        onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing ONNX weights: {onnx_path}")
        deploy = json.loads(
            (model_dir.parent / "manifests" / f"{manifest.model_id}.json").read_text()
        )
        # Preference order: GPU-class providers first (CUDA/TensorRT/CoreML),
        # CPU last. Blocked providers (measured corrupt) are never tried.
        # ORT has no MPS provider; torch-MPS stays the native reference path.
        preference = deploy.get(
            "providers_preference",
            ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        blocked = set(deploy.get("providers_blocked", {}))
        available = ort.get_available_providers()
        wanted = select_providers(preference, blocked, available)
        if not wanted:
            raise RuntimeError(
                f"no usable provider (preference {preference}, blocked {sorted(blocked)}, "
                f"available {available})"
            )
        cache_key = (manifest.model_id, str(onnx_path.resolve()), tuple(wanted))
        with _SESSION_CACHE_LOCK:
            cached = _SESSION_CACHE.get(cache_key) if sess_options is None else None
            if cached is None:
                session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=sess_options or ort.SessionOptions(),
                    providers=wanted,
                )
                active = session.get_providers()
                if sess_options is None:
                    _SESSION_CACHE[cache_key] = (session, active)
                log.info("onnx load model=%s providers=%s", manifest.model_id, active)
            else:
                session, active = cached
        self._session = session
        if active[0] in blocked:
            raise RuntimeError(f"provider fell through to blocked {active}")
        self._providers = active
        self._model_id = manifest.model_id

    def infer(self, frame: contracts.Frame) -> contracts.InferenceResult:
        assert self._session is not None, "runtime not loaded"
        pixels = frame.pixels
        if pixels is None:
            return contracts.InferenceResult(detections=[], model_id=self._model_id)
        start = time.monotonic()
        batch = preprocess(pixels)
        logits, boxes = self._session.run(["logits", "pred_boxes"], {"images": batch})[:2]
        logits = np.asarray(logits)
        _reject_degenerate(logits, self._providers[0] if self._providers else "unknown")
        duration_ms = (time.monotonic() - start) * 1000
        detections = decode(
            logits[0],
            np.asarray(boxes)[0],
            self._model_id,
            frame.meta.seq,
            self._threshold,
        )
        return contracts.InferenceResult(
            detections=detections,
            model_id=self._model_id,
            provider=self._providers[0] if self._providers else "unknown",
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        self._session = None
        self._providers = []
