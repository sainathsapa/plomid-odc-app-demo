"""Open-vocabulary miner adapter (Phase 7) — Grounding DINO, offline only.

Local loader (no `supervision` dep): config + checkpoint at pinned paths.
Runs on full frames or seat-ROI crops; outputs namespaced `vlm:` phrases in
frame-normalized coords. NEVER onboard, NEVER real-time, NEVER a safety
state by itself. MPS attempted, CPU fallback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .logging import get_logger

log = get_logger("vlm")

CODE_DIR = Path(__file__).resolve().parent.parent / "models" / "grounding-dino-code"
PROMPT_SET_VERSION = "gd-v1"


def cxcywh_to_frame_xyxy(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Normalized cxcywh → clipped frame-normalized xyxy."""
    cx, cy, w, h = (float(v) for v in box)
    return (
        min(max(cx - w / 2, 0.0), 1.0),
        min(max(cy - h / 2, 0.0), 1.0),
        min(max(cx + w / 2, 0.0), 1.0),
        min(max(cy + h / 2, 0.0), 1.0),
    )


@dataclass(frozen=True)
class GroundedPhrase:
    phrase: str
    score: float
    bbox: tuple[float, float, float, float]  # frame-normalized xyxy
    prompt_version: str = PROMPT_SET_VERSION


def _ensure_code() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))


def _patch_bert_for_legacy_code() -> None:
    """GroundingDINO's 2024 code calls BertModel.get_extended_attention_mask
    / invert_attention_mask, removed in transformers 5.x. Reattach the
    classic implementations on the class (eval-only glue; logged)."""
    try:
        from transformers.models.bert.modeling_bert import BertModel
    except Exception:
        return
    if not hasattr(BertModel, "get_extended_attention_mask"):

        def _extended(self: Any, attention_mask: Any, input_shape: Any, device: Any = None) -> Any:
            import torch

            dtype = torch.float32
            if attention_mask.dim() == 3:
                extended = attention_mask[:, None, :, :]
            else:
                extended = attention_mask[:, None, None, :]
            extended = extended.to(dtype=dtype)
            return (1.0 - extended) * torch.finfo(dtype).min

        BertModel.get_extended_attention_mask = _extended  # type: ignore[attr-defined]
        log.info("vlm compat: reattached BertModel.get_extended_attention_mask")
    if not hasattr(BertModel, "invert_attention_mask"):

        def _invert(self: Any, encoder_attention_mask: Any) -> Any:
            import torch

            dtype = torch.float32
            return (1.0 - encoder_attention_mask.to(dtype=dtype)) * torch.finfo(dtype).min

        BertModel.invert_attention_mask = _invert  # type: ignore[attr-defined]
        log.info("vlm compat: reattached BertModel.invert_attention_mask")
    if not hasattr(BertModel, "get_head_mask"):

        def _head_mask(
            self: Any, head_mask: Any, num_hidden_layers: Any, is_attention_chunked: Any = False
        ) -> Any:
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        BertModel.get_head_mask = _head_mask  # type: ignore[attr-defined]
        log.info("vlm compat: reattached BertModel.get_head_mask")


@lru_cache(maxsize=4)
def load_grounding_dino(
    config_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    tokenizer_path: str = "bert-base-uncased",
) -> tuple[Any, Any]:
    """Returns (model, tokenizer). Raises clearly when unavailable."""
    _ensure_code()
    _patch_bert_for_legacy_code()
    from groundingdino.models import build_model  # type: ignore[import-not-found]
    from groundingdino.util.misc import clean_state_dict  # type: ignore[import-not-found]
    from groundingdino.util.slconfig import SLConfig  # type: ignore[import-not-found]
    from transformers import AutoTokenizer

    args = SLConfig.fromfile(config_path)
    local_bert = _cached_hf_snapshot("bert-base-uncased")
    if local_bert is not None:
        args.text_encoder_type = str(local_bert)
        from groundingdino.util import get_tokenlizer as _tokenizer_module

        original_tokenizer = _tokenizer_module.get_tokenlizer
        original_language_model = _tokenizer_module.get_pretrained_language_model

        def _local_tokenizer(text_encoder_type: Any) -> Any:
            return (
                original_tokenizer(str(local_bert))
                if text_encoder_type == "bert-base-uncased"
                else original_tokenizer(text_encoder_type)
            )

        def _local_language_model(text_encoder_type: Any) -> Any:
            return (
                original_language_model(str(local_bert))
                if text_encoder_type == "bert-base-uncased"
                else original_language_model(text_encoder_type)
            )

        _tokenizer_module.get_tokenlizer = _local_tokenizer
        _tokenizer_module.get_pretrained_language_model = _local_language_model
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
    log.info("grounding-dino loaded missing=%d unexpected=%d", len(missing), len(unexpected))
    model.to(device)
    model.eval()
    # ODC inference is local-only.  ``local_files_only`` prevents a healthy
    # checkpoint from silently turning into a network-dependent runtime.
    cached_tokenizer = _cached_hf_snapshot(tokenizer_path)
    if cached_tokenizer is not None:
        tokenizer_path = str(cached_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return model, tokenizer


def _cached_hf_snapshot(model_id: str) -> Path | None:
    """Resolve a local Hugging Face snapshot without contacting the Hub."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = sorted(
        (cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*")
    )
    return snapshots[-1] if snapshots else None


def preprocess_pil(
    bgr: np.ndarray, shortest: int = 800, longest: int = 1333
) -> tuple[object, tuple[int, int], tuple[int, int]]:
    """BGR → normalized tensor + (orig_wh, resized_wh). Matches repo transform:
    shortest edge → 800, longest capped at 1333, ImageNet normalize."""
    import torchvision.transforms as T  # type: ignore[import-untyped]

    rgb = bgr[:, :, ::-1]
    h, w = rgb.shape[0], rgb.shape[1]
    scale = shortest / min(h, w)
    if max(h, w) * scale > longest:
        scale = longest / max(h, w)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    pil = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
    tensor = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(pil)
    return tensor, (w, h), (nw, nh)


@torch.no_grad()
def ground(
    model: Any,
    tokenizer: Any,
    image_tensor: Any,
    caption: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Verbatim replication of groundingdino.util.inference.predict
    (minus the supervision-based annotate helper). Returns
    (boxes_cxcywh_0to1, scores, phrases) — raw miner output."""
    import bisect

    from groundingdino.util.utils import get_phrases_from_posmap  # type: ignore[import-not-found]

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model.to(device)
    tensor = image_tensor.to(device)
    outputs = model(tensor[None], captions=[caption])
    logits_all = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes_all = outputs["pred_boxes"].cpu()[0]
    mask = logits_all.max(dim=1)[0] > box_threshold
    logits = logits_all[mask]
    boxes = boxes_all[mask]
    tokenized = tokenizer(caption)
    sep_idx = [
        i
        for i in range(len(tokenized["input_ids"]))
        if tokenized["input_ids"][i] in [101, 102, 1012]
    ]
    phrases = []
    for logit in logits:
        max_idx = int(logit.argmax())
        insert_idx = bisect.bisect_left(sep_idx, max_idx)
        right_idx = sep_idx[insert_idx]
        left_idx = sep_idx[insert_idx - 1]
        phrases.append(
            get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer, left_idx, right_idx
            ).replace(".", "")
        )
    return boxes.numpy(), logits.max(dim=1)[0].numpy(), phrases
