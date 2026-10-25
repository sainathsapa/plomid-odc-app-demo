"""Export RT-DETRv2-S to ONNX (Phase 4) — reproducible, provenance-recorded.

Legacy torch exporter (matches upstream's own export path); dynamo fails on
data-dependent assert ops in this architecture. Tracer warnings are expected;
the parity gate (integration test) decides correctness, not the warnings.
Usage: .venv/bin/python scripts/export_onnx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "rtdetrv2-s-coco"
OPSET = 17


def main() -> int:
    import torch
    from transformers import RTDetrV2ForObjectDetection

    src = ROOT / "models" / MODEL_ID
    out_dir = src / "onnx"
    out_dir.mkdir(exist_ok=True)
    model = RTDetrV2ForObjectDetection.from_pretrained(src, local_files_only=True)
    model.eval()
    dummy = torch.zeros(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy,
        out_dir / "model.onnx",
        input_names=["images"],
        output_names=["logits", "pred_boxes"],
        opset_version=OPSET,
        dynamo=False,
    )
    import onnx

    onnx_model = onnx.load(out_dir / "model.onnx")
    info = {
        "model_id": MODEL_ID,
        "opset": onnx_model.opset_import[0].version,
        "inputs": [
            (i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
            for i in onnx_model.graph.input
        ],
        "outputs": [
            (o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
            for o in onnx_model.graph.output
        ][:2],
        "torch_version": torch.__version__,
        "exporter": "legacy torch.onnx (upstream path); dynamo rejected (aten._is_all_true)",
        "note": "tracer baked data-dependent branches as constants; "
        "parity vs MPS-native on 3 cabin frames is the gate",
    }
    (out_dir / "onnx_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
