"""Zero-shot attribute evaluation (Phase 7) — offline batch, no training.

For each clip × gd-v1 prompt: sample frames, run Grounding DINO, record
hit rates + example boxes. Compares against eval/attribute_labels.json
clip tags → per-tag recall-ish/false-alarm-ish + FP/FN examples.
Verdicts are written by a human into eval/attribute_verdicts.yaml —
this script produces EVIDENCE, never verdicts.
Usage: .venv/bin/python scripts/eval_attributes.py [--clips F1 C1 ...] [--frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GD_PROMPTS = [
    "a person sitting in a car",
    "rear facing infant car seat",
    "forward facing child car seat",
    "booster seat in a car",
    "seat belt across chest and torso",
    "seat belt behind a person's back",
    "seat belt buckle",
    "blanket covering a car seat",
    "bag on a car seat",
    "child in a car",
]
PROMPT_VERSION = "gd-v1"

CLIPS = {
    "F1": "footage/F1_family_ride.mp4",
    "C1": "footage/C1_baby_carseat.mp4",
    "C2": "footage/C2_child_seat.mp4",
    "E1": "footage/E1_empty_cabin.mp4",
    "N1": "footage/N1_night_drive.mp4",
    "S1": "footage/S1_seatbelt_closeup.mp4",
    "S2": "footage/S2_belt_up.mp4",
    "R1": "footage/R1_rear_wide.mp4",
    "O2": "footage/O2_laptop_rear.mp4" if Path("footage/O2_laptop_rear.mp4").exists() else None,
    "D1": "footage/D1_driver_day.mp4" if Path("footage/D1_driver_day.mp4").exists() else None,
}

TAG_TO_PROMPTS = {
    "person": ["a person sitting in a car"],
    "child-seat": [
        "rear facing infant car seat",
        "forward facing child car seat",
        "booster seat in a car",
    ],
    "belt": ["seat belt across chest and torso", "seat belt buckle"],
    "belt-misuse": ["seat belt behind a person's back"],
    "small-occupant": ["child in a car"],
    "object": ["bag on a car seat"],
    "blanket": ["blanket covering a car seat"],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--out", default="eval/attribute_evidence.json")
    args = parser.parse_args(argv)

    import cv2

    from odc.vlm import ground, load_grounding_dino, preprocess_pil

    model, tokenizer = load_grounding_dino(
        "models/grounding-dino-code/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "models/grounding-dino-swint-ogc.pth",
        device="cpu",
    )
    labels = json.loads(Path("eval/attribute_labels.json").read_text())
    clip_ids = args.clips or [c for c, p in CLIPS.items() if p]
    evidence: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "clips": {}}
    for cid in clip_ids:
        path = CLIPS[cid]
        assert path is not None
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = [int(total * (k + 1) / (args.frames + 1)) for k in range(args.frames)]
        clip_ev: dict[str, Any] = {"path": path, "frames": {}}
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            tensor, _, _ = preprocess_pil(frame)
            frame_ev = []
            for prompt in GD_PROMPTS:
                t = time.monotonic()
                boxes, scores, phrases = ground(model, tokenizer, tensor, prompt, device="cpu")
                frame_ev.append(
                    {
                        "prompt": prompt,
                        "seconds": round(time.monotonic() - t, 1),
                        "hits": [
                            {
                                "phrase": p,
                                "score": round(float(s), 3),
                                "box": [round(float(v), 3) for v in b],
                            }
                            for b, s, p in zip(boxes, scores, phrases, strict=True)
                        ],
                    }
                )
            clip_ev["frames"][str(idx)] = frame_ev
        cap.release()
        tag_key = next((k for k in labels if k.startswith(f"{cid}_")), None)
        clip_ev["tags"] = labels.get(tag_key, []) if tag_key else []
        evidence["clips"][cid] = clip_ev
        print(f"{cid}: {len(clip_ev['frames'])} frames done", flush=True)

    # Per-tag summary: fraction of sampled frames with ≥1 hit per mapped prompt.
    summary: dict[str, Any] = {}
    for tag, prompts in TAG_TO_PROMPTS.items():
        present: list[Any] = []
        absent: list[Any] = []
        for cid, clip_ev in evidence["clips"].items():
            tags = clip_ev.get("tags", [])
            rates = []
            for frame_ev in clip_ev["frames"].values():
                hit = any(h for item in frame_ev if item["prompt"] in prompts for h in item["hits"])
                rates.append(1.0 if hit else 0.0)
            rate = sum(rates) / max(len(rates), 1)
            (present if tag in tags else absent).append((cid, round(rate, 2)))
        summary[tag] = {"present": present, "absent": absent}
    evidence["summary"] = summary
    Path(args.out).write_text(json.dumps(evidence, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
