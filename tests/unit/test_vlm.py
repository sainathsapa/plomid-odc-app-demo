"""Phase 7 unit tests — prompts pinned, mapping math, no-training guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odc.vlm import PROMPT_SET_VERSION, cxcywh_to_frame_xyxy


def test_prompt_set_pinned_to_ontology():
    import scripts.eval_attributes as eval_script

    assert eval_script.PROMPT_VERSION == PROMPT_SET_VERSION == "gd-v1"
    assert "rear facing infant car seat" in eval_script.GD_PROMPTS
    assert "seat belt across chest and torso" in eval_script.GD_PROMPTS
    assert len(eval_script.GD_PROMPTS) == 10


def test_box_mapping_clips():
    x0, y0, x1, y1 = cxcywh_to_frame_xyxy((0.84, 0.64, 0.3, 0.71))
    assert (x0, y0, x1, y1) == pytest.approx((0.69, 0.285, 0.99, 0.995))
    x0, y0, x1, y1 = cxcywh_to_frame_xyxy((2.0, -1.0, 0.5, 0.5))
    assert 0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0


def test_attribute_labels_schema():
    labels = json.loads(Path("eval/attribute_labels.json").read_text())
    assert labels["_note"].startswith("Eval-only")
    clips = {k: v for k, v in labels.items() if not k.startswith("_")}
    assert len(clips) >= 8
    assert "child-seat" in labels["C1_baby_carseat"]
    assert labels["E1_empty_cabin"] == ["empty-cabin", "day"]
    allowed = {
        "person",
        "person-multiple",
        "child-seat",
        "small-occupant",
        "belt",
        "belt-action",
        "object",
        "empty-cabin",
        "day",
        "night",
    }
    for clip, tags in clips.items():
        assert set(tags) <= allowed, (clip, tags)


def test_no_training_imports_anywhere():
    banned = ("torch.optim", "LoraConfig", "peft", "Trainer(", ".fit(", "fine_tune", "finetune")
    roots = [Path("odc"), Path("scripts")]
    hits = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            for token in banned:
                if token in text:
                    hits.append(f"{path}:{token}")
    assert hits == [], hits


def test_manifests_record_provenance():
    grounding = json.loads(Path("models/manifests/grounding-dino-swint-ogc.json").read_text())
    assert grounding["approved_for_commercial_use"] is False
    assert grounding["checkpoint_sha256"] and len(grounding["checkpoint_sha256"]) == 64
    assert grounding["code_revision"] and "never onboard" in grounding["role"]
