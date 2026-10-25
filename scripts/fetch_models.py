"""Fetch + verify model checkpoints (Phase 4). No training, no execution.

Reads models/manifests/*.json, downloads pinned files at pinned revisions
into models/<model_id>/, verifies sha256, writes CHECKSUMS.txt.
Usage: .venv/bin/python scripts/fetch_models.py [--model ID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "models" / "manifests"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(manifest_path: Path) -> None:
    from huggingface_hub import snapshot_download

    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model_id"]
    if "hf_repo" not in manifest:
        # Candidate/eval-only manifests (D-FINE challenger, Grounding DINO)
        # pin a direct checkpoint URL instead of an HF snapshot. They are not
        # part of the default runtime, so leave them to their own fetch path.
        print(f"{model_id}: skipped (no hf_repo; not an HF snapshot manifest)")
        return
    dest = ROOT / "models" / model_id
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=manifest["hf_repo"],
        revision=manifest["hf_revision"],
        allow_patterns=manifest["files"],
        local_dir=dest,
    )
    lines = []
    for name in manifest["files"]:
        got = sha256_of(dest / name)
        want = manifest.get("sha256", {}).get(name)
        status = "OK" if (want in (None, "FILL-BY-FETCH", got)) else "MISMATCH"
        if want not in (None, "FILL-BY-FETCH") and got != want:
            raise SystemExit(f"checksum MISMATCH for {name}: got {got} want {want}")
        lines.append(f"{got}  {name}  [{status}]")
        if want == "FILL-BY-FETCH":
            manifest["sha256"][name] = got
    (dest / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")
    if manifest["sha256"].get("model.safetensors", "").startswith(("FILL", "")):
        pass
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{model_id}: verified\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    manifests = sorted(MANIFESTS.glob("*.json"))
    if args.model:
        manifests = [p for p in manifests if p.stem == args.model]
        if not manifests:
            print(f"unknown model {args.model}")
            return 1
    for path in manifests:
        fetch_one(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
