"""PLOMID-backed seat store: seat maps, calibration, probe + analytics reuse.

Videos stay on disk (footage/ is the only filesystem dependency); every
seat map, calibration profile, probe result and analytics snapshot lives
in PLOMID tables (migration V003). Saved work survives restarts, reloads
automatically with the same video, and invalidates by content hash when
any input changes.

Keying: each video is fingerprinted by size + mtime + bounded content
hash (first 2 MB), so renames still match identical content while edits
invalidate. Analytics keys additionally mix the layout and config hashes,
so only the affected results regenerate after a calibration or settings
change.

Every loader is total: missing rows, bad JSON, schema drift, an outdated
cache or an unreachable database return (None, reason) — never raise into
the request path. Stale data never masquerades as fresh.

``configs/seats/*.yaml`` remain as versioned seed files: the first load of
a curated name imports it into PLOMID (origin='seed'); from then on the
database row is the source of truth.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Repository
from .logging import get_logger

log = get_logger("seat_store")

SEED_DIR = Path("configs/seats")
_HASH_BYTES = 2 * 1024 * 1024


def video_fingerprint(path: str | Path) -> dict[str, Any]:
    """Stable identity for a video file. Never raises: missing/unreadable
    files yield {"ok": False, "reason": ...} so callers degrade honestly."""
    src = Path(path)
    try:
        stat = src.stat()
    except Exception as e:
        return {"ok": False, "reason": f"missing file: {type(e).__name__}"}
    if stat.st_size <= 0:
        return {"ok": False, "reason": "empty file"}
    digest = hashlib.sha256()
    try:
        with src.open("rb") as fh:
            digest.update(fh.read(_HASH_BYTES))
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    except Exception as e:
        return {"ok": False, "reason": f"unreadable: {type(e).__name__}"}
    short = digest.hexdigest()[:16]
    stem = src.stem[:32] or "video"
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)
    return {
        "ok": True,
        "key": f"{safe}-{short}",
        "sha": digest.hexdigest(),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def layout_fingerprint(layout: Any) -> str:
    """Hash of the seat geometry that analytics depend on. Pure."""
    try:
        seats = getattr(layout, "seats", []) or []
        canonical = sorted(
            (
                str(getattr(s, "seat_id", "?")),
                [[round(float(x), 4), round(float(y), 4)] for x, y in s.polygon],
                str(getattr(s, "camera_profile", "")),
                str(getattr(s, "version", "")),
            )
            for s in seats
        )
        blob = json.dumps(canonical, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return "unknown"


def layout_to_json(layout: Any) -> dict[str, Any]:
    """SeatLayout -> JSON-ready doc for PLOMID storage. Pure."""
    seats: list[dict[str, Any]] = []
    for s in getattr(layout, "seats", []) or []:
        entry: dict[str, Any] = {
            "seat_id": str(s.seat_id),
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in s.polygon],
            "camera_profile": str(getattr(s, "camera_profile", "custom")),
            "version": str(getattr(s, "version", "v1")),
        }
        if getattr(s, "row", None) is not None:
            entry["row"] = int(s.row)
        if getattr(s, "col", None) is not None:
            entry["col"] = int(s.col)
        if getattr(s, "label", None):
            entry["label"] = str(s.label)
        if getattr(s, "display_number", None) is not None:
            entry["display_number"] = int(s.display_number)
        if getattr(s, "physical_role", None):
            entry["physical_role"] = str(s.physical_role)
        seats.append(entry)
    return {
        "camera_profile": str(getattr(layout, "camera_profile", "custom")),
        "version": str(getattr(layout, "version", "v1")),
        "layout_type": str(getattr(layout, "layout_type", "custom")),
        "orientation": str(getattr(layout, "orientation", "RHD")),
        **({"orientation_reason": str(layout.orientation_reason)} if getattr(layout, "orientation_reason", None) else {}),
        **({"display_name": str(layout.display_name)} if getattr(layout, "display_name", None) else {}),
        "seats": seats,
    }


def layout_from_json(doc: Any) -> Any:
    """JSON doc -> SeatLayout. Raises ValueError with a reason on bad data."""
    from .domain import SeatGeometry
    from .seats import SeatLayout

    if not isinstance(doc, dict) or not isinstance(doc.get("seats"), list):
        raise ValueError("bad schema: need {camera_profile, version, seats[]}")
    profile = str(doc.get("camera_profile") or "custom")
    version = str(doc.get("version") or "saved-v1")
    seats = []
    for s in doc["seats"]:
        if not isinstance(s, dict) or not isinstance(s.get("polygon"), list):
            raise ValueError("bad schema: seat needs {seat_id, polygon}")
        seats.append(
            SeatGeometry(
                seat_id=str(s["seat_id"]),
                polygon=[(float(x), float(y)) for x, y in s["polygon"]],
                camera_profile=str(s.get("camera_profile", profile)),
                version=str(s.get("version", version)),
                row=int(s["row"]) if isinstance(s.get("row"), int) else None,
                col=int(s["col"]) if isinstance(s.get("col"), int) else None,
                label=str(s["label"]) if isinstance(s.get("label"), str) else None,
                display_number=int(s["display_number"]) if isinstance(s.get("display_number"), int) else None,
                physical_role=str(s["physical_role"]) if isinstance(s.get("physical_role"), str) else None,
            )
        )
    return SeatLayout(
        camera_profile=profile,
        version=version,
        seats=seats,
        layout_type=str(doc.get("layout_type") or "custom"),
        orientation=str(doc.get("orientation") or "RHD"),
        orientation_reason=str(doc["orientation_reason"]) if isinstance(doc.get("orientation_reason"), str) else None,
        display_name=str(doc["display_name"]) if isinstance(doc.get("display_name"), str) else None,
    )


def _track_video(repo: Repository, video_path: str | Path, fp: dict[str, Any]) -> None:
    """Register the video asset row (best-effort; failures only log)."""
    try:
        repo.upsert_video_asset(
            {
                "video_key": fp["key"],
                "video_path": str(video_path),
                "video_sha": fp["sha"],
                "size_bytes": fp["size"],
                "mtime": fp["mtime"],
            }
        )
    except Exception as e:
        log.warning("video asset track degraded: %s", type(e).__name__)


def save_video_seatmap(
    repo: Repository,
    video_path: str | Path,
    layout: Any,
    camera_id: str = "",
    calibration_profile: str = "",
    origin: str = "auto",
) -> str | None:
    """Persist the seat map bound to one video in PLOMID. Returns the
    video key, or None when the video is missing or PLOMID is down."""
    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        log.warning("seatmap not saved: %s", fp.get("reason"))
        return None
    try:
        _track_video(repo, video_path, fp)
        doc = layout_to_json(layout)
        repo.save_video_seatmap(
            {
                "video_key": fp["key"],
                "video_sha": fp["sha"],
                "camera_id": camera_id,
                "calibration_profile": calibration_profile
                or doc.get("camera_profile", ""),
                "layout_version": doc.get("version", ""),
                "origin": origin,
                "layout": doc,
            }
        )
        return str(fp["key"])
    except Exception as e:
        log.warning("seatmap save degraded: %s", type(e).__name__)
        return None


def load_video_seatmap(
    repo: Repository, video_path: str | Path
) -> tuple[Any | None, str]:
    """Reload the seat map for this video from PLOMID, or (None, reason)."""
    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        return None, str(fp.get("reason"))
    try:
        row = repo.load_video_seatmap(str(fp["key"]))
    except Exception as e:
        return None, f"seat store unavailable ({type(e).__name__})"
    if row is None:
        # Same path, different key = the file changed since the map was
        # saved (edits invalidate; nothing stale is ever returned).
        try:
            others = repo.list_video_seatmaps_by_path(str(video_path))
        except Exception:
            others = []
        if others:
            return None, "video changed since the saved seat map — recalibration needed"
        return None, "no saved seat map for this video"
    if row.get("video_sha") and row["video_sha"] != fp["sha"]:
        return None, "saved map is for a different cut of this video"
    try:
        return layout_from_json(row.get("layout")), (
            f"reloaded from PLOMID ({row.get('origin', 'auto')}, "
            f"{row.get('layout_version', '')})"
        )
    except ValueError as e:
        return None, f"invalid saved geometry ({e})"


def save_probe_cache(repo: Repository, video_path: str | Path, probe: Any) -> bool:
    """Cache an autolayout probe in PLOMID so unchanged videos skip
    re-probing. Returns False when not stored (never raises)."""
    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        return False
    try:
        _track_video(repo, video_path, fp)
        layout = getattr(probe, "layout", None)
        repo.save_probe_entry(
            {
                "video_key": fp["key"],
                "video_sha": fp["sha"],
                "frames_probed": int(getattr(probe, "frames_probed", 0)),
                "persons_seen": int(getattr(probe, "persons_seen", 0)),
                "peaks": int(getattr(probe, "peaks", 0)),
                "reason": str(getattr(probe, "reason", "")),
                "layout": None if layout is None else layout_to_json(layout),
            }
        )
        return True
    except Exception as e:
        log.warning("probe cache save degraded: %s", type(e).__name__)
        return False


def load_probe_cache(
    repo: Repository, video_path: str | Path, max_age_s: float = 7 * 86400
) -> tuple[Any | None, str]:
    """Cached ProbeResult for an unchanged video, else (None, reason)."""
    from .autolayout import ProbeResult

    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        return None, str(fp.get("reason"))
    try:
        row = repo.load_probe_entry(str(fp["key"]))
    except Exception as e:
        return None, f"seat store unavailable ({type(e).__name__})"
    if row is None:
        try:
            assets = repo.list_video_assets_by_path(str(video_path))
        except Exception:
            assets = []
        if [a for a in assets if a.get("video_key") != fp["key"]]:
            return None, "video changed since probe"
        return None, "no cached probe"
    if row.get("video_sha") != fp["sha"]:
        return None, "video changed since probe"
    try:
        updated = datetime.fromisoformat(str(row.get("updated_at") or "")) if row.get(
            "updated_at"
        ) else None
        if updated is not None:
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - updated).total_seconds()
            if age > max_age_s:
                return None, "cached probe expired"
    except Exception:
        pass
    try:
        raw = row.get("layout")
        layout = None if raw is None else layout_from_json(raw)
        return ProbeResult(
            layout=layout,
            frames_probed=int(row.get("frames_probed", 0)),
            persons_seen=int(row.get("persons_seen", 0)),
            peaks=int(row.get("peaks", 0)),
            reason=str(row.get("reason", "cached")),
        ), "reused cached probe (video unchanged)"
    except ValueError as e:
        return None, f"invalid probe cache ({e})"


def analytics_cache_key(
    video_path: str | Path, layout: Any, config_hash: str
) -> str | None:
    """Cache key mixing video + layout + config hashes. Pure."""
    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        return None
    raw = f"{fp['sha']}|{layout_fingerprint(layout)}|{config_hash}".encode()
    return f"{fp['key']}-{hashlib.sha256(raw).hexdigest()[:12]}"


def config_hash_for(cfg: Any) -> str:
    """Hash of the settings that analytics depend on: model, threshold,
    mask mode, pose/face toggles and state-rules version. Pure — any change
    here invalidates cached snapshots so only affected results regenerate."""
    try:
        det = getattr(cfg, "detection", None)
        pose = getattr(cfg, "pose", None)
        dms = getattr(cfg, "dms", None)
        state = getattr(cfg, "state", None)
        canonical = {
            "model": str(getattr(det, "model_id", "")),
            "threshold": round(float(getattr(det, "threshold", 0.0)), 4),
            "mask": str(getattr(det, "mask_mode", "")),
            "pose": bool(getattr(pose, "enabled", False)),
            "dms": bool(getattr(dms, "enabled", False)),
            "state": str(getattr(state, "version", "")),
        }
        blob = json.dumps(canonical, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return "unknown"


def save_analytics_snapshot(
    repo: Repository,
    video_path: str | Path,
    layout: Any,
    config_hash: str,
    payload: dict[str, Any],
) -> str | None:
    """Persist an analytics snapshot in PLOMID; reused while video+layout+
    config are unchanged. Returns the key, or None when not stored."""
    key = analytics_cache_key(video_path, layout, config_hash)
    if key is None:
        return None
    fp = video_fingerprint(video_path)
    if not fp.get("ok"):
        return None
    try:
        _track_video(repo, video_path, fp)
        repo.save_analytics_entry(
            {
                "cache_key": key,
                "video_key": fp["key"],
                "layout_hash": layout_fingerprint(layout),
                "config_hash": config_hash,
                "payload": payload,
            }
        )
        return key
    except Exception as e:
        log.warning("analytics save degraded: %s", type(e).__name__)
        return None


def load_analytics_snapshot(
    repo: Repository, video_path: str | Path, layout: Any, config_hash: str
) -> tuple[dict[str, Any] | None, str]:
    """Reuse cached analytics when inputs are unchanged, else (None, reason
    naming what changed so only affected results regenerate)."""
    key = analytics_cache_key(video_path, layout, config_hash)
    if key is None:
        return None, "video unavailable for cache key"
    try:
        row = repo.load_analytics_entry(key)
    except Exception as e:
        return None, f"seat store unavailable ({type(e).__name__})"
    if row is None:
        return None, "no cached analytics for this configuration — recalculation needed"
    if row.get("layout_hash") != layout_fingerprint(layout):
        return None, "seat layout changed — recalculation needed"
    if row.get("config_hash") != config_hash:
        return None, "settings changed — recalculation needed"
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None, "invalid analytics payload"
    return payload, "reused cached analytics (inputs unchanged)"


def save_named_layout(
    repo: Repository,
    name: str,
    seats: list[dict[str, Any]],
    camera_profile: str | None = None,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """Validate and store a calibration layout in PLOMID.

    Returns (True, layout_name) or (False, error). Duplicate names need
    overwrite=True. Pure validation first (layouts module); PLOMID second.
    """
    from .layouts import validate_layout_payload

    errors = validate_layout_payload(name, seats)
    if errors:
        return False, "; ".join(errors)
    try:
        existing = repo.load_seat_layout(name)
    except Exception as e:
        return False, f"seat store unavailable ({type(e).__name__})"
    if existing is not None and not overwrite:
        return False, f"layout {name} exists (tick overwrite to replace it)"
    profile = camera_profile or (existing or {}).get("camera_profile") or f"custom-{name}"
    doc = {
        "camera_profile": profile,
        "version": name,
        "origin": "hand",
        "seats": [
            {
                "seat_id": seat["seat_id"],
                "polygon": [
                    [round(float(x), 4), round(float(y), 4)] for x, y in seat["polygon"]
                ],
                "camera_profile": profile,
                "version": name,
                **({"row": int(seat["row"])} if isinstance(seat.get("row"), int) else {}),
                **({"col": int(seat["col"])} if isinstance(seat.get("col"), int) else {}),
                **(
                    {"label": str(seat["label"])}
                    if isinstance(seat.get("label"), str) and seat.get("label")
                    else {}
                ),
            }
            for seat in seats
        ],
    }
    try:
        repo.upsert_seat_layout(name, doc)
        return True, name
    except Exception as e:
        return False, f"save failed ({type(e).__name__})"


def load_named_layout(
    repo: Repository, name: str, seed_dir: str | Path = SEED_DIR
) -> tuple[Any | None, str]:
    """Load a curated layout: PLOMID first, versioned YAML seed fallback.

    The first load of a seed name imports it into PLOMID (origin='seed');
    from then on the database row is the source of truth.
    """
    try:
        doc = repo.load_seat_layout(name)
    except Exception as e:
        doc = None
        store_error = f"seat store unavailable ({type(e).__name__})"
    else:
        store_error = ""
    if doc is not None:
        try:
            return layout_from_json(
                {
                    "camera_profile": doc.get("camera_profile", ""),
                    "version": doc.get("version", name),
                    "layout_type": doc.get("layout_type", "custom"),
                    "orientation": doc.get("orientation", "RHD"),
                    "display_name": doc.get("display_name"),
                    "seats": doc.get("seats", []),
                }
            ), f"layout {name} from PLOMID"
        except ValueError as e:
            return None, f"invalid stored layout {name} ({e})"
    # Seed fallback: configs/seats/<name>.yaml, imported on first use.
    try:
        from .seats import load_layout as _load_file

        layout = _load_file(str(Path(seed_dir) / f"{name}.yaml"))
    except Exception:
        return None, store_error or f"unknown layout {name}"
    try:
        repo.upsert_seat_layout(
            name,
            {
                "camera_profile": layout.camera_profile,
                "version": layout.version,
                "layout_type": layout.layout_type,
                "orientation": layout.orientation,
                "origin": "seed",
                "seats": [
                    {
                        "seat_id": s.seat_id,
                        "polygon": [[float(x), float(y)] for x, y in s.polygon],
                        "camera_profile": s.camera_profile,
                        "version": s.version,
                        **({} if s.row is None else {"row": int(s.row)}),
                        **({} if s.col is None else {"col": int(s.col)}),
                        **({} if not s.label else {"label": str(s.label)}),
                        **({} if s.display_number is None else {"display_number": int(s.display_number)}),
                        **({} if not s.physical_role else {"physical_role": str(s.physical_role)}),
                    }
                    for s in layout.seats
                ],
            },
        )
    except Exception as e:
        log.warning("seed import degraded: %s", type(e).__name__)
    return layout, f"layout {name} from seed (imported to PLOMID)"


def list_named_layouts(
    repo: Repository, seed_dir: str | Path = SEED_DIR
) -> list[str]:
    """Curated layout names: PLOMID rows plus YAML seeds (union, sorted).

    Read-only: seeding happens on load, never on list.
    """
    names: set[str] = set()
    try:
        for row in repo.list_seat_layouts():
            if row.get("layout_name"):
                names.add(str(row["layout_name"]))
    except Exception as e:
        log.warning("layout list degraded: %s", type(e).__name__)
    try:
        for path in Path(seed_dir).glob("*.yaml"):
            names.add(path.stem)
    except Exception:
        pass
    return sorted(names)
