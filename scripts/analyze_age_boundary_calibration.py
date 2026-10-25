"""Summarize age-boundary behavior on labeled development evaluation reports.

The input reports must be development data only. The analysis aggregates each
person track before scoring so adjacent video frames do not count as independent
examples. It does not fit a probabilistic calibrator or use holdout reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from odc.config import AgeConfig

DEFAULT_REPORTS = (
    "reports/f1-family-ride-current.json",
    "reports/mivolo-C1-dev-fixed.json",
    "reports/current-C2-seat-clipped-posture.json",
    "reports/phase2-MiVOLO-6181423.json",
    "reports/current-adult-only-9519139.json",
)
DEFAULT_CHILD_LIMITS = (6.0, 8.0, 11.0, 14.0, 17.0)
DEFAULT_ADULT_LIMITS = (20.0, 22.0, 23.0, 24.0, 26.0)
AMBIGUOUS_REASONS = {
    "person-box-seat-association-ambiguous",
    "person-box-overlaps-multiple-seats",
    "face-associated-with-different-seat",
}
MINIMUM_DEVELOPMENT_CLEARANCE_YEARS = 1.0


def _read_track_observations(report_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    included_videos: list[str] = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        video = Path(str(data.get("video", path.stem))).name
        included_videos.append(video)
        truth = data.get("ground_truth", {})
        child_seats = set(truth.get("child_seats", []))
        adult_seats = set(truth.get("adult_seats", []))
        for frame in data.get("frames", []):
            for row in frame.get("rows", []):
                seat_id = row.get("seat_id")
                if seat_id not in child_seats | adult_seats:
                    continue
                label = "child" if seat_id in child_seats else "adult"
                for prediction in row.get("age_predictions", []):
                    track_id = prediction.get("track_id")
                    age = prediction.get("age_years")
                    reason = prediction.get("classification_reason", prediction.get("reason"))
                    if (
                        not track_id
                        or not isinstance(age, (int, float))
                        or prediction.get("seat_id") != seat_id
                        or reason in AMBIGUOUS_REASONS
                    ):
                        continue
                    key = (video, str(seat_id), str(track_id), label)
                    track = grouped.setdefault(
                        key,
                        {
                            "video": video,
                            "seat_id": seat_id,
                            "track_id": track_id,
                            "ground_truth": label,
                            "ages": [],
                        },
                    )
                    track["ages"].append(float(age))
    tracks = []
    for track in grouped.values():
        observations = track.pop("ages")
        if len(observations) < 2:
            continue
        tracks.append(
            {
                **track,
                "observation_count": len(observations),
                "median_age_years": round(float(median(observations)), 3),
                "min_age_years": round(min(observations), 3),
                "max_age_years": round(max(observations), 3),
            }
        )
    return sorted(
        tracks, key=lambda item: (item["video"], item["seat_id"], item["track_id"])
    ), included_videos


def _score(
    tracks: list[dict[str, Any]], child_limit: float, adult_limit: float
) -> dict[str, int | float]:
    counts = {
        "child_tracks": 0,
        "adult_tracks": 0,
        "unknown_tracks": 0,
        "adult_as_child": 0,
        "child_as_adult": 0,
    }
    for track in tracks:
        age = track["median_age_years"]
        category = "child" if age <= child_limit else "adult" if age > adult_limit else "unknown"
        truth = track["ground_truth"]
        if category == "unknown":
            counts["unknown_tracks"] += 1
        elif category == truth:
            counts[f"{truth}_tracks"] += 1
        elif truth == "adult":
            counts["adult_as_child"] += 1
        else:
            counts["child_as_adult"] += 1
    return counts


def _select_development_margin(
    tracks: list[dict[str, Any]], semantic_boundary_years: float
) -> dict[str, Any] | None:
    """Choose a conservative integer margin while preserving labeled track medians.

    The one-year clearance is an explicit guard band for the observed development
    extrema, not a statistical confidence interval or a population guarantee.
    """
    children = [item["median_age_years"] for item in tracks if item["ground_truth"] == "child"]
    adults = [item["median_age_years"] for item in tracks if item["ground_truth"] == "adult"]
    if not children or not adults:
        return None
    child_clearance = semantic_boundary_years - max(children)
    adult_clearance = min(adults) - semantic_boundary_years
    available_margin = min(child_clearance, adult_clearance) - MINIMUM_DEVELOPMENT_CLEARANCE_YEARS
    if available_margin < 0:
        return {
            "status": "no_margin_satisfies_development_clearance",
            "semantic_boundary_years": semantic_boundary_years,
            "minimum_clearance_years": MINIMUM_DEVELOPMENT_CLEARANCE_YEARS,
            "child_extreme_clearance_years": round(child_clearance, 3),
            "adult_extreme_clearance_years": round(adult_clearance, 3),
        }
    margin = float(math.floor(available_margin))
    child_limit = semantic_boundary_years - margin
    adult_limit = semantic_boundary_years + margin
    return {
        "status": "development_selected_provisional",
        "selection_method": (
            "largest whole-year symmetric margin around age 17 that preserves all included "
            "development track medians with at least one year clearance from both observed class extrema"
        ),
        "semantic_boundary_years": semantic_boundary_years,
        "minimum_clearance_years": MINIMUM_DEVELOPMENT_CLEARANCE_YEARS,
        "available_margin_before_rounding_years": round(available_margin, 3),
        "selected_margin_years": margin,
        "child_confirmed_max_years_inclusive": child_limit,
        "adult_confirmed_min_years_exclusive": adult_limit,
        "child_extreme_clearance_years": round(child_clearance, 3),
        "adult_extreme_clearance_years": round(adult_clearance, 3),
        "development_track_results": _score(tracks, child_limit, adult_limit),
        "limitations": (
            "provisional development-set selection only; track/video counts are small and correlated; "
            "MiVOLO age is an uncalibrated regression estimate"
        ),
    }


def analyze(report_paths: list[Path]) -> dict[str, Any]:
    tracks, videos = _read_track_observations(report_paths)
    age_config = AgeConfig()
    current_child_limit = age_config.child_max_years - age_config.uncertainty_margin_years
    current_adult_limit = age_config.child_max_years + age_config.uncertainty_margin_years
    child_ages = [item["median_age_years"] for item in tracks if item["ground_truth"] == "child"]
    adult_ages = [item["median_age_years"] for item in tracks if item["ground_truth"] == "adult"]
    candidates = []
    for child_limit in DEFAULT_CHILD_LIMITS:
        for adult_limit in DEFAULT_ADULT_LIMITS:
            if adult_limit < child_limit:
                continue
            candidates.append(
                {
                    "child_confirmed_max_years_inclusive": child_limit,
                    "adult_confirmed_min_years_exclusive": adult_limit,
                    **_score(tracks, child_limit, adult_limit),
                }
            )
    return {
        "purpose": "development-only empirical boundary sensitivity; not probability calibration",
        "reports": [str(path) for path in report_paths],
        "videos": sorted(set(videos)),
        "independent_video_count": len(set(videos)),
        "track_aggregation": "median age per (video, ground-truth seat, track); at least two observations",
        "exclusions": "unlabeled seats, seat-mismatched predictions, ambiguous seat/face associations, one-observation tracks",
        "track_counts": {
            "child": sum(item["ground_truth"] == "child" for item in tracks),
            "adult": sum(item["ground_truth"] == "adult" for item in tracks),
        },
        "observed_track_median_ranges": {
            "child": {"min": min(child_ages), "max": max(child_ages)} if child_ages else None,
            "adult": {"min": min(adult_ages), "max": max(adult_ages)} if adult_ages else None,
            "gap_years": round(min(adult_ages) - max(child_ages), 3)
            if child_ages and adult_ages
            else None,
        },
        "current_policy": {
            "child_confirmed_max_years_inclusive": current_child_limit,
            "adult_confirmed_min_years_exclusive": current_adult_limit,
            **_score(tracks, current_child_limit, current_adult_limit),
        },
        "selected_development_policy": _select_development_margin(
            tracks, age_config.child_max_years
        ),
        "candidate_sweep": candidates,
        "tracks": tracks,
        "interpretation": (
            "Track-level separation in these few development clips cannot establish age-boundary "
            "calibration or population error rates. Candidate settings that produce identical "
            "labels are not distinguishable from this sample. Keep independent reports locked."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="*", type=Path, default=[Path(item) for item in DEFAULT_REPORTS]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/age-boundary-development-analysis.json")
    )
    args = parser.parse_args()
    result = analyze(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "independent_video_count",
                    "track_counts",
                    "observed_track_median_ranges",
                    "current_policy",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
