from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invoice_grounding import GroundingConfig, ground_invoice_values
from invoice_grounding.visualization import render_grounding_overlay


RESOLVED_STATUSES = {"matched", "inherited"}
REVIEW_STATUSES = {"ambiguous", "unmatched", "error"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate invoice grounding over local sample invoices.")
    parser.add_argument("--data-dir", default="data", help="Directory containing sample_invoice*.png and sample_json*.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write grounding outputs and summary.json")
    parser.add_argument("--baseline-dir", help="Optional previous output directory to compare status transitions")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--include-overlays", action="store_true", help="Render overlay PNGs for every sample")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GroundingConfig(device=args.device, include_ocr_words=True)
    aggregate: Counter[str] = Counter()
    per_sample: dict[str, dict[str, int]] = {}

    for sample_index, image_path, json_path in _sample_pairs(data_dir):
        sample_name = f"sample_{sample_index}"
        print(f"Running {sample_name}...")
        result = ground_invoice_values(image_path, json_path, config=config)
        (output_dir / f"{sample_name}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if args.include_overlays:
            render_grounding_overlay(image_path, result, output_dir / f"{sample_name}_overlay.png")
        counts = Counter(field.status.value for field in result.fields)
        aggregate.update(counts)
        per_sample[sample_name] = dict(sorted(counts.items()))

    summary: dict[str, Any] = {
        "total_fields": sum(aggregate.values()),
        "aggregate": dict(sorted(aggregate.items())),
        "per_sample": per_sample,
    }
    if args.baseline_dir:
        summary["comparison"] = _compare_outputs(Path(args.baseline_dir), output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _sample_pairs(data_dir: Path) -> list[tuple[int, Path, Path]]:
    pairs: list[tuple[int, Path, Path]] = []
    for image_path in data_dir.glob("sample_invoice*.png"):
        sample_index = _sample_index(image_path)
        suffix = "" if sample_index == 1 else f"_{sample_index}"
        json_path = data_dir / f"sample_json{suffix}.json"
        if json_path.exists():
            pairs.append((sample_index, image_path, json_path))
    return sorted(pairs, key=lambda item: item[0])


def _sample_index(path: Path) -> int:
    stem = path.stem
    if stem == "sample_invoice":
        return 1
    return int(stem.rsplit("_", 1)[1])


def _compare_outputs(baseline_dir: Path, current_dir: Path) -> dict[str, Any]:
    transition_counts: Counter[str] = Counter()
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    changed_matches: list[dict[str, Any]] = []

    for current_path in sorted(current_dir.glob("sample_*.json"), key=lambda path: _result_sort_key(path)):
        baseline_path = baseline_dir / current_path.name
        if not baseline_path.exists():
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = json.loads(current_path.read_text(encoding="utf-8"))
        baseline_fields = {field["json_path"]: field for field in baseline.get("fields", [])}
        current_fields = {field["json_path"]: field for field in current.get("fields", [])}
        sample_name = current_path.stem

        for json_path, current_field in current_fields.items():
            baseline_field = baseline_fields.get(json_path)
            if baseline_field is None:
                continue
            old_status = baseline_field.get("status")
            new_status = current_field.get("status")
            transition_counts[f"{old_status}->{new_status}"] += 1
            if old_status in REVIEW_STATUSES and new_status in RESOLVED_STATUSES:
                improvements.append(_transition_item(sample_name, json_path, baseline_field, current_field))
            elif old_status in RESOLVED_STATUSES and new_status in REVIEW_STATUSES:
                regressions.append(_transition_item(sample_name, json_path, baseline_field, current_field))
            elif (
                old_status in RESOLVED_STATUSES
                and new_status in RESOLVED_STATUSES
                and baseline_field.get("word_ids") != current_field.get("word_ids")
            ):
                changed_matches.append(_transition_item(sample_name, json_path, baseline_field, current_field))

    return {
        "status_transitions": dict(sorted(transition_counts.items())),
        "improvements": improvements,
        "regressions": regressions,
        "changed_matches": changed_matches,
    }


def _result_sort_key(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _transition_item(
    sample_name: str,
    json_path: str,
    baseline_field: dict[str, Any],
    current_field: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample": sample_name,
        "json_path": json_path,
        "value_as_text": current_field.get("value_as_text"),
        "old_status": baseline_field.get("status"),
        "new_status": current_field.get("status"),
        "old_matched_text": baseline_field.get("matched_text"),
        "new_matched_text": current_field.get("matched_text"),
        "old_word_ids": baseline_field.get("word_ids", []),
        "new_word_ids": current_field.get("word_ids", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
