"""Independent fail-closed verifier for generated nD-RoPE evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, object]:
    c1_path = root / "outputs/claim1/claim1_report.json"
    c2_path = root / "outputs/claim2/claim2_report.json"
    audit_path = root / "outputs/source_audit/source_audit.json"
    c6_path = root / "outputs/claim6/claim6_report.json"
    c6_data_path = root / "repro/data/paper_claim6.json"
    c1 = json.loads(c1_path.read_text(encoding="utf-8"))
    c2 = json.loads(c2_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    c6 = json.loads(c6_path.read_text(encoding="utf-8"))
    c6_data = json.loads(c6_data_path.read_text(encoding="utf-8"))
    c6_bases = c6_data["frequency_base"]["bases"]
    c6_training_points = c6_data["frequency_base"]["training_points"]
    c6_training_row = next(
        row
        for row in c6_data["frequency_base"]["rows"]
        if row["input_points"] == c6_training_points
    )
    theta2 = c6_training_row["miou"][c6_bases.index(2)]
    theta100 = c6_training_row["miou"][c6_bases.index(100)]
    c6_allocations = [
        row["scales"] * row["heads"] for row in c6_data["scale_head"]["rows"]
    ]

    with (root / "outputs/claim1/translation_trials.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        translation = list(csv.DictReader(handle))
    with (root / "outputs/claim2/geometry_cases.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        geometry = list(csv.DictReader(handle))
    with (root / "outputs/claim2/economy_cases.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        economy = list(csv.DictReader(handle))

    checks = {
        "claim1_all_checks": c1["assessment"] == "verified" and all(c1["checks"].values()),
        "claim1_trial_count": len(translation) == c1["trial_counts"]["rotary"],
        "claim1_raw_translation_bound": max(float(row["translation_error"]) for row in translation) < 1e-10,
        "claim1_raw_relative_bound": max(float(row["relative_error"]) for row in translation) < 1e-10,
        "claim1_negative_control": max(float(row["additive_control_shift"]) for row in translation) > 1e-3,
        "claim2_all_checks": c2["assessment"] == "verified" and all(c2["checks"].values()),
        "claim2_geometry_count": len(geometry) == c2["geometry_cases"],
        "claim2_rank": all(int(row["rank"]) == int(row["dimension"]) for row in geometry),
        "claim2_economy_count": len(economy) == c2["economy_cases"],
        "claim2_numeric_optimum": max(float(row["absolute_error"]) for row in economy) < 1e-6,
        "source_commit": audit["official_source"]["commit"] == "f2cae70760806451f5e58be4b7e3dc4d0d856a1e",
        "source_has_no_checkpoints": audit["official_source"]["checkpoint_count"] == 0,
        "claim5_falsification": audit["claim5_cross_modal"]["assessment"] == "falsified_as_written",
        "claim5_shapenet_entrypoint": audit["claim5_cross_modal"]["entrypoints_use_shapenet_part_path"],
        "claim5_not_modelnet_entrypoint": not audit["claim5_cross_modal"]["entrypoints_use_modelnet_loader"],
        "claim5_16_categories_50_parts": audit["claim5_cross_modal"]["uses_16_shape_categories"] and audit["claim5_cross_modal"]["uses_50_part_labels"],
        "honest_claim3_scope": not audit["claim3_imagenet"]["paper_metric_reproduced"],
        "honest_claim4_scope": not audit["claim4_rotation"]["paper_metric_reproduced"],
        "honest_claim6_scope": not audit["claim6_ablation_cost"]["paper_metrics_reproduced"],
        "claim6_verdict": c6["verdict"] == "FALSIFIED",
        "claim6_exact_training_grid": c6_training_points == 2048,
        "claim6_raw_counterexample_values": theta2 == 85.80
        and theta100 == 85.58,
        "claim6_raw_counterexample_strict": theta2 > theta100,
        "claim6_report_counterexample_matches_raw": (
            c6["decisive_counterexamples"][0]["input_points"] == c6_training_points
            and c6["decisive_counterexamples"][0]["better_base"] == 2
            and abs(
                c6["decisive_counterexamples"][0]["margin_percentage_points"]
                - (theta2 - theta100)
            )
            < 1e-12
        ),
        "claim6_raw_channel_inconsistency": c6_allocations
        == [64, 64, 64, 60, 64, 64, 64],
        "claim6_negative_controls": all(c6["negative_controls"].values()),
        "claim6_frequency_buffer_not_parameter": (
            c6["cost_consistency"]["official_ndrope_freqs_registered_buffer"]
            and not c6["cost_consistency"]["official_ndrope_freqs_learnable_parameter"]
        ),
    }
    report = {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "artifact_sha256": {
            str(path.relative_to(root)): digest(path)
            for path in (c1_path, c2_path, audit_path, c6_path, c6_data_path)
        },
    }
    output = root / "outputs/verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


if __name__ == "__main__":
    result = verify(parse_args().root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_checks_pass"] else 1)
