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
    c1 = json.loads(c1_path.read_text(encoding="utf-8"))
    c2 = json.loads(c2_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

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
    }
    report = {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "artifact_sha256": {
            str(path.relative_to(root)): digest(path)
            for path in (c1_path, c2_path, audit_path)
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
