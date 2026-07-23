# Conclusion


---
<!-- trackio-cell
{"type": "code", "id": "cell_531701654a80", "created_at": "2026-07-19T17:13:54+00:00", "title": "Run: env (exit 0)", "command": ["env", "PYTHONPATH=repro/src", "pytest", "-q", "repro/tests"], "exit_code": 0, "duration_s": 0.35}
-->
````bash
$ env PYTHONPATH=repro/src pytest -q repro/tests
````

exit 0 · 0.3s


````output
....................                                                     [100%]
20 passed in 0.10s

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_cac62baedafe", "created_at": "2026-07-19T17:13:54+00:00", "title": "Run: env verify_results.py (exit 0)", "command": ["env", "PYTHONPATH=repro/src", "python", "repro/src/verify_results.py", "--root", "."], "exit_code": 0, "duration_s": 0.086}
-->
````bash
$ env PYTHONPATH=repro/src python repro/src/verify_results.py --root .
````

exit 0 · 0.1s


````python title=verify_results.py
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

````


````output
{
  "all_checks_pass": true,
  "checks": {
    "claim1_all_checks": true,
    "claim1_trial_count": true,
    "claim1_raw_translation_bound": true,
    "claim1_raw_relative_bound": true,
    "claim1_negative_control": true,
    "claim2_all_checks": true,
    "claim2_geometry_count": true,
    "claim2_rank": true,
    "claim2_economy_count": true,
    "claim2_numeric_optimum": true,
    "source_commit": true,
    "source_has_no_checkpoints": true,
    "claim5_falsification": true,
    "claim5_shapenet_entrypoint": true,
    "claim5_not_modelnet_entrypoint": true,
    "claim5_16_categories_50_parts": true,
    "honest_claim3_scope": true,
    "honest_claim4_scope": true,
    "honest_claim6_scope": true
  },
  "artifact_sha256": {
    "outputs/claim1/claim1_report.json": "a488222f228e447148739b18240ec616fc00c9cb6a44b4a6414592109df3d97a",
    "outputs/claim2/claim2_report.json": "75af99be54dd9cf35deab7f3be3c24cf24cfffe48a6e7cca071cc9e7af846f52",
    "outputs/source_audit/source_audit.json": "0cf180982bf4f902c481142563d2ff2714193f5a5689580bc9eead17ee345856"
  }
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a69f98f66c9b", "created_at": "2026-07-19T17:16:52+00:00", "title": "Conclusion"}
-->
**Outcome: two claims verified, one claim falsified as written, three empirical claims unresolved.** C1's Fourier/translation mechanism and C2's simplex/economy construction pass exact independent certificates; C5's ModelNet40-mIoU subclaim is contradicted by the pinned executable source, which runs ShapeNetPart. C3, C4, and the empirical portion of C6 remain inconclusive because the release has no checkpoints and no SemanticKITTI path; no toy result is promoted.

All 20 tests pass, and an independent verifier re-reads raw CSV/JSON evidence, checks trial counts and bounds, validates source hashes/commit, confirms the ShapeNetPart entrypoint, and enforces the inconclusive labels. Rerun the five relative commands shown in this logbook. The reproduction bundle contains scripts, configs, tests, raw outputs, source inventory, claim audit, and the approach ledger; virtual environments, secrets, and replaceable caches are excluded.


---
<!-- trackio-cell
{"type": "code", "id": "cell_a3c095c8e110", "created_at": "2026-07-19T17:26:53+00:00", "title": "Run: env log_reproduction_artifact.py (exit 0)", "command": ["env", "PYTHONPATH=repro/src", "python", "repro/src/log_reproduction_artifact.py"], "exit_code": 0, "duration_s": 0.989}
-->
````bash
$ env PYTHONPATH=repro/src python repro/src/log_reproduction_artifact.py
````

exit 0 · 1.0s


````python title=log_reproduction_artifact.py
#!/usr/bin/env python3
"""Create the Trackio evidence bundle for OpenReview YPXDQkU7XW."""

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    trackio.init(
        project="nd-rope-reproduction-audit",
        name="full-evidence",
        config={
            "paper": "YPXDQkU7XW",
            "official_commit": "f2cae70760806451f5e58be4b7e3dc4d0d856a1e",
            "claim1_rotary_trials": 7680,
            "claim2_symmetry_trials": 7936,
            "tests": 20,
        },
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    artifact = trackio.Artifact(
        "repro-bundle",
        type="dataset",
        description=(
            "Clean-room Fourier/simplex certificates, pinned-source benchmark audit, "
            "tests, raw results, independent verification, and strict Posterly output "
            "for nD-RoPE (YPXDQkU7XW)."
        ),
        metadata={
            "paper": "YPXDQkU7XW",
            "claim1": "verified",
            "claim2": "verified",
            "claim5": "falsified_as_written",
        },
    )
    for relative in [
        "README.md",
        "STATUS.md",
        "repro/configs/full.json",
        "repro/src/ndrope_core.py",
        "repro/src/official_extract.py",
        "repro/src/run_claim1.py",
        "repro/src/run_claim2.py",
        "repro/src/run_source_audit.py",
        "repro/src/verify_results.py",
        "repro/src/log_reproduction_artifact.py",
        "repro/tests/test_ndrope.py",
        "docs/CLAIM_AUDIT.md",
        "docs/APPROACH_LEDGER.md",
        "poster/poster.html",
        "poster/poster_embed.html",
        "poster/poster_preview.pdf",
        "poster/poster_preview.png",
        "poster/GATE_REPORT.json",
        "poster/CONTENT_AUDIT.md",
        "poster/design_tokens.json",
        "poster/assets/evidence.svg",
        "vendor/nD-RoPE/README.md",
        "vendor/nD-RoPE/rope-vit-ndrope/deit/models_v2_ndRope.py",
        "vendor/nD-RoPE/rope-vit-ndrope/deit/models_v2_rope.py",
        "vendor/nD-RoPE/rope-vit-ndrope/deit/run_ndrope.bash",
        "vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/train_partseg_ndrope.py",
        "vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/test_partseg.py",
        "vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/dataset.py",
    ]:
        artifact.add_file(ROOT / relative, name=relative)
    artifact.add_dir(ROOT / "outputs", name="outputs")
    logged = trackio.log_artifact(artifact, aliases=["verified"])
    print(f"{logged.project}/{logged.name}:v{logged.version}")
    trackio.finish()


if __name__ == "__main__":
    main()

````


````output
* Trackio project initialized: nd-rope-reproduction-audit
* Trackio metrics logged locally (machine-specific cache path omitted)
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "nd-rope-reproduction-audit"[0m
* or by running in Python: trackio.show(project="nd-rope-reproduction-audit")
* Created new run: full-evidence
nd-rope-reproduction-audit/repro-bundle:vv0
* Run finished. Uploading logs to Trackio (please wait...)

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_eb16983725f7", "created_at": "2026-07-19T17:27:06+00:00", "title": "Reproduction bundle", "artifact": "nd-rope-reproduction-audit/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `nd-rope-reproduction-audit/repro-bundle:v0` · dataset

https://huggingface.co/buckets/DineshAI/YPXDQkU7XW-artifacts#nd-rope-reproduction-audit/repro-bundle:v0
