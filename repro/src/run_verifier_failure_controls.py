"""Prove that the cumulative verifier fails closed for every claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


CASES = [
    {
        "claim": 1,
        "name": "corrupt_claim1_universal_proof",
        "path": "outputs/symbolic_proof_certificates.json",
        "keys": ["checks", "claim1_shared_translation_cancels_symbolically"],
        "value": False,
        "expected_failed_check": "universal_symbolic_proofs",
    },
    {
        "claim": 2,
        "name": "corrupt_claim2_universal_proof",
        "path": "outputs/symbolic_proof_certificates.json",
        "keys": ["checks", "claim2_economy_derivative_identity"],
        "value": False,
        "expected_failed_check": "universal_symbolic_proofs",
    },
    {
        "claim": 3,
        "name": "corrupt_claim3_blocked_verdict",
        "path": "outputs/claim34/claim34_report.json",
        "keys": ["verdicts", "claim3"],
        "value": "VERIFIED",
        "expected_failed_check": "claim34_verdict_stage",
    },
    {
        "claim": 4,
        "name": "corrupt_claim4_blocked_verdict",
        "path": "outputs/claim34/claim34_report.json",
        "keys": ["verdicts", "claim4"],
        "value": "VERIFIED",
        "expected_failed_check": "claim34_verdict_stage",
    },
    {
        "claim": 5,
        "name": "corrupt_claim5_source_verdict",
        "path": "outputs/source_audit/source_audit.json",
        "keys": ["claim5_cross_modal", "assessment"],
        "value": "CORRUPTED",
        "expected_failed_check": "claim5_falsification",
    },
    {
        "claim": 6,
        "name": "corrupt_claim6_dynamic_verdict",
        "path": "outputs/claim6/dynamic_flop_report.json",
        "keys": ["verdict"],
        "value": "VERIFIED",
        "expected_failed_check": "claim6_dynamic_flop_verdict",
    },
]


def mutate(path: Path, keys: list[str], value: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    verifier = root / "repro/src/verify_results.py"
    started = time.perf_counter()
    results = []

    for case in CASES:
        with tempfile.TemporaryDirectory(prefix=f"ndrope-c{case['claim']}-") as tmp:
            case_root = Path(tmp)
            shutil.copytree(root / "outputs", case_root / "outputs")
            shutil.copytree(root / "repro/data", case_root / "repro/data")
            prior_controls = case_root / "outputs/verifier_failure_controls.json"
            if prior_controls.exists():
                prior_controls.unlink()
            evidence_path = case_root / str(case["path"])
            mutate(evidence_path, list(case["keys"]), case["value"])
            completed = subprocess.run(
                [sys.executable, str(verifier), "--root", str(case_root)],
                check=False,
                capture_output=True,
                text=True,
            )
            verification = json.loads(
                (case_root / "outputs/verification.json").read_text(
                    encoding="utf-8"
                )
            )
            failed_checks = sorted(
                name
                for name, passed in verification["checks"].items()
                if not passed
            )
            expected = str(case["expected_failed_check"])
            results.append(
                {
                    **case,
                    "verifier_exit_code": completed.returncode,
                    "all_checks_pass": verification["all_checks_pass"],
                    "failed_checks": failed_checks,
                    "expected_check_failed": expected in failed_checks,
                    "stdout_sha256_recorded_by_parent_verifier": False,
                    "stderr_was_empty": completed.stderr == "",
                }
            )

    report = {
        "purpose": (
            "Mutation controls proving the published cumulative verifier exits "
            "nonzero when each claim's decisive evidence or verdict is corrupted."
        ),
        "cases": results,
        "all_mutations_rejected": all(
            case["verifier_exit_code"] != 0
            and not case["all_checks_pass"]
            and case["expected_check_failed"]
            for case in results
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_mutations_rejected"] else 1)


if __name__ == "__main__":
    main()
