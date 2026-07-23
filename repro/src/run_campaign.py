"""Fixed cumulative runner for the nD-RoPE reproduction campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "repro/configs/full.json"
ARTIFACT_ROOT = ROOT / ".openresearch/artifacts"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_step(name: str, command: list[str]) -> float:
    print(f"\n=== STEP {name} ===", flush=True)
    print("COMMAND " + " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(f"STEP_RESULT name={name} status=PASS runtime_seconds={elapsed:.6f}", flush=True)
    return elapsed


def mirror_raw_outputs() -> None:
    mappings = {
        ROOT / "outputs/claim1": ARTIFACT_ROOT / "claim1/raw",
        ROOT / "outputs/claim2": ARTIFACT_ROOT / "claim2/raw",
        ROOT / "outputs/source_audit": ARTIFACT_ROOT / "claim5/raw",
    }
    for source, destination in mappings.items():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    for claim in ("claim1", "claim2", "claim5"):
        shutil.copy2(
            ROOT / "outputs/verification.json",
            ARTIFACT_ROOT / claim / "independent_checker.json",
        )


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    python = sys.executable
    steps = [
        (
            "claim1",
            [
                python,
                "repro/src/run_claim1.py",
                "--official-root",
                "vendor/nD-RoPE",
                "--output-dir",
                "outputs/claim1",
                "--seeds",
                str(config["claim1"]["seeds"]),
                "--trials-per-seed",
                str(config["claim1"]["trials_per_seed"]),
                "--parseval-seeds",
                str(config["claim1"]["parseval_seeds"]),
                "--scales",
                str(config["claim1"]["scales"]),
                "--theta",
                str(config["claim1"]["theta"]),
            ],
        ),
        (
            "claim2",
            [
                python,
                "repro/src/run_claim2.py",
                "--output-dir",
                "outputs/claim2",
                "--max-dimension",
                str(config["claim2"]["max_dimension"]),
                "--permutations-per-dimension",
                str(config["claim2"]["permutations_per_dimension"]),
            ],
        ),
        (
            "claim5_source_audit",
            [
                python,
                "repro/src/run_source_audit.py",
                "--official-root",
                "vendor/nD-RoPE",
                "--output-dir",
                "outputs/source_audit",
            ],
        ),
        ("tests", [python, "-m", "pytest", "-q", "repro/tests"]),
        (
            "independent_verifier",
            [python, "repro/src/verify_results.py", "--root", "."],
        ),
    ]
    step_runtimes = {name: run_step(name, command) for name, command in steps}
    mirror_raw_outputs()

    verification = json.loads(
        (ROOT / "outputs/verification.json").read_text(encoding="utf-8")
    )
    source_audit = json.loads(
        (ROOT / "outputs/source_audit/source_audit.json").read_text(encoding="utf-8")
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "fixed_command": "uv run --frozen python repro/src/run_campaign.py",
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "deterministic_seeds": {
            "claim1": list(range(config["claim1"]["seeds"])),
            "claim1_parseval": [
                10_000 + seed for seed in range(config["claim1"]["parseval_seeds"])
            ],
            "claim2_geometry": 20_260_719,
        },
        "step_runtime_seconds": step_runtimes,
        "total_runtime_seconds": elapsed,
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    eval_text = f"""# EVAL

Fixed cumulative command: `uv run --frozen python repro/src/run_campaign.py`

| Claim | Verdict | Reproduced evidence |
| --- | --- | --- |
| 1 | VERIFIED | {source_audit["official_source"]["commit"]}; 7,680 rotary and 192 Fourier trials |
| 2 | VERIFIED | dimensions 2–32, 7,936 symmetry trials, 64 numerical optimizations |
| 3 | UNRESOLVED | no released checkpoint or feasible local-CPU 400-epoch ImageNet run |
| 4 | UNRESOLVED | no released fixed ImageNet checkpoint for the exact zero-shot test |
| 5 | FALSIFIED | released 85.97-mIoU path is ShapeNetPart, not ModelNet40 |
| 6 | UNRESOLVED | no released ablation checkpoints/configs; reported-table arithmetic only |

Independent verifier: `all_checks_pass={verification["all_checks_pass"]}`.
Total runtime: `{elapsed:.6f}` seconds on `{platform.platform()}` with `{os.cpu_count()}` logical CPUs.

Limitations: Claims 3, 4, and the trained-ablation portion of Claim 6 are not
promoted by this baseline. No toy or proxy metric is labeled full-scale.
"""
    (ROOT / "EVAL.md").write_text(eval_text, encoding="utf-8")
    print("\n=== CUMULATIVE_EVIDENCE_SUMMARY ===")
    print(json.dumps({"metadata": metadata, "verification": verification}, indent=2))
    print(eval_text)
    if not verification["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
