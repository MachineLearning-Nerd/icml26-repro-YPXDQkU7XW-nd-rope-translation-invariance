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
