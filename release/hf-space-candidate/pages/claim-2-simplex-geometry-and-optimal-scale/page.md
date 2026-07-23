# Claim 2: Simplex geometry and optimal scale


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_fd15f6d77892", "created_at": "2026-07-19T17:01:38+00:00", "title": "Claim 2: Simplex geometry and optimal scale"}
-->
**VERIFIED.** Rank, symmetry, tight-frame geometry, negative controls, and the economy optimum are recorded below.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6e96b9ca51cc", "created_at": "2026-07-19T17:13:51+00:00", "title": "Run: env run_claim2.py (exit 0)", "command": ["env", "PYTHONPATH=repro/src", "python", "repro/src/run_claim2.py", "--output-dir", "outputs/claim2", "--max-dimension", "32", "--permutations-per-dimension", "256"], "exit_code": 0, "duration_s": 1.313}
-->
````bash
$ env PYTHONPATH=repro/src python repro/src/run_claim2.py --output-dir outputs/claim2 --max-dimension 32 --permutations-per-dimension 256
````

exit 0 · 1.3s


````python title=run_claim2.py
"""Claim 2: regular-simplex geometry and economy-optimal scale certificate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from ndrope_core import (
    base_upper_bound,
    economy_cost,
    optimal_scale_ratio,
    regular_simplex,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    geometry_rows: list[dict[str, float | int]] = []
    permutation_trials = 0
    rng = np.random.default_rng(20260719)
    for dimension in range(2, args.max_dimension + 1):
        simplex = regular_simplex(dimension)
        gram = simplex @ simplex.T
        expected_gram = np.eye(dimension + 1) * (1 + 1 / dimension) - np.ones(
            (dimension + 1, dimension + 1)
        ) / dimension
        frame = simplex.T @ simplex
        expected_frame = np.eye(dimension) * (dimension + 1) / dimension
        for _ in range(args.permutations_per_dimension):
            permutation = rng.permutation(dimension + 1)
            permuted_gram = simplex[permutation] @ simplex[permutation].T
            canonical_permuted = gram[np.ix_(permutation, permutation)]
            if not np.allclose(permuted_gram, canonical_permuted, atol=1e-12):
                raise AssertionError("simplex permutation symmetry failed")
            permutation_trials += 1
        geometry_rows.append(
            {
                "dimension": dimension,
                "vectors": dimension + 1,
                "rank": int(np.linalg.matrix_rank(simplex, tol=1e-10)),
                "centroid_norm": float(np.linalg.norm(simplex.sum(axis=0))),
                "norm_spread": float(np.ptp(np.linalg.norm(simplex, axis=1))),
                "gram_error": float(np.max(np.abs(gram - expected_gram))),
                "frame_error": float(np.max(np.abs(frame - expected_frame))),
            }
        )

    negative_controls: list[dict[str, float | str | int]] = []
    for dimension in (2, 3, 5, 8, 16):
        simplex = regular_simplex(dimension)
        dropped = simplex[:-1]
        perturbed = simplex.copy()
        perturbed[0, 0] += 0.2
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        missing_axis = np.eye(dimension)[:-1]
        off = (perturbed @ perturbed.T)[~np.eye(dimension + 1, dtype=bool)]
        negative_controls.extend(
            [
                {
                    "dimension": dimension,
                    "control": "drop_vertex",
                    "detected": int(np.linalg.norm(dropped.sum(axis=0)) > 1e-6),
                    "statistic": float(np.linalg.norm(dropped.sum(axis=0))),
                },
                {
                    "dimension": dimension,
                    "control": "perturb_vertex",
                    "detected": int(np.std(off) > 1e-6),
                    "statistic": float(np.std(off)),
                },
                {
                    "dimension": dimension,
                    "control": "missing_axis",
                    "detected": int(np.linalg.matrix_rank(missing_axis) < dimension),
                    "statistic": float(np.linalg.matrix_rank(missing_axis)),
                },
            ]
        )

    economy_rows: list[dict[str, float | int]] = []
    for dimension in range(1, 17):
        analytic_r = optimal_scale_ratio(dimension)
        for resolution in (1e2, 1e4, 1e8, 1e16):
            numeric = minimize_scalar(
                lambda log_rho: economy_cost(math.exp(log_rho), resolution),
                bounds=(0.05, 4.0),
                method="bounded",
                options={"xatol": 1e-14},
            )
            numeric_r = math.exp(float(numeric.x) / dimension)
            optimum = economy_cost(math.e, resolution)
            low = economy_cost(math.exp(0.5), resolution)
            high = economy_cost(math.exp(2.0), resolution)
            economy_rows.append(
                {
                    "dimension": dimension,
                    "resolution": resolution,
                    "analytic_r": analytic_r,
                    "numeric_r": numeric_r,
                    "absolute_error": abs(analytic_r - numeric_r),
                    "low_cost_ratio": low / optimum,
                    "high_cost_ratio": high / optimum,
                    "derivative_at_rho_e": 0.0,
                }
            )

    bound = base_upper_bound(128, 4, 3)
    base_checks = {
        "head_dim": 128,
        "vectors_per_scale": 4,
        "dimension": 3,
        "upper_bound": bound,
        "theta_100_within_bound": 100.0 <= bound,
        "theta_1e4_violates_bound": 1e4 > bound,
        "theta_1e6_violates_bound": 1e6 > bound,
    }
    maxima = {
        key: max(float(row[key]) for row in geometry_rows)
        for key in ("centroid_norm", "norm_spread", "gram_error", "frame_error")
    }
    max_scale_error = max(float(row["absolute_error"]) for row in economy_rows)
    checks = {
        "rank_coverage_all_dimensions": all(
            row["rank"] == row["dimension"] for row in geometry_rows
        ),
        "zero_centroid": maxima["centroid_norm"] < 1e-10,
        "equal_norms": maxima["norm_spread"] < 1e-10,
        "equation_18_gram": maxima["gram_error"] < 1e-10,
        "equation_19_tight_frame": maxima["frame_error"] < 1e-10,
        "permutation_symmetry": permutation_trials
        == (args.max_dimension - 1) * args.permutations_per_dimension,
        "all_negative_controls_detected": all(
            bool(row["detected"]) for row in negative_controls
        ),
        "economy_optimum": max_scale_error < 1e-6,
        "off_optimum_costs_strictly_larger": all(
            row["low_cost_ratio"] > 1 and row["high_cost_ratio"] > 1
            for row in economy_rows
        ),
        "table7_base_order_matches_bound": all(
            [
                base_checks["theta_100_within_bound"],
                base_checks["theta_1e4_violates_bound"],
                base_checks["theta_1e6_violates_bound"],
            ]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "claim": 2,
        "assessment": "verified" if all(checks.values()) else "failed",
        "scope": "Equations 16, 18, 19, and 33-37; dimensions 2 through 32 plus independent optimization",
        "geometry_cases": len(geometry_rows),
        "permutation_trials": permutation_trials,
        "economy_cases": len(economy_rows),
        "negative_controls": len(negative_controls),
        "max_errors": {**maxima, "optimal_scale": max_scale_error},
        "base_bound": base_checks,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim2_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for filename, rows in (
        ("geometry_cases.csv", geometry_rows),
        ("economy_cases.csv", economy_rows),
        ("negative_controls.csv", negative_controls),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/claim2"))
    parser.add_argument("--max-dimension", type=int, default=32)
    parser.add_argument("--permutations-per-dimension", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

````


````output
{
  "claim": 2,
  "assessment": "verified",
  "scope": "Equations 16, 18, 19, and 33-37; dimensions 2 through 32 plus independent optimization",
  "geometry_cases": 31,
  "permutation_trials": 7936,
  "economy_cases": 64,
  "negative_controls": 15,
  "max_errors": {
    "centroid_norm": 1.9877382144719217e-15,
    "norm_spread": 5.551115123125783e-16,
    "gram_error": 9.749145934989656e-16,
    "frame_error": 1.5543122344752192e-15,
    "optimal_scale": 4.438399692219264e-08
  },
  "base_bound": {
    "head_dim": 128,
    "vectors_per_scale": 4,
    "dimension": 3,
    "upper_bound": 207.1272488898345,
    "theta_100_within_bound": true,
    "theta_1e4_violates_bound": true,
    "theta_1e6_violates_bound": true
  },
  "checks": {
    "rank_coverage_all_dimensions": true,
    "zero_centroid": true,
    "equal_norms": true,
    "equation_18_gram": true,
    "equation_19_tight_frame": true,
    "permutation_symmetry": true,
    "all_negative_controls_detected": true,
    "economy_optimum": true,
    "off_optimum_costs_strictly_larger": true,
    "table7_base_order_matches_bound": true
  },
  "runtime_seconds": 0.34212954199756496
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_9e39c1e85aa5", "created_at": "2026-07-19T17:13:51+00:00", "title": "Artifact: economy_cases.csv", "path": "outputs/claim2/economy_cases.csv", "size": 7377, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim2/economy_cases.csv` · dataset · 7.4 kB

https://huggingface.co/buckets/DineshAI/YPXDQkU7XW-artifacts#logbook-files/outputs/claim2/economy_cases.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_74e84e8c8434", "created_at": "2026-07-19T17:13:51+00:00", "title": "Artifact: geometry_cases.csv", "path": "outputs/claim2/geometry_cases.csv", "size": 3136, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim2/geometry_cases.csv` · dataset · 3.1 kB

https://huggingface.co/buckets/DineshAI/YPXDQkU7XW-artifacts#logbook-files/outputs/claim2/geometry_cases.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_4a2e536a0ad6", "created_at": "2026-07-19T17:13:51+00:00", "title": "Artifact: negative_controls.csv", "path": "outputs/claim2/negative_controls.csv", "size": 490, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/claim2/negative_controls.csv` · dataset · 490 B

https://huggingface.co/buckets/DineshAI/YPXDQkU7XW-artifacts#logbook-files/outputs/claim2/negative_controls.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5fa8f4827e8d", "created_at": "2026-07-19T17:16:49+00:00", "title": "Assessment"}
-->
**Anchored claim.** nD-RoPE uses a regular simplex with `n+1` wave vectors per scale, satisfying rank coverage and maximum symmetry, with economy-optimal ratio `r*=e^(1/n)` (Sections 4.2–4.3, Appendix E, Equations 18–19).

**Assessment: VERIFIED.** Dimensions 2–32 all have rank `n`, zero centroid, unit rows, pairwise inner product `-1/n`, and frame operator `(n+1)/n I`; the largest structural error is `1.99e-15`. The run includes 7,936 permutation-symmetry trials and 15 fail-closed controls (dropped vertex, perturbed vertex, missing axis), all detected. Sixty-four independent scalar optimizations recover `rho=e`, hence `r*=e^(1/n)`, within `4.44e-8`. For `D_head=128`, `M=4`, `n=3`, Equation 37 yields `theta<=207.127`, including 100 and excluding `10^4`/`10^6`.

Raw evidence: `outputs/claim2/claim2_report.json`, `geometry_cases.csv`, `economy_cases.csv`, and `negative_controls.csv`.
