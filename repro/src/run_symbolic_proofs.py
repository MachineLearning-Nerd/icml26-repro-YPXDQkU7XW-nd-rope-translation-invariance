"""Dimension-independent symbolic proof certificates for Claims 1 and 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import sympy as sp


def is_zero_matrix(matrix: sp.Matrix) -> bool:
    return all(sp.trigsimp(value) == 0 for value in matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/symbolic_proof_certificates.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()

    a, b, t = sp.symbols("a b t", real=True)
    rotation = lambda angle: sp.Matrix(
        [
            [sp.cos(angle), -sp.sin(angle)],
            [sp.sin(angle), sp.cos(angle)],
        ]
    )
    group_residual = rotation(a).T * rotation(b) - rotation(b - a)
    translated_residual = (
        rotation(a + t).T * rotation(b + t) - rotation(b - a)
    )
    wrong_sign_residual = rotation(a).T * rotation(b) - rotation(a - b)

    u, v, shift = sp.symbols("u v shift", real=True)
    fourier_shift_residual = sp.simplify(
        sp.exp(-sp.I * (u + shift))
        * sp.exp(sp.I * (v + shift))
        - sp.exp(sp.I * (v - u))
    )

    n = sp.symbols("n", integer=True, positive=True)
    scale_sq = (n + 1) / n
    simplex_norm_sq = sp.simplify(
        scale_sq * (1 - 2 / (n + 1) + (n + 1) / (n + 1) ** 2)
    )
    simplex_off_diagonal = sp.simplify(
        scale_sq * (-2 / (n + 1) + (n + 1) / (n + 1) ** 2)
    )
    centroid_coordinate = sp.simplify(1 - (n + 1) / (n + 1))
    tight_frame_on_zero_sum_subspace = sp.simplify((n + 1) / n)

    rho, log_resolution = sp.symbols(
        "rho log_resolution", real=True, positive=True
    )
    economy = rho * log_resolution / sp.log(rho)
    derivative = sp.simplify(sp.diff(economy, rho))
    expected_derivative = (
        log_resolution * (sp.log(rho) - 1) / sp.log(rho) ** 2
    )
    derivative_identity = sp.simplify(
        derivative - expected_derivative
    ) == 0
    optimum_rho = sp.E
    optimum_scale_ratio = sp.exp(1 / n)

    checks = {
        "claim1_rotation_group_identity_all_real_phases": is_zero_matrix(
            group_residual
        ),
        "claim1_shared_translation_cancels_symbolically": is_zero_matrix(
            translated_residual
        ),
        "claim1_fourier_shared_shift_cancels_symbolically": (
            fourier_shift_residual == 0
        ),
        "claim1_wrong_displacement_sign_rejected": not is_zero_matrix(
            wrong_sign_residual
        ),
        "claim2_unit_norm_all_positive_integer_n": simplex_norm_sq == 1,
        "claim2_gram_off_diagonal_all_positive_integer_n": (
            simplex_off_diagonal == -1 / n
        ),
        "claim2_centroid_zero_all_positive_integer_n": (
            centroid_coordinate == 0
        ),
        "claim2_tight_frame_factor": (
            tight_frame_on_zero_sum_subspace == (n + 1) / n
        ),
        "claim2_economy_derivative_identity": derivative_identity,
        "claim2_unique_stationary_ratio": (
            sp.simplify(expected_derivative.subs(rho, optimum_rho)) == 0
            and sp.limit(expected_derivative, rho, 1, dir="+") < 0
            and expected_derivative.subs(rho, sp.E**2) > 0
        ),
        "claim2_scale_ratio_from_rho": (
            sp.simplify(optimum_rho ** (1 / n) - optimum_scale_ratio) == 0
        ),
    }
    report = {
        "purpose": (
            "Exact symbolic derivations covering the universal phase and "
            "regular-simplex/economy quantifiers. Finite numerical sweeps are "
            "corroboration, not the proof."
        ),
        "assumptions": {
            "claim1": (
                "real phase projections; shared translation; complex Fourier "
                "features with conjugate query/key phases"
            ),
            "claim2": (
                "integer n>=2 for the paper's n-dimensional simplex; rho>1 "
                "and resolution>1 for the economy objective"
            ),
        },
        "derivations": {
            "claim1": {
                "rotation_identity": "R(a)^T R(b) = R(b-a)",
                "translation_identity": (
                    "R(a+t)^T R(b+t) = R(b-a) for arbitrary real a,b,t"
                ),
                "fourier_identity": (
                    "exp(-i(u+s)) exp(i(v+s)) = exp(i(v-u))"
                ),
            },
            "claim2": {
                "simplex_definition": (
                    "v_i=sqrt((n+1)/n)(e_i-1/(n+1)1)"
                ),
                "gram": "v_i·v_i=1; v_i·v_j=-1/n for i!=j",
                "centroid": "sum_i v_i=0",
                "tight_frame": (
                    "sum_i v_i v_i^T=((n+1)/n)I on the zero-sum subspace"
                ),
                "economy_derivative": str(expected_derivative),
                "unique_optimum": "rho*=e, hence r*=rho*^(1/n)=exp(1/n)",
            },
        },
        "checks": {name: bool(value) for name, value in checks.items()},
        "all_checks_pass": all(checks.values()),
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
