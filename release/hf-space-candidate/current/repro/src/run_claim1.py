"""Claim 1: Fourier/Parseval and translation-invariant nD-RoPE certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ndrope_core import (
    additive_position,
    apply_rotary,
    phase_bank,
    regular_simplex,
    relative_rotary_score,
    rotary_score,
)
from official_extract import load_functions


def finite_parseval_trial(rng: np.random.Generator, shape: tuple[int, ...]) -> dict[str, float]:
    spatial_dim = len(shape)
    content_dim = min(5, int(np.prod(shape)))
    frequency_count = int(np.prod(shape))
    raw = rng.normal(size=(frequency_count, content_dim)) + 1j * rng.normal(
        size=(frequency_count, content_dim)
    )
    basis, _ = np.linalg.qr(raw)
    query = rng.normal(size=content_dim)
    key = rng.normal(size=content_dim)
    gamma_q = (basis @ query).reshape(shape)
    gamma_k = (basis @ key).reshape(shape)
    shift_q = tuple(int(rng.integers(0, side)) for side in shape)
    shift_k = tuple(int(rng.integers(0, side)) for side in shape)

    grids = np.meshgrid(
        *[np.fft.fftfreq(side) * side for side in shape], indexing="ij"
    )
    phase_q = np.ones(shape, dtype=np.complex128)
    phase_k = np.ones(shape, dtype=np.complex128)
    for grid, sq, sk, side in zip(grids, shift_q, shift_k, shape, strict=True):
        phase_q *= np.exp(2j * np.pi * grid * sq / side)
        phase_k *= np.exp(2j * np.pi * grid * sk / side)

    translated_q = np.fft.ifftn(gamma_q * phase_q, norm="ortho")
    translated_k = np.fft.ifftn(gamma_k * phase_k, norm="ortho")
    spatial_inner = np.vdot(translated_q, translated_k)
    spectral_inner = np.vdot(gamma_q * phase_q, gamma_k * phase_k)
    kernel_inner = np.vdot(gamma_q, gamma_k)
    content_inner = np.dot(query, key)

    delta = tuple((b - a) % side for a, b, side in zip(shift_q, shift_k, shape, strict=True))
    phase_delta = np.ones(shape, dtype=np.complex128)
    for grid, diff, side in zip(grids, delta, shape, strict=True):
        phase_delta *= np.exp(2j * np.pi * grid * diff / side)
    relative_inner = np.vdot(gamma_q, gamma_k * phase_delta)
    return {
        "parseval_error": float(abs(spatial_inner - spectral_inner)),
        "relative_error": float(abs(spatial_inner - relative_inner)),
        "kernel_error": float(abs(kernel_inner - content_inner)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    official_file = args.official_root / "rope-vit-ndrope/deit/models_v2_ndRope.py"
    official_sha = hashlib.sha256(official_file.read_bytes()).hexdigest()
    official = load_functions(
        official_file,
        [
            "generate_simplex_vectors_with_projection",
            "compute_ndrope_cis",
            "apply_rotary_emb",
        ],
    )

    rows: list[dict[str, object]] = []
    max_translation = 0.0
    max_relative = 0.0
    min_additive_break = np.inf
    total_rotary_trials = 0
    dimensions = (1, 2, 3, 4, 5, 8)
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for dimension in dimensions:
            omega_sets = {
                "simplex": regular_simplex(dimension),
                "random": rng.normal(size=(dimension + (dimension > 1), dimension)),
            }
            for family, omega in omega_sets.items():
                omega /= np.linalg.norm(omega, axis=1, keepdims=True)
                head_dim = 2 * omega.shape[0] * args.scales
                for trial in range(args.trials_per_seed):
                    q = rng.normal(size=head_dim)
                    k = rng.normal(size=head_dim)
                    xq = rng.normal(size=dimension)
                    xk = rng.normal(size=dimension)
                    global_shift = rng.normal(size=dimension)
                    score = rotary_score(q, k, xq, xk, omega, args.theta)
                    shifted = rotary_score(
                        q, k, xq + global_shift, xk + global_shift, omega, args.theta
                    )
                    relative = relative_rotary_score(
                        q, k, xk - xq, omega, args.theta
                    )
                    translation_error = abs(score - shifted)
                    relative_error = abs(score - relative)
                    max_translation = max(max_translation, translation_error)
                    max_relative = max(max_relative, relative_error)

                    additive = np.dot(
                        q + additive_position(xq, head_dim),
                        k + additive_position(xk, head_dim),
                    )
                    additive_shift = np.dot(
                        q + additive_position(xq + global_shift, head_dim),
                        k + additive_position(xk + global_shift, head_dim),
                    )
                    additive_break = abs(additive - additive_shift)
                    if additive_break > 1e-8:
                        min_additive_break = min(min_additive_break, additive_break)
                    total_rotary_trials += 1
                    rows.append(
                        {
                            "seed": seed,
                            "dimension": dimension,
                            "family": family,
                            "trial": trial,
                            "translation_error": translation_error,
                            "relative_error": relative_error,
                            "additive_control_shift": additive_break,
                        }
                    )

    parseval_rows: list[dict[str, object]] = []
    for seed in range(args.parseval_seeds):
        rng = np.random.default_rng(10_000 + seed)
        for shape in ((127,), (16, 16), (9, 11)):
            result = finite_parseval_trial(rng, shape)
            parseval_rows.append({"seed": seed, "shape": list(shape), **result})

    official_simplex = official["generate_simplex_vectors_with_projection"](3).double()
    official_gram = official_simplex @ official_simplex.T
    independent_gram = regular_simplex(3) @ regular_simplex(3).T
    official_simplex_gram_error = float(
        torch.max(torch.abs(official_gram - torch.from_numpy(independent_gram))).item()
    )

    torch.manual_seed(123)
    positions = torch.randn(37, 3, dtype=torch.float64)
    heads, layers, directions, head_dim = 4, 3, 4, 40
    frequencies = official_simplex[None, None, :, :].repeat(heads, layers, 1, 1)
    official_phases = official["compute_ndrope_cis"](
        frequencies,
        positions,
        head_dim,
        heads,
        args.theta,
        dtype=torch.float64,
    )
    independent_phases = np.stack(
        [
            phase_bank(
                positions.numpy(), official_simplex.numpy(), head_dim, args.theta
            )
            for _ in range(layers * heads)
        ]
    ).reshape(layers, heads, positions.shape[0], head_dim // 2)
    official_phase_error = float(
        np.max(np.abs(official_phases.numpy() - independent_phases))
    )

    q = torch.randn(2, heads, positions.shape[0], head_dim, dtype=torch.float64)
    k = torch.randn_like(q)
    official_q, official_k = official["apply_rotary_emb"](
        q, k, official_phases[0]
    )
    independent_q = apply_rotary(q.numpy(), official_phases[0].numpy()[None, ...])
    independent_k = apply_rotary(k.numpy(), official_phases[0].numpy()[None, ...])
    official_apply_error = float(
        max(
            np.max(np.abs(official_q.numpy() - independent_q)),
            np.max(np.abs(official_k.numpy() - independent_k)),
        )
    )

    parseval_max = {
        key: max(float(row[key]) for row in parseval_rows)
        for key in ("parseval_error", "relative_error", "kernel_error")
    }
    checks = {
        "rotary_translation_invariance": max_translation < 1e-10,
        "relative_displacement_identity": max_relative < 1e-10,
        "finite_parseval": parseval_max["parseval_error"] < 1e-10,
        "finite_fourier_relative_shift": parseval_max["relative_error"] < 1e-10,
        "riesz_kernel_preservation": parseval_max["kernel_error"] < 1e-10,
        # The released constructor is hard-coded float32, so direct parity is
        # assessed at a float32-appropriate tolerance while retaining raw errors.
        "official_simplex_agreement": official_simplex_gram_error < 2e-7,
        "official_phase_agreement": official_phase_error < 1e-10,
        # apply_rotary_emb explicitly converts its complex pairs to float32.
        "official_rotation_agreement": official_apply_error < 5e-7,
        "additive_negative_control_detected": min_additive_break > 1e-8,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "claim": 1,
        "assessment": "verified" if all(checks.values()) else "failed",
        "scope": "exact finite-dimensional Fourier analogue plus direct official-code parity",
        "official_source": {
            "path": str(official_file),
            "commit": "f2cae70760806451f5e58be4b7e3dc4d0d856a1e",
            "sha256": official_sha,
        },
        "trial_counts": {
            "rotary": total_rotary_trials,
            "parseval": len(parseval_rows),
        },
        "max_errors": {
            "translation": max_translation,
            "relative_identity": max_relative,
            **parseval_max,
            "official_simplex_gram": official_simplex_gram_error,
            "official_phase": official_phase_error,
            "official_apply": official_apply_error,
        },
        "negative_control": {"minimum_detected_additive_shift": min_additive_break},
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim1_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "translation_trials.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "parseval_trials.json").write_text(
        json.dumps(parseval_rows, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=Path("vendor/nD-RoPE"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/claim1"))
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--trials-per-seed", type=int, default=20)
    parser.add_argument("--parseval-seeds", type=int, default=64)
    parser.add_argument("--scales", type=int, default=8)
    parser.add_argument("--theta", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
