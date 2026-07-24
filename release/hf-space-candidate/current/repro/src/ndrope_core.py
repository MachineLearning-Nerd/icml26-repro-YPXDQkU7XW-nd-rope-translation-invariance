"""Independent NumPy implementation of the mathematical core of nD-RoPE.

This module intentionally does not import the authors' implementation.  The
official PyTorch functions are loaded separately by ``official_extract.py`` so
that implementation agreement is a genuine cross-check.
"""

from __future__ import annotations

import math

import numpy as np


def regular_simplex(dimension: int) -> np.ndarray:
    """Return unit regular-simplex rows in R^dimension.

    For d >= 2 this constructs the centered projector in R^(d+1), finds an
    orthonormal basis for its rank-d image, and normalizes each projected row.
    The d=1 convention follows the paper's real-valued RoPE reduction and keeps
    one positive frequency rather than two conjugate directions.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if dimension == 1:
        return np.ones((1, 1), dtype=np.float64)
    count = dimension + 1
    centered = np.eye(count) - np.ones((count, count)) / count
    values, vectors = np.linalg.eigh(centered)
    basis = vectors[:, values > 0.5]
    simplex = centered @ basis
    return simplex / np.linalg.norm(simplex, axis=1, keepdims=True)


def frequency_scales(scale_count: int, theta: float) -> np.ndarray:
    if scale_count < 1:
        raise ValueError("scale_count must be positive")
    if theta <= 1:
        raise ValueError("theta must exceed one")
    return theta ** (-np.arange(scale_count, dtype=np.float64) / scale_count)


def phase_bank(
    positions: np.ndarray,
    wave_vectors: np.ndarray,
    head_dim: int,
    theta: float,
) -> np.ndarray:
    """Return exp(i omega^T x) phases in paper/official M-major order."""

    positions = np.asarray(positions, dtype=np.float64)
    wave_vectors = np.asarray(wave_vectors, dtype=np.float64)
    if positions.ndim != 2 or wave_vectors.ndim != 2:
        raise ValueError("positions and wave_vectors must be matrices")
    if positions.shape[1] != wave_vectors.shape[1]:
        raise ValueError("position and wave-vector dimensions differ")
    direction_count = wave_vectors.shape[0]
    per_scale = 2 * direction_count
    if head_dim % per_scale:
        raise ValueError("head_dim must be divisible by 2 times direction_count")
    scales = frequency_scales(head_dim // per_scale, theta)
    projections = positions @ wave_vectors.T
    angles = (projections[..., None] * scales).reshape(positions.shape[0], -1)
    return np.exp(1j * angles)


def real_pairs_to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] % 2:
        raise ValueError("last dimension must be even")
    paired = values.reshape(*values.shape[:-1], -1, 2)
    return paired[..., 0] + 1j * paired[..., 1]


def complex_to_real_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.stack((values.real, values.imag), axis=-1).reshape(
        *values.shape[:-1], 2 * values.shape[-1]
    )


def apply_rotary(values: np.ndarray, phases: np.ndarray) -> np.ndarray:
    return complex_to_real_pairs(real_pairs_to_complex(values) * phases)


def rotary_score(
    query: np.ndarray,
    key: np.ndarray,
    query_position: np.ndarray,
    key_position: np.ndarray,
    wave_vectors: np.ndarray,
    theta: float,
) -> float:
    head_dim = query.shape[-1]
    phases = phase_bank(
        np.stack((query_position, key_position)), wave_vectors, head_dim, theta
    )
    return float(np.dot(apply_rotary(query, phases[0]), apply_rotary(key, phases[1])))


def relative_rotary_score(
    query: np.ndarray,
    key: np.ndarray,
    displacement: np.ndarray,
    wave_vectors: np.ndarray,
    theta: float,
) -> float:
    head_dim = query.shape[-1]
    relative_phase = phase_bank(
        np.asarray(displacement, dtype=np.float64)[None, :],
        wave_vectors,
        head_dim,
        theta,
    )[0]
    q_complex = real_pairs_to_complex(query)
    k_complex = real_pairs_to_complex(key)
    return float(np.real(np.sum(np.conj(q_complex) * k_complex * relative_phase)))


def additive_position(position: np.ndarray, head_dim: int) -> np.ndarray:
    """Absolute additive sinusoid used only as a negative control."""

    position = np.asarray(position, dtype=np.float64)
    slots = head_dim // 2
    frequencies = np.exp(np.linspace(0.0, -4.0, slots))
    scalar = np.resize(position, slots)
    angles = scalar * frequencies
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1).reshape(-1)


def economy_cost(rho: float, resolution: float = 1e6) -> float:
    """Equation 33 up to the positive constant d: rho * log_rho(R)."""

    if rho <= 1 or resolution <= 1:
        return math.inf
    return rho * math.log(resolution) / math.log(rho)


def optimal_scale_ratio(dimension: int) -> float:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return math.exp(1.0 / dimension)


def base_upper_bound(head_dim: int, vectors_per_scale: int, dimension: int) -> float:
    return math.exp(head_dim / (2.0 * vectors_per_scale * dimension))
