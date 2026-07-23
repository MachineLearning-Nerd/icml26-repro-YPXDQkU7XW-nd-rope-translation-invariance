from __future__ import annotations

import math

import numpy as np
import pytest

from ndrope_core import (
    apply_rotary,
    base_upper_bound,
    economy_cost,
    optimal_scale_ratio,
    phase_bank,
    regular_simplex,
    relative_rotary_score,
    rotary_score,
)


@pytest.mark.parametrize("dimension", [1, 2, 3, 5, 8, 16])
def test_simplex_shape_rank_and_norm(dimension: int) -> None:
    simplex = regular_simplex(dimension)
    expected_rows = 1 if dimension == 1 else dimension + 1
    assert simplex.shape == (expected_rows, dimension)
    assert np.linalg.matrix_rank(simplex) == dimension
    np.testing.assert_allclose(np.linalg.norm(simplex, axis=1), 1, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3, 5, 8, 16])
def test_simplex_equations_18_and_19(dimension: int) -> None:
    simplex = regular_simplex(dimension)
    gram = simplex @ simplex.T
    expected = np.eye(dimension + 1) * (1 + 1 / dimension) - np.ones_like(gram) / dimension
    np.testing.assert_allclose(gram, expected, atol=1e-12)
    np.testing.assert_allclose(simplex.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        simplex.T @ simplex, np.eye(dimension) * (dimension + 1) / dimension, atol=1e-12
    )


def test_phase_bank_rejects_incompatible_head_dim() -> None:
    with pytest.raises(ValueError):
        phase_bank(np.zeros((1, 2)), regular_simplex(2), 64, 100)


def test_rotation_preserves_pair_norms() -> None:
    rng = np.random.default_rng(1)
    simplex = regular_simplex(3)
    values = rng.normal(size=(11, 64))
    phases = phase_bank(rng.normal(size=(11, 3)), simplex, 64, 100)
    rotated = apply_rotary(values, phases)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(values, axis=1), atol=1e-12)


def test_relative_score_identity_and_translation() -> None:
    rng = np.random.default_rng(2)
    simplex = regular_simplex(4)
    q = rng.normal(size=100)
    k = rng.normal(size=100)
    x = rng.normal(size=4)
    y = rng.normal(size=4)
    shift = rng.normal(size=4)
    direct = rotary_score(q, k, x, y, simplex, 100)
    translated = rotary_score(q, k, x + shift, y + shift, simplex, 100)
    relative = relative_rotary_score(q, k, y - x, simplex, 100)
    assert direct == pytest.approx(translated, abs=1e-11)
    assert direct == pytest.approx(relative, abs=1e-11)


@pytest.mark.parametrize("dimension", [1, 2, 3, 8, 16])
def test_optimal_ratio(dimension: int) -> None:
    rho = optimal_scale_ratio(dimension) ** dimension
    assert rho == pytest.approx(math.e, rel=1e-12)
    optimum = economy_cost(rho)
    assert economy_cost(math.exp(0.5)) > optimum
    assert economy_cost(math.exp(2.0)) > optimum


def test_table7_base_bound() -> None:
    bound = base_upper_bound(128, 4, 3)
    assert bound == pytest.approx(math.exp(128 / 24))
    assert 100 < bound < 1e4
