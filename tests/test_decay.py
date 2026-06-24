"""
Tests for pracpy.single_decay and pracpy.chain_decay
Run with:  pytest -v
"""

import math
import pytest
import numpy as np
import pracpy

# ---------------------------------------------------------------------------
# Tests for single_decay
# ---------------------------------------------------------------------------
class TestSingleDecay:
    def test_returns_dict_with_required_keys(self):
        res = pracpy.single_decay(10000, 0.3, 10)
        for key in ("t", "N", "decay", "lambda", "t_half"):
            assert key in res, f"Missing key: {key}"

    def test_array_lengths_default(self):
        res = pracpy.single_decay(10000, 0.3, 10)
        assert len(res["t"]) == 1000
        assert len(res["N"]) == 1000
        assert len(res["decay"]) == 1000

    def test_array_lengths_custom_points(self):
        res = pracpy.single_decay(10000, 0.3, 10, points=500)
        assert len(res["t"]) == 500

    def test_t_starts_at_zero_ends_at_t_end(self):
        res = pracpy.single_decay(10000, 0.3, 10)
        assert res["t"][0] == pytest.approx(0.0)
        assert res["t"][-1] == pytest.approx(10.0)

    def test_N_starts_at_N0(self):
        N0 = 10000
        res = pracpy.single_decay(N0, 0.3, 10)
        assert res["N"][0] == pytest.approx(N0)

    def test_N_is_monotonically_decreasing(self):
        res = pracpy.single_decay(10000, 0.3, 10)
        N = res["N"]
        assert all(N[i] >= N[i + 1] for i in range(len(N) - 1))

    def test_decay_plus_N_equals_N0(self):
        N0 = 10000
        res = pracpy.single_decay(N0, 0.3, 10)
        assert np.allclose(res["N"] + res["decay"], N0)

    def test_half_life_formula(self):
        lam = 0.3
        res = pracpy.single_decay(10000, lam, 10)
        expected_t_half = math.log(2) / lam
        assert res["t_half"] == pytest.approx(expected_t_half, rel=1e-9)

    # --- error handling ---
    def test_negative_N0_raises(self):
        with pytest.raises(ValueError):
            pracpy.single_decay(-1, 0.3, 10)

    def test_zero_N0_raises(self):
        with pytest.raises(ValueError):
            pracpy.single_decay(0, 0.3, 10)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            pracpy.single_decay(10000, -0.3, 10)


# ---------------------------------------------------------------------------
# Tests for chain_decay
# ---------------------------------------------------------------------------
class TestChainDecay:
    def test_returns_required_keys(self):
        res = pracpy.chain_decay(N0=10000, lambdas=[0.5, 0.2], t_end=10)
        assert "t" in res
        assert "N" in res

    def test_matrix_shape(self):
        lambdas = [0.8, 0.3, 0.1]
        points = 500
        res = pracpy.chain_decay(N0=10000, lambdas=lambdas, t_end=10, points=points)

        # N should be a 2D array: (number of isotopes) x (number of time points)
        assert isinstance(res["N"], np.ndarray)
        assert res["N"].shape == (len(lambdas), points)

    def test_parent_starts_at_N0_others_at_zero(self):
        N0 = 10000
        res = pracpy.chain_decay(N0, [0.8, 0.3], 10)

        # Parent isotope at t=0 should be N0
        assert res["N"][0, 0] == pytest.approx(N0)

        # Daughter isotope at t=0 should be 0
        assert res["N"][1, 0] == pytest.approx(0.0)

    def test_identical_lambdas_raises_error(self):
        # The analytical Bateman solver breaks if decay constants are identical
        with pytest.raises(ValueError, match="Degenerate lambdas"):
            pracpy.chain_decay(10000, [0.5, 0.5], 10)

    def test_empty_lambda_list_raises_error(self):
        with pytest.raises(ValueError):
            pracpy.chain_decay(10000, [], 10)
