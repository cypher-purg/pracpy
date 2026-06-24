"""
Tests for pracpy.ra_decay
Run with:  pytest -v
"""

import math
import pytest
import numpy as np
import pracpy


class TestExpone:
    def test_returns_dict_with_required_keys(self):
        res = pracpy.ra_decay(10000, 0.3, 10)
        for key in ("t", "N", "decay", "lambda", "t_half"):
            assert key in res, f"Missing key: {key}"

    def test_array_lengths_default(self):
        res = pracpy.ra_decay(10000, 0.3, 10)
        assert len(res["t"]) == 1000
        assert len(res["N"]) == 1000
        assert len(res["decay"]) == 1000

    def test_array_lengths_custom_points(self):
        res = pracpy.ra_decay(10000, 0.3, 10, points=500)
        assert len(res["t"]) == 500

    def test_t_starts_at_zero_ends_at_t_end(self):
        res = pracpy.ra_decay(10000, 0.3, 10)
        assert res["t"][0] == pytest.approx(0.0)
        assert res["t"][-1] == pytest.approx(10.0)

    def test_N_starts_at_N0(self):
        N0 = 10000
        res = pracpy.ra_decay(N0, 0.3, 10)
        assert res["N"][0] == pytest.approx(N0)

    def test_N_is_monotonically_decreasing(self):
        res = pracpy.ra_decay(10000, 0.3, 10)
        N = res["N"]
        assert all(N[i] >= N[i + 1] for i in range(len(N) - 1))

    def test_decay_plus_N_equals_N0(self):
        N0 = 10000
        res = pracpy.ra_decay(N0, 0.3, 10)
        assert np.allclose(res["N"] + res["decay"], N0)

    def test_half_life_formula(self):
        lam = 0.3
        res = pracpy.ra_decay(10000, lam, 10)
        expected_t_half = math.log(2) / lam
        assert res["t_half"] == pytest.approx(expected_t_half, rel=1e-9)

    def test_N_at_half_life_is_half_N0(self):
        N0 = 10000
        lam = 0.3
        res = pracpy.ra_decay(N0, lam, 10, points=100_000)
        t_half = res["t_half"]
        # find the index closest to t_half
        idx = np.argmin(np.abs(res["t"] - t_half))
        assert res["N"][idx] == pytest.approx(N0 / 2, rel=1e-3)

    def test_lambda_stored_in_result(self):
        lam = 0.3
        res = pracpy.ra_decay(10000, lam, 10)
        assert res["lambda"] == lam

    # --- error handling ---

    def test_negative_N0_raises(self):
        with pytest.raises(ValueError):
            pracpy.ra_decay(-1, 0.3, 10)

    def test_zero_N0_raises(self):
        with pytest.raises(ValueError):
            pracpy.ra_decay(0, 0.3, 10)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            pracpy.ra_decay(10000, -0.3, 10)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError):
            pracpy.ra_decay(10000, 0.3, -5)

    def test_points_less_than_2_raises(self):
        with pytest.raises(ValueError):
            pracpy.ra_decay(10000, 0.3, 10, points=1)
