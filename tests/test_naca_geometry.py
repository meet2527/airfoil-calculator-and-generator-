"""
Tests for airfoil_config.naca_geometry — NACA coordinate generator.

Covers:
    - Cosine spacing (endpoints, monotonicity, clustering)
    - NACA 4-digit symmetric (0012) and cambered (2412, 4415)
    - NACA 5-digit standard (23012) and reflex (23112)
    - String parsing (naca_from_string)
    - Single-surface conversion
    - Max thickness location
    - Validation / edge-case errors
"""

from __future__ import annotations

import numpy as np
import pytest

from airfoil_config.naca_geometry import (
    AirfoilCoordinates,
    AirfoilSingleSurface,
    cosine_spacing,
    max_thickness_at,
    naca4,
    naca5,
    naca_from_string,
    to_single_surface,
)

REL_TOL = 1e-4
ABS_TOL = 1e-6


# ===================================================================
# Cosine spacing
# ===================================================================
class TestCosineSpacing:
    """Tests for the cosine point distribution."""

    def test_endpoints(self) -> None:
        x = cosine_spacing(50)
        assert x[0] == pytest.approx(0.0, abs=ABS_TOL)
        assert x[-1] == pytest.approx(1.0, abs=ABS_TOL)

    def test_length(self) -> None:
        assert len(cosine_spacing(100)) == 100

    def test_monotonic(self) -> None:
        x = cosine_spacing(50)
        assert np.all(np.diff(x) > 0)

    def test_clustering_at_le(self) -> None:
        """Points denser near x = 0 than at mid-chord."""
        dx = np.diff(cosine_spacing(100))
        assert dx[0] < dx[len(dx) // 2]

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="n_points"):
            cosine_spacing(1)


# ===================================================================
# NACA 4-digit — symmetric (0012)
# ===================================================================
class TestNaca4Symmetric:
    """NACA 0012 — zero camber."""

    def test_designation(self) -> None:
        assert naca4(0, 0, 12).designation == "NACA 0012"

    def test_symmetry(self) -> None:
        c = naca4(0, 0, 12)
        np.testing.assert_allclose(c.y_upper, -c.y_lower, atol=ABS_TOL)

    def test_x_match(self) -> None:
        c = naca4(0, 0, 12)
        np.testing.assert_allclose(c.x_upper, c.x_lower, atol=ABS_TOL)

    def test_zero_camber(self) -> None:
        c = naca4(0, 0, 12)
        np.testing.assert_allclose(c.y_camber, 0.0, atol=ABS_TOL)

    def test_le_origin(self) -> None:
        c = naca4(0, 0, 12)
        assert c.x_upper[0] == pytest.approx(0.0, abs=ABS_TOL)
        assert c.y_upper[0] == pytest.approx(0.0, abs=ABS_TOL)

    def test_closed_te(self) -> None:
        c = naca4(0, 0, 12, closed_te=True)
        assert abs(c.y_upper[-1]) < 1e-4
        assert abs(c.y_lower[-1]) < 1e-4

    def test_open_te_has_gap(self) -> None:
        c = naca4(0, 0, 12, closed_te=False)
        assert c.y_upper[-1] - c.y_lower[-1] > 0.001

    def test_max_thickness_value(self) -> None:
        c = naca4(0, 0, 12, n_points=200)
        _, t_max = max_thickness_at(c)
        assert t_max == pytest.approx(0.12, rel=0.02)

    def test_max_thickness_location(self) -> None:
        c = naca4(0, 0, 12, n_points=200)
        x_t, _ = max_thickness_at(c)
        assert 0.25 < x_t < 0.35

    def test_shape(self) -> None:
        c = naca4(0, 0, 12, n_points=50)
        for arr in (c.x_upper, c.y_upper, c.x_lower, c.y_lower):
            assert arr.shape == (50,)


# ===================================================================
# NACA 4-digit — cambered (2412)
# ===================================================================
class TestNaca4Cambered:
    """NACA 2412 — 2 % camber at 40 % chord."""

    def test_designation(self) -> None:
        assert naca4(2, 4, 12).designation == "NACA 2412"

    def test_camber_positive(self) -> None:
        c = naca4(2, 4, 12, n_points=200)
        assert np.all(c.y_camber[1:-1] > 0)

    def test_camber_peak_position(self) -> None:
        c = naca4(2, 4, 12, n_points=200)
        idx = int(np.argmax(c.y_camber))
        assert c.x_camber[idx] == pytest.approx(0.4, abs=0.02)

    def test_camber_peak_value(self) -> None:
        c = naca4(2, 4, 12, n_points=200)
        assert np.max(c.y_camber) == pytest.approx(0.02, rel=0.05)

    def test_upper_above_lower(self) -> None:
        c = naca4(2, 4, 12, n_points=200)
        yl = np.interp(c.x_upper, c.x_lower, c.y_lower)
        assert np.all(c.y_upper[1:-1] > yl[1:-1])

    def test_not_symmetric(self) -> None:
        c = naca4(2, 4, 12)
        assert not np.allclose(c.y_upper, -c.y_lower)

    def test_thicker_airfoil(self) -> None:
        """NACA 4415 should have max thickness ≈ 0.15."""
        c = naca4(4, 4, 15, n_points=200)
        _, t = max_thickness_at(c)
        assert t == pytest.approx(0.15, rel=0.03)


# ===================================================================
# NACA 4-digit — validation
# ===================================================================
class TestNaca4Validation:
    """Parameter validation for naca4."""

    def test_camber_no_position_raises(self) -> None:
        with pytest.raises(ValueError, match="p_pct > 0"):
            naca4(5, 0, 12)

    def test_position_no_camber_raises(self) -> None:
        with pytest.raises(ValueError, match="p_pct=0"):
            naca4(0, 5, 12)

    def test_zero_thickness_raises(self) -> None:
        with pytest.raises(ValueError, match="t_pct"):
            naca4(0, 0, 0)

    def test_negative_m_raises(self) -> None:
        with pytest.raises(ValueError, match="m_pct"):
            naca4(-1, 0, 12)

    def test_float_m_raises(self) -> None:
        with pytest.raises(ValueError, match="m_pct"):
            naca4(2.5, 4, 12)  # type: ignore[arg-type]

    def test_n_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n_points"):
            naca4(0, 0, 12, n_points=1)


# ===================================================================
# NACA 5-digit — standard (23012)
# ===================================================================
class TestNaca5Standard:
    """NACA 23012 — standard 5-digit."""

    def test_designation(self) -> None:
        assert naca5(2, 3, 0, 12).designation == "NACA 23012"

    def test_camber_positive(self) -> None:
        c = naca5(2, 3, 0, 12, n_points=200)
        assert np.max(c.y_camber) > 0

    def test_max_thickness(self) -> None:
        c = naca5(2, 3, 0, 12, n_points=200)
        _, t = max_thickness_at(c)
        assert t == pytest.approx(0.12, rel=0.05)

    def test_shape(self) -> None:
        c = naca5(2, 3, 0, 12, n_points=80)
        assert c.x_upper.shape == (80,)

    def test_camber_zero_at_te(self) -> None:
        c = naca5(2, 3, 0, 12, n_points=200)
        assert abs(c.y_camber[-1]) < 1e-6


# ===================================================================
# NACA 5-digit — reflex (23112)
# ===================================================================
class TestNaca5Reflex:
    """NACA 23112 — reflex camber."""

    def test_designation(self) -> None:
        assert naca5(2, 3, 1, 12).designation == "NACA 23112"

    def test_different_from_standard(self) -> None:
        """Reflex camber line should differ from standard."""
        std = naca5(2, 3, 0, 12, n_points=100)
        rfx = naca5(2, 3, 1, 12, n_points=100)
        assert not np.allclose(std.y_camber, rfx.y_camber)

    def test_generates_valid_shape(self) -> None:
        c = naca5(2, 3, 1, 12, n_points=100)
        assert c.x_upper.shape == (100,)
        assert np.all(np.isfinite(c.y_upper))


# ===================================================================
# NACA 5-digit — validation
# ===================================================================
class TestNaca5Validation:
    """Parameter validation for naca5."""

    def test_p_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="p_digit"):
            naca5(2, 0, 0, 12)

    def test_reflex_p1_raises(self) -> None:
        with pytest.raises(ValueError, match="Reflex"):
            naca5(2, 1, 1, 12)

    def test_l_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="l_digit"):
            naca5(0, 3, 0, 12)

    def test_bad_s_raises(self) -> None:
        with pytest.raises(ValueError, match="s_digit"):
            naca5(2, 3, 2, 12)


# ===================================================================
# String parsing
# ===================================================================
class TestNacaFromString:
    """Tests for designation string parsing."""

    def test_four_digit(self) -> None:
        assert naca_from_string("NACA 2412").designation == "NACA 2412"

    def test_five_digit(self) -> None:
        assert naca_from_string("NACA 23012").designation == "NACA 23012"

    def test_no_prefix(self) -> None:
        assert naca_from_string("0012").designation == "NACA 0012"

    def test_case_insensitive(self) -> None:
        assert naca_from_string("naca2412").designation == "NACA 2412"

    def test_with_spaces(self) -> None:
        assert naca_from_string("  NACA  2412  ").designation == "NACA 2412"

    def test_three_digits_raises(self) -> None:
        with pytest.raises(ValueError):
            naca_from_string("NACA 123")

    def test_letters_raises(self) -> None:
        with pytest.raises(ValueError):
            naca_from_string("NACA abcd")


# ===================================================================
# Single-surface conversion
# ===================================================================
class TestToSingleSurface:
    """Tests for the Selig-format loop."""

    def test_length(self) -> None:
        c = naca4(0, 0, 12, n_points=50)
        s = to_single_surface(c)
        assert len(s.x) == 2 * 50 - 1

    def test_starts_near_te(self) -> None:
        s = to_single_surface(naca4(0, 0, 12))
        assert s.x[0] == pytest.approx(1.0, abs=0.01)

    def test_ends_near_te(self) -> None:
        s = to_single_surface(naca4(0, 0, 12))
        assert s.x[-1] == pytest.approx(1.0, abs=0.01)

    def test_le_in_middle(self) -> None:
        n = 100
        s = to_single_surface(naca4(0, 0, 12, n_points=n))
        assert s.x[n - 1] == pytest.approx(0.0, abs=ABS_TOL)

    def test_designation_kept(self) -> None:
        s = to_single_surface(naca4(2, 4, 12))
        assert s.designation == "NACA 2412"

    def test_isinstance(self) -> None:
        s = to_single_surface(naca4(0, 0, 12))
        assert isinstance(s, AirfoilSingleSurface)


# ===================================================================
# Max thickness helper
# ===================================================================
class TestMaxThickness:
    """Tests for max_thickness_at."""

    def test_naca0012(self) -> None:
        c = naca4(0, 0, 12, n_points=200)
        x_t, t = max_thickness_at(c)
        assert t == pytest.approx(0.12, rel=0.02)
        assert 0.25 < x_t < 0.35

    def test_scales_with_t(self) -> None:
        _, t12 = max_thickness_at(naca4(0, 0, 12, n_points=200))
        _, t24 = max_thickness_at(naca4(0, 0, 24, n_points=200))
        assert t24 == pytest.approx(2.0 * t12, rel=0.02)
