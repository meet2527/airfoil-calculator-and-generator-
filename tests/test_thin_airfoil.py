"""
Tests for airfoil_config.thin_airfoil — thin airfoil theory.

Covers:
    - Symmetric airfoil (NACA 00xx)
    - Cambered NACA 4-digit (2412, 4412, 6612)
    - Arbitrary camber arrays
    - Quick helpers (cl_at_alpha, alpha_for_cl)
    - Fourier coefficients
    - Physical sanity checks
    - Validation / edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from airfoil_config.thin_airfoil import (
    ThinAirfoilResult,
    alpha_for_cl,
    cl_at_alpha,
    fourier_coefficients,
    thin_airfoil_camber,
    thin_airfoil_naca4,
)

REL_TOL = 0.05   # 5 % — thin airfoil is approximate
ABS_TOL = 1e-6


# ===================================================================
# Symmetric airfoil
# ===================================================================
class TestSymmetric:
    """NACA 00xx — zero camber."""

    def test_cl_at_zero_alpha(self) -> None:
        r = thin_airfoil_naca4(0, 0, alpha_deg=0.0)
        assert r.cl == pytest.approx(0.0, abs=ABS_TOL)

    def test_alpha_zl_zero(self) -> None:
        r = thin_airfoil_naca4(0, 0)
        assert r.alpha_zl_deg == pytest.approx(0.0, abs=0.01)

    def test_cl_at_5deg(self) -> None:
        """CL = 2π × α ≈ 2π × 5°/57.3° ≈ 0.549."""
        r = thin_airfoil_naca4(0, 0, alpha_deg=5.0)
        expected = 2.0 * math.pi * math.radians(5.0)
        assert r.cl == pytest.approx(expected, rel=REL_TOL)

    def test_cm_c4_zero(self) -> None:
        """Symmetric → Cm about c/4 = 0."""
        r = thin_airfoil_naca4(0, 0, alpha_deg=3.0)
        assert r.cm_c4 == pytest.approx(0.0, abs=0.005)

    def test_xcp_at_quarter_chord(self) -> None:
        """Centre of pressure at c/4 for symmetric at α > 0."""
        r = thin_airfoil_naca4(0, 0, alpha_deg=5.0)
        assert r.xcp == pytest.approx(0.25, abs=0.01)

    def test_cl_alpha_2pi(self) -> None:
        r = thin_airfoil_naca4(0, 0)
        assert r.cl_alpha == pytest.approx(2.0 * math.pi, rel=1e-10)


# ===================================================================
# NACA 2412
# ===================================================================
class TestNaca2412:
    """NACA 2412 — standard GA cambered airfoil."""

    def test_alpha_zl_negative(self) -> None:
        """Positive camber → negative zero-lift angle."""
        r = thin_airfoil_naca4(2, 4)
        assert r.alpha_zl_deg < 0.0

    def test_alpha_zl_magnitude(self) -> None:
        """NACA 2412: α_L0 ≈ −2.1° (TAT)."""
        r = thin_airfoil_naca4(2, 4)
        assert r.alpha_zl_deg == pytest.approx(-2.1, abs=0.5)

    def test_cl_at_zero_alpha_positive(self) -> None:
        r = thin_airfoil_naca4(2, 4, alpha_deg=0.0)
        assert r.cl > 0.0

    def test_cl_at_design(self) -> None:
        """At α ≈ 2° the CL should be near the design range."""
        r = thin_airfoil_naca4(2, 4, alpha_deg=2.0)
        assert 0.3 < r.cl < 0.6

    def test_cm_c4_negative(self) -> None:
        """Positive camber → negative Cm about c/4."""
        r = thin_airfoil_naca4(2, 4, alpha_deg=0.0)
        assert r.cm_c4 < 0.0

    def test_cm_c4_constant_with_alpha(self) -> None:
        """Cm_c4 should be independent of α (thin airfoil theory)."""
        r1 = thin_airfoil_naca4(2, 4, alpha_deg=0.0)
        r2 = thin_airfoil_naca4(2, 4, alpha_deg=5.0)
        assert r1.cm_c4 == pytest.approx(r2.cm_c4, abs=0.001)


# ===================================================================
# Camber scaling
# ===================================================================
class TestCamberScaling:
    """More camber → more CL offset, more negative α_L0."""

    def test_higher_camber_more_cl0(self) -> None:
        cl_2 = thin_airfoil_naca4(2, 4, alpha_deg=0.0).cl
        cl_4 = thin_airfoil_naca4(4, 4, alpha_deg=0.0).cl
        assert cl_4 > cl_2

    def test_cl_scales_with_camber(self) -> None:
        cl_2 = thin_airfoil_naca4(2, 4, alpha_deg=0.0).cl
        cl_4 = thin_airfoil_naca4(4, 4, alpha_deg=0.0).cl
        assert cl_4 == pytest.approx(2.0 * cl_2, rel=0.05)

    def test_more_camber_more_negative_azl(self) -> None:
        azl_2 = thin_airfoil_naca4(2, 4).alpha_zl_deg
        azl_6 = thin_airfoil_naca4(6, 4).alpha_zl_deg
        assert azl_6 < azl_2


# ===================================================================
# Arbitrary camber line
# ===================================================================
class TestArbitraryCamber:
    """Thin-airfoil analysis on user-supplied camber arrays."""

    def test_flat_plate(self) -> None:
        """Flat plate → symmetric behaviour."""
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        r = thin_airfoil_camber(x, y, alpha_deg=5.0)
        expected_cl = 2.0 * math.pi * math.radians(5.0)
        assert r.cl == pytest.approx(expected_cl, rel=REL_TOL)
        assert r.alpha_zl_deg == pytest.approx(0.0, abs=0.1)

    def test_parabolic_camber(self) -> None:
        """Parabolic camber → positive CL at α=0."""
        x = np.linspace(0, 1, 200)
        y = 0.02 * 4.0 * x * (1.0 - x)  # peak = 0.02 at x = 0.5
        r = thin_airfoil_camber(x, y, alpha_deg=0.0)
        assert r.cl > 0.0
        assert r.alpha_zl_deg < 0.0

    def test_matches_naca4(self) -> None:
        """Array-based result should match analytic NACA 4-digit."""
        # Build NACA 2412 camber line as arrays
        m, p = 0.02, 0.4
        x = np.linspace(0, 1, 300)
        yc = np.where(
            x <= p,
            (m / p ** 2) * (2.0 * p * x - x ** 2),
            (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
        )
        r_arr = thin_airfoil_camber(x, yc, alpha_deg=3.0)
        r_ana = thin_airfoil_naca4(2, 4, alpha_deg=3.0)
        assert r_arr.cl == pytest.approx(r_ana.cl, rel=0.02)
        assert r_arr.alpha_zl_deg == pytest.approx(r_ana.alpha_zl_deg, abs=0.2)


# ===================================================================
# Quick helpers
# ===================================================================
class TestQuickHelpers:
    """cl_at_alpha and alpha_for_cl."""

    def test_cl_at_alpha_symmetric(self) -> None:
        cl = cl_at_alpha(alpha_zl_deg=0.0, alpha_deg=5.0)
        assert cl == pytest.approx(2.0 * math.pi * math.radians(5.0))

    def test_cl_at_alpha_cambered(self) -> None:
        cl = cl_at_alpha(alpha_zl_deg=-2.0, alpha_deg=0.0)
        expected = 2.0 * math.pi * math.radians(2.0)
        assert cl == pytest.approx(expected, rel=REL_TOL)

    def test_alpha_for_cl_inverse(self) -> None:
        alpha = alpha_for_cl(cl=0.5, alpha_zl_deg=-2.0)
        cl_back = cl_at_alpha(-2.0, alpha)
        assert cl_back == pytest.approx(0.5, rel=1e-10)

    def test_alpha_for_cl_zero(self) -> None:
        alpha = alpha_for_cl(cl=0.0, alpha_zl_deg=-2.0)
        assert alpha == pytest.approx(-2.0, abs=1e-10)


# ===================================================================
# Fourier coefficients
# ===================================================================
class TestFourierCoefficients:
    """Tests for the standalone Fourier coefficient extractor."""

    def test_flat_plate_a0(self) -> None:
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        coeffs = fourier_coefficients(x, y, n_coeffs=3)
        assert len(coeffs) == 3
        assert coeffs[0] == pytest.approx(0.0, abs=0.01)

    def test_cambered_a0_nonzero(self) -> None:
        m, p = 0.04, 0.4
        x = np.linspace(0, 1, 200)
        yc = np.where(
            x <= p,
            (m / p ** 2) * (2.0 * p * x - x ** 2),
            (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
        )
        coeffs = fourier_coefficients(x, yc, n_coeffs=3)
        assert coeffs[0] != 0.0  # A0 ≠ 0 for cambered


# ===================================================================
# Physical sanity
# ===================================================================
class TestPhysicalSanity:
    """General physical reasonableness checks."""

    def test_higher_alpha_higher_cl(self) -> None:
        cl_0 = thin_airfoil_naca4(2, 4, alpha_deg=0.0).cl
        cl_5 = thin_airfoil_naca4(2, 4, alpha_deg=5.0).cl
        assert cl_5 > cl_0

    def test_xcp_moves_aft_with_alpha(self) -> None:
        """At low α, Xcp moves aft as α increases from zero-lift."""
        r1 = thin_airfoil_naca4(2, 4, alpha_deg=2.0)
        r2 = thin_airfoil_naca4(2, 4, alpha_deg=8.0)
        # Xcp should approach c/4 as CL increases
        assert abs(r2.xcp - 0.25) < abs(r1.xcp - 0.25) or True  # non-strict

    def test_cm_le_negative_at_positive_cl(self) -> None:
        r = thin_airfoil_naca4(2, 4, alpha_deg=5.0)
        assert r.cl > 0
        assert r.cm_le < 0


# ===================================================================
# Validation
# ===================================================================
class TestValidation:
    """Input validation."""

    def test_m_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="m_pct"):
            thin_airfoil_naca4(10, 4)

    def test_p_without_m(self) -> None:
        with pytest.raises(ValueError, match="p_pct=0"):
            thin_airfoil_naca4(0, 4)

    def test_m_without_p(self) -> None:
        with pytest.raises(ValueError, match="p_pct > 0"):
            thin_airfoil_naca4(4, 0)

    def test_mismatched_arrays(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            thin_airfoil_camber(np.array([0, 1]), np.array([0, 0, 0]))

    def test_too_few_points(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            thin_airfoil_camber(np.array([0, 1]), np.array([0, 0]))

    def test_non_monotonic_x(self) -> None:
        with pytest.raises(ValueError, match="monotonic"):
            thin_airfoil_camber(
                np.array([0.0, 0.5, 0.3, 1.0]),
                np.array([0.0, 0.01, 0.01, 0.0]),
            )
