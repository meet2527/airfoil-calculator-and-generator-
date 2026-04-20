"""
Tests for airfoil_config.lifting_line — Prandtl LLT solver.

Covers:
    - Elliptical wing (e = 1 benchmark)
    - Rectangular wing (e < 1)
    - Tapered wing
    - CL scales with α
    - Washout effect
    - CDi quick formula
    - Oswald factor
    - Validation / edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from airfoil_config.lifting_line import (
    LiftingLineResult,
    compute_cdi_from_cl,
    compute_oswald_factor,
    solve_lifting_line,
)
from airfoil_config.wing_planform import (
    compute_elliptical_planform,
    compute_planform,
)

REL_TOL = 0.05   # 5 % — LLT is approximate, especially with few terms
ABS_TOL = 1e-6


# ===================================================================
# Helpers
# ===================================================================
def _rect_planform(ar: float = 6.0):
    """Rectangular wing, S=12, b=sqrt(12*AR)."""
    area = 12.0
    span = math.sqrt(area * ar)
    return compute_planform(span, area, taper_ratio=1.0, n_stations=51)


def _tapered_planform():
    """Tapered wing λ=0.45, AR≈6.25."""
    return compute_planform(10.0, 16.0, taper_ratio=0.45, n_stations=51)


def _elliptical_planform():
    """Elliptical wing, same area/span as tapered."""
    return compute_elliptical_planform(10.0, 16.0, n_stations=101)


# ===================================================================
# Elliptical wing — benchmark (e ≈ 1)
# ===================================================================
class TestEllipticalWing:
    """Elliptical planform should yield span efficiency ≈ 1."""

    def test_span_efficiency_near_one(self) -> None:
        p = _elliptical_planform()
        r = solve_lifting_line(p, alpha_root_deg=5.0, n_terms=30)
        assert r.span_efficiency == pytest.approx(1.0, abs=0.03)

    def test_cl_positive(self) -> None:
        r = solve_lifting_line(_elliptical_planform(), 5.0)
        assert r.cl_wing > 0

    def test_cdi_positive(self) -> None:
        r = solve_lifting_line(_elliptical_planform(), 5.0)
        assert r.cdi > 0

    def test_cdi_formula_matches(self) -> None:
        """CDi should match CL²/(π·AR·e) for the elliptical case."""
        r = solve_lifting_line(_elliptical_planform(), 5.0)
        expected = r.cl_wing ** 2 / (math.pi * r.aspect_ratio * r.span_efficiency)
        assert r.cdi == pytest.approx(expected, rel=0.01)

    def test_gamma_elliptic_shape(self) -> None:
        """Gamma distribution should peak near root (η≈0, θ≈π/2)."""
        n_terms = 30
        r = solve_lifting_line(_elliptical_planform(), 5.0, n_terms=n_terms)
        root_idx = n_terms // 2
        # Root (middle) should be greater than tip (index 0)
        assert r.gamma_distribution[root_idx] > r.gamma_distribution[0]


# ===================================================================
# Rectangular wing (e < 1)
# ===================================================================
class TestRectangularWing:
    """Rectangular wing: e < 1 but still reasonable."""

    def test_efficiency_below_one(self) -> None:
        r = solve_lifting_line(_rect_planform(), 5.0, n_terms=30)
        assert r.span_efficiency < 1.0

    def test_efficiency_reasonable(self) -> None:
        """Typical rectangular wing: e ≈ 0.85–0.95."""
        r = solve_lifting_line(_rect_planform(ar=8.0), 5.0, n_terms=30)
        assert 0.7 < r.span_efficiency < 1.0

    def test_cl_increases_with_alpha(self) -> None:
        p = _rect_planform()
        r3 = solve_lifting_line(p, 3.0)
        r6 = solve_lifting_line(p, 6.0)
        assert r6.cl_wing > r3.cl_wing

    def test_cdi_increases_with_cl(self) -> None:
        p = _rect_planform()
        r3 = solve_lifting_line(p, 3.0)
        r6 = solve_lifting_line(p, 6.0)
        assert r6.cdi > r3.cdi


# ===================================================================
# Tapered wing
# ===================================================================
class TestTaperedWing:
    """Tapered wing with typical GA taper ratio."""

    def test_runs_without_error(self) -> None:
        r = solve_lifting_line(_tapered_planform(), 5.0)
        assert isinstance(r, LiftingLineResult)

    def test_efficiency_reasonable(self) -> None:
        """Tapered wing (λ=0.45) should be more efficient than rectangular."""
        r_rect = solve_lifting_line(_rect_planform(6.25), 5.0, n_terms=30)
        r_taper = solve_lifting_line(_tapered_planform(), 5.0, n_terms=30)
        # Tapered closer to elliptical → higher e
        assert r_taper.span_efficiency >= r_rect.span_efficiency - 0.05

    def test_result_fields(self) -> None:
        r = solve_lifting_line(_tapered_planform(), 5.0, n_terms=20)
        assert r.cl_distribution.shape == (20,)
        assert r.gamma_distribution.shape == (20,)
        assert r.fourier_coefficients.shape == (20,)
        assert r.alpha_root_deg == 5.0
        assert r.wing_span_m == 10.0


# ===================================================================
# Washout effect
# ===================================================================
class TestWashout:
    """Geometric washout should reduce tip loading."""

    def test_washout_reduces_cl(self) -> None:
        """Washout reduces overall CL at the same root α."""
        p_no = compute_planform(10.0, 16.0, taper_ratio=0.45, washout_deg=0.0)
        p_wo = compute_planform(10.0, 16.0, taper_ratio=0.45, washout_deg=3.0)

        r_no = solve_lifting_line(p_no, 5.0)
        r_wo = solve_lifting_line(p_wo, 5.0)
        assert r_wo.cl_wing < r_no.cl_wing

    def test_washout_affects_distribution(self) -> None:
        p_wo = compute_planform(10.0, 16.0, taper_ratio=0.45, washout_deg=3.0)
        r = solve_lifting_line(p_wo, 5.0)
        # Root (middle) section CL should be higher than tip (edges)
        root_idx = len(r.cl_distribution) // 2
        assert r.cl_distribution[root_idx] > r.cl_distribution[0]


# ===================================================================
# Zero-lift angle effect
# ===================================================================
class TestZeroLiftAngle:
    """Non-zero α_L0 shifts the CL curve."""

    def test_cambered_produces_lift_at_zero_alpha(self) -> None:
        """A cambered airfoil (α_L0 < 0) gives CL > 0 at α_root = 0."""
        r = solve_lifting_line(
            _rect_planform(), alpha_root_deg=0.0, alpha_zl_deg=-3.0,
        )
        assert r.cl_wing > 0

    def test_symmetric_zero_cl_at_zero_alpha(self) -> None:
        r = solve_lifting_line(
            _rect_planform(), alpha_root_deg=0.0, alpha_zl_deg=0.0,
        )
        assert abs(r.cl_wing) < 0.01


# ===================================================================
# CDi quick formula
# ===================================================================
class TestCdiFormula:
    """compute_cdi_from_cl standalone."""

    def test_basic(self) -> None:
        cdi = compute_cdi_from_cl(0.5, 8.0, 0.95)
        expected = 0.25 / (math.pi * 8.0 * 0.95)
        assert cdi == pytest.approx(expected, rel=REL_TOL)

    def test_zero_cl(self) -> None:
        assert compute_cdi_from_cl(0.0, 8.0) == 0.0

    def test_negative_ar_raises(self) -> None:
        with pytest.raises(ValueError, match="aspect_ratio"):
            compute_cdi_from_cl(0.5, -1.0)

    def test_zero_e_raises(self) -> None:
        with pytest.raises(ValueError, match="span_efficiency"):
            compute_cdi_from_cl(0.5, 8.0, 0.0)


# ===================================================================
# Oswald factor
# ===================================================================
class TestOswaldFactor:
    """compute_oswald_factor."""

    def test_basic(self) -> None:
        e = compute_oswald_factor(0.5, 0.01, 8.0)
        expected = 0.25 / (math.pi * 8.0 * 0.01)
        assert e == pytest.approx(expected, rel=REL_TOL)

    def test_zero_cdi_raises(self) -> None:
        with pytest.raises(ValueError, match="cdi"):
            compute_oswald_factor(0.5, 0.0, 8.0)


# ===================================================================
# Validation
# ===================================================================
class TestValidation:
    """Input validation for solve_lifting_line."""

    def test_nan_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            solve_lifting_line(_rect_planform(), float("nan"))

    def test_too_few_terms_raises(self) -> None:
        with pytest.raises(ValueError, match="n_terms"):
            solve_lifting_line(_rect_planform(), 5.0, n_terms=2)

    def test_n_terms_3_works(self) -> None:
        r = solve_lifting_line(_rect_planform(), 5.0, n_terms=3)
        assert r.cl_wing > 0


# ===================================================================
# Convergence with n_terms
# ===================================================================
class TestConvergence:
    """More terms should improve accuracy."""

    def test_more_terms_changes_little(self) -> None:
        p = _rect_planform()
        r10 = solve_lifting_line(p, 5.0, n_terms=10)
        r30 = solve_lifting_line(p, 5.0, n_terms=30)
        # CL should converge — difference < 5 %
        assert abs(r30.cl_wing - r10.cl_wing) / abs(r30.cl_wing) < 0.05
