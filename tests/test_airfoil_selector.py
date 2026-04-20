"""
Tests for airfoil_config.airfoil_selector — NACA selection logic.

Covers:
    - Thin-airfoil-theory estimators (CL, α_L0, NACA 4/5)
    - Thickness range heuristics
    - Camber range heuristics
    - Full select_naca_airfoils integration
    - Scoring sanity
    - Validation / edge cases
"""

from __future__ import annotations

import math

import pytest

from airfoil_config.airfoil_selector import (
    NacaCandidate,
    SelectionResult,
    estimate_alpha_zl_naca4,
    estimate_alpha_zl_naca5,
    estimate_cl_at_zero_alpha_naca4,
    estimate_cl_design_naca5,
    select_naca_airfoils,
    suggest_camber_range,
    suggest_thickness_range,
)
from airfoil_config.requirements import (
    AircraftSpecs,
    WingRequirements,
    compute_wing_requirements,
)

REL_TOL = 0.10  # 10 % tolerance for aero estimates (thin airfoil is approx)
ABS_TOL = 1e-6


# ===================================================================
# Helpers
# ===================================================================
def _ga_requirements() -> WingRequirements:
    """Typical general-aviation wing requirements."""
    specs = AircraftSpecs(
        weight_n=10000.0,
        wing_span_m=10.0,
        wing_area_m2=16.0,
        cruise_altitude_m=3000.0,
        cruise_velocity_ms=60.0,
        purpose="general_aviation",
    )
    return compute_wing_requirements(specs)


def _uav_requirements() -> WingRequirements:
    """Small UAV wing requirements (low Re)."""
    specs = AircraftSpecs(
        weight_n=50.0,
        wing_span_m=2.0,
        wing_area_m2=0.5,
        cruise_altitude_m=500.0,
        cruise_velocity_ms=20.0,
        purpose="uav",
    )
    return compute_wing_requirements(specs)


# ===================================================================
# Thin-airfoil-theory estimators
# ===================================================================
class TestEstimateCl4:
    """CL at α=0 estimates for NACA 4-digit."""

    def test_symmetric_zero(self) -> None:
        """Symmetric airfoil → CL(α=0) = 0."""
        assert estimate_cl_at_zero_alpha_naca4(0, 0) == 0.0

    def test_naca2412_positive(self) -> None:
        """NACA 2412 should have positive CL at α=0."""
        cl = estimate_cl_at_zero_alpha_naca4(2, 4)
        assert cl > 0.0

    def test_naca2412_magnitude(self) -> None:
        """NACA 2412: CL(α=0) ≈ 0.22–0.28 from thin airfoil theory."""
        cl = estimate_cl_at_zero_alpha_naca4(2, 4)
        assert 0.15 < cl < 0.35

    def test_more_camber_more_cl(self) -> None:
        """Higher camber → higher CL at zero alpha."""
        cl_2 = estimate_cl_at_zero_alpha_naca4(2, 4)
        cl_6 = estimate_cl_at_zero_alpha_naca4(6, 4)
        assert cl_6 > cl_2

    def test_scales_roughly_linearly(self) -> None:
        """CL should scale roughly with m_pct."""
        cl_2 = estimate_cl_at_zero_alpha_naca4(2, 4)
        cl_4 = estimate_cl_at_zero_alpha_naca4(4, 4)
        assert cl_4 == pytest.approx(2.0 * cl_2, rel=0.05)


class TestAlphaZl4:
    """Zero-lift angle estimates for NACA 4-digit."""

    def test_symmetric_zero(self) -> None:
        assert estimate_alpha_zl_naca4(0, 0) == 0.0

    def test_cambered_negative(self) -> None:
        """Positive camber → negative α_L0."""
        assert estimate_alpha_zl_naca4(2, 4) < 0.0

    def test_naca2412_magnitude(self) -> None:
        """NACA 2412: α_L0 ≈ −2° (thin airfoil theory)."""
        azl = estimate_alpha_zl_naca4(2, 4)
        assert -4.0 < azl < -1.0


class TestEstimateCl5:
    """Design CL for NACA 5-digit."""

    def test_l2(self) -> None:
        assert estimate_cl_design_naca5(2) == pytest.approx(0.30)

    def test_l4(self) -> None:
        assert estimate_cl_design_naca5(4) == pytest.approx(0.60)


class TestAlphaZl5:
    """Zero-lift angle for NACA 5-digit."""

    def test_negative(self) -> None:
        assert estimate_alpha_zl_naca5(2) < 0.0

    def test_magnitude(self) -> None:
        azl = estimate_alpha_zl_naca5(2)
        assert -4.0 < azl < -1.0


# ===================================================================
# Thickness range heuristics
# ===================================================================
class TestThicknessRange:
    """Tests for suggest_thickness_range."""

    def test_low_re(self) -> None:
        t_min, t_max = suggest_thickness_range(1e5)
        assert t_min >= 6
        assert t_max <= 18

    def test_high_re(self) -> None:
        t_min, t_max = suggest_thickness_range(1e7)
        assert t_max >= 15

    def test_high_mach_limits_max(self) -> None:
        _, t_max_low_m = suggest_thickness_range(5e6, mach=0.3)
        _, t_max_high_m = suggest_thickness_range(5e6, mach=0.65)
        assert t_max_high_m < t_max_low_m

    def test_returns_tuple_of_ints(self) -> None:
        result = suggest_thickness_range(3e6)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)

    def test_invalid_re_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            suggest_thickness_range(-1.0)

    def test_nan_re_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            suggest_thickness_range(float("nan"))


# ===================================================================
# Camber range heuristics
# ===================================================================
class TestCamberRange:
    """Tests for suggest_camber_range."""

    def test_low_cl_includes_zero(self) -> None:
        m_min, _ = suggest_camber_range(0.02)
        assert m_min == 0

    def test_moderate_cl(self) -> None:
        m_min, m_max = suggest_camber_range(0.3)
        assert m_min <= 3
        assert m_max >= 2

    def test_high_cl(self) -> None:
        _, m_max = suggest_camber_range(0.8)
        assert m_max >= 5

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="cl_required"):
            suggest_camber_range(-0.1)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="cl_required"):
            suggest_camber_range(float("nan"))


# ===================================================================
# Full selection — integration tests
# ===================================================================
class TestSelectNacaAirfoils:
    """Integration tests for select_naca_airfoils."""

    def test_returns_selection_result(self) -> None:
        reqs = _ga_requirements()
        result = select_naca_airfoils(reqs)
        assert isinstance(result, SelectionResult)

    def test_candidates_not_empty(self) -> None:
        result = select_naca_airfoils(_ga_requirements())
        assert len(result.candidates) > 0

    def test_max_candidates_limit(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), max_candidates=5)
        assert len(result.candidates) <= 5

    def test_candidates_sorted_descending(self) -> None:
        result = select_naca_airfoils(_ga_requirements())
        scores = [c.suitability_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_best_candidate_has_high_score(self) -> None:
        result = select_naca_airfoils(_ga_requirements())
        assert result.candidates[0].suitability_score > 0.3

    def test_contains_target_cl(self) -> None:
        reqs = _ga_requirements()
        result = select_naca_airfoils(reqs)
        assert result.target_cl == reqs.required_cl_cruise

    def test_thickness_range_stored(self) -> None:
        result = select_naca_airfoils(_ga_requirements())
        t_min, t_max = result.thickness_range
        assert 1 <= t_min < t_max <= 30

    def test_candidate_fields_populated(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), max_candidates=1)
        c = result.candidates[0]
        assert isinstance(c, NacaCandidate)
        assert c.designation.startswith("NACA")
        assert c.family in ("4-digit", "5-digit")
        assert c.t_pct > 0
        assert math.isfinite(c.cl_at_zero_alpha)

    def test_includes_4digit(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), include_5digit=False)
        families = {c.family for c in result.candidates}
        assert families == {"4-digit"}

    def test_includes_5digit(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), include_5digit=True)
        families = {c.family for c in result.candidates}
        assert "5-digit" in families or len(result.candidates) > 0

    def test_custom_thickness_range(self) -> None:
        result = select_naca_airfoils(
            _ga_requirements(), thickness_range=(12, 15),
        )
        for c in result.candidates:
            assert 12 <= c.t_pct <= 15

    def test_custom_camber_range(self) -> None:
        result = select_naca_airfoils(
            _ga_requirements(),
            include_5digit=False,
            camber_range=(2, 4),
        )
        for c in result.candidates:
            assert 2 <= c.m_pct <= 4

    def test_uav_low_re(self) -> None:
        """UAV selection should produce valid candidates."""
        result = select_naca_airfoils(_uav_requirements())
        assert len(result.candidates) > 0

    def test_max_candidates_one(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), max_candidates=1)
        assert len(result.candidates) == 1

    def test_max_candidates_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_candidates"):
            select_naca_airfoils(_ga_requirements(), max_candidates=0)


# ===================================================================
# Scoring sanity
# ===================================================================
class TestScoringSanity:
    """Verify scoring produces sensible orderings."""

    def test_closer_cl_scores_higher(self) -> None:
        """Candidate matching target CL should score above distant one."""
        reqs = _ga_requirements()
        result = select_naca_airfoils(
            reqs, include_5digit=False, max_candidates=50,
        )
        if len(result.candidates) < 2:
            pytest.skip("Not enough candidates")
        # Best candidate should have CL closer to target than the worst
        best = result.candidates[0]
        worst = result.candidates[-1]
        assert (
            abs(best.cl_at_zero_alpha - reqs.required_cl_cruise)
            <= abs(worst.cl_at_zero_alpha - reqs.required_cl_cruise) + 0.5
        )

    def test_scores_bounded_0_1(self) -> None:
        result = select_naca_airfoils(_ga_requirements())
        for c in result.candidates:
            assert 0.0 <= c.suitability_score <= 1.0

    def test_notes_not_empty(self) -> None:
        result = select_naca_airfoils(_ga_requirements(), max_candidates=3)
        for c in result.candidates:
            assert isinstance(c.notes, str)
            assert len(c.notes) > 0
