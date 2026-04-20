"""
Tests for airfoil_config.wing_planform — chord and twist distributions.

Covers:
    - Rectangular planform (taper_ratio=1)
    - Tapered planform (typical λ=0.45)
    - Elliptical planform
    - MAC analytic formulas
    - Sweep offsets
    - Washout / twist
    - Area consistency (integrated ≈ input)
    - Cosine vs uniform spacing
    - chord_at_eta standalone
    - Validation / edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from airfoil_config.wing_planform import (
    PlanformResult,
    WingStation,
    chord_at_eta,
    compute_elliptical_planform,
    compute_planform,
    mac_tapered,
)

REL_TOL = 1e-3   # 0.1 % for analytic checks
AREA_TOL = 0.02  # 2 % for numerical integration vs. analytic area


# ===================================================================
# Rectangular planform (λ = 1)
# ===================================================================
class TestRectangularPlanform:
    """Taper ratio = 1 → constant chord."""

    def test_uniform_chord(self) -> None:
        p = compute_planform(10.0, 16.0, taper_ratio=1.0, n_stations=11)
        chords = [s.chord_m for s in p.stations]
        assert all(c == pytest.approx(chords[0], rel=REL_TOL) for c in chords)

    def test_root_equals_tip(self) -> None:
        p = compute_planform(10.0, 16.0, taper_ratio=1.0)
        assert p.root_chord_m == pytest.approx(p.tip_chord_m, rel=REL_TOL)

    def test_root_chord_value(self) -> None:
        # S = b * c_r  for rectangular  →  c_r = S/b = 16/10 = 1.6
        p = compute_planform(10.0, 16.0, taper_ratio=1.0)
        assert p.root_chord_m == pytest.approx(1.6, rel=REL_TOL)

    def test_mac_equals_chord(self) -> None:
        p = compute_planform(10.0, 16.0, taper_ratio=1.0)
        assert p.mac_m == pytest.approx(p.root_chord_m, rel=REL_TOL)

    def test_area_matches(self) -> None:
        p = compute_planform(10.0, 16.0, taper_ratio=1.0, n_stations=51)
        assert p.wing_area_m2 == pytest.approx(16.0, rel=AREA_TOL)

    def test_planform_type(self) -> None:
        p = compute_planform(10.0, 16.0)
        assert p.planform_type == "tapered"


# ===================================================================
# Tapered planform (typical GA)
# ===================================================================
class TestTaperedPlanform:
    """Linearly tapered wing with λ = 0.45."""

    @pytest.fixture()
    def plan(self) -> PlanformResult:
        return compute_planform(
            wing_span_m=10.0, wing_area_m2=16.0,
            taper_ratio=0.45, n_stations=51,
        )

    def test_root_larger_than_tip(self, plan: PlanformResult) -> None:
        assert plan.root_chord_m > plan.tip_chord_m

    def test_tip_chord(self, plan: PlanformResult) -> None:
        assert plan.tip_chord_m == pytest.approx(
            plan.root_chord_m * 0.45, rel=REL_TOL
        )

    def test_chord_decreases_monotonically(self, plan: PlanformResult) -> None:
        chords = [s.chord_m for s in plan.stations]
        diffs = np.diff(chords)
        assert np.all(diffs <= 1e-10)  # non-increasing

    def test_area_matches(self, plan: PlanformResult) -> None:
        assert plan.wing_area_m2 == pytest.approx(16.0, rel=AREA_TOL)

    def test_mac_between_root_and_tip(self, plan: PlanformResult) -> None:
        assert plan.tip_chord_m < plan.mac_m < plan.root_chord_m

    def test_mac_analytic(self, plan: PlanformResult) -> None:
        """MAC should match the analytic formula."""
        lam = 0.45
        cr = plan.root_chord_m
        expected = (2.0 / 3.0) * cr * (1 + lam + lam**2) / (1 + lam)
        assert plan.mac_m == pytest.approx(expected, rel=REL_TOL)

    def test_aspect_ratio(self, plan: PlanformResult) -> None:
        assert plan.aspect_ratio == pytest.approx(
            10.0 ** 2 / 16.0, rel=REL_TOL
        )

    def test_station_count(self, plan: PlanformResult) -> None:
        assert len(plan.stations) == 51

    def test_root_at_centreline(self, plan: PlanformResult) -> None:
        assert plan.stations[0].y_m == pytest.approx(0.0, abs=1e-10)
        assert plan.stations[0].eta == pytest.approx(0.0, abs=1e-10)

    def test_tip_at_semi_span(self, plan: PlanformResult) -> None:
        assert plan.stations[-1].y_m == pytest.approx(5.0, rel=REL_TOL)
        assert plan.stations[-1].eta == pytest.approx(1.0, rel=REL_TOL)


# ===================================================================
# Washout / twist
# ===================================================================
class TestWashout:
    """Geometric twist (washout)."""

    def test_zero_at_root(self) -> None:
        p = compute_planform(10.0, 16.0, washout_deg=3.0, n_stations=11)
        assert p.stations[0].twist_deg == pytest.approx(0.0, abs=1e-10)

    def test_negative_at_tip(self) -> None:
        p = compute_planform(10.0, 16.0, washout_deg=3.0, n_stations=11)
        assert p.stations[-1].twist_deg == pytest.approx(-3.0, rel=REL_TOL)

    def test_linear_variation(self) -> None:
        p = compute_planform(
            10.0, 16.0, washout_deg=4.0,
            n_stations=11, cosine_spacing=False,
        )
        twists = [s.twist_deg for s in p.stations]
        # Uniform spacing → constant twist increment
        diffs = np.diff(twists)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)

    def test_no_twist_when_zero(self) -> None:
        p = compute_planform(10.0, 16.0, washout_deg=0.0, n_stations=11)
        for s in p.stations:
            assert s.twist_deg == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# Sweep
# ===================================================================
class TestSweep:
    """Leading-edge sweep offsets."""

    def test_zero_sweep_no_offset(self) -> None:
        p = compute_planform(10.0, 16.0, le_sweep_deg=0.0, n_stations=5)
        for s in p.stations:
            assert s.le_sweep_offset_m == pytest.approx(0.0, abs=1e-10)

    def test_positive_sweep(self) -> None:
        p = compute_planform(10.0, 16.0, le_sweep_deg=30.0, n_stations=5)
        offsets = [s.le_sweep_offset_m for s in p.stations]
        assert offsets[0] == pytest.approx(0.0, abs=1e-10)
        assert offsets[-1] > 0.0  # swept back
        # offset = y * tan(30°)
        expected = 5.0 * math.tan(math.radians(30.0))
        assert offsets[-1] == pytest.approx(expected, rel=REL_TOL)


# ===================================================================
# Elliptical planform
# ===================================================================
class TestEllipticalPlanform:
    """Elliptical chord distribution."""

    @pytest.fixture()
    def plan(self) -> PlanformResult:
        return compute_elliptical_planform(
            10.0, 16.0, n_stations=101,
        )

    def test_root_chord(self, plan: PlanformResult) -> None:
        expected = (4.0 * 16.0) / (math.pi * 10.0)
        assert plan.root_chord_m == pytest.approx(expected, rel=REL_TOL)

    def test_tip_chord_zero(self, plan: PlanformResult) -> None:
        assert plan.tip_chord_m == pytest.approx(0.0, abs=1e-10)

    def test_tip_station_zero_chord(self, plan: PlanformResult) -> None:
        assert plan.stations[-1].chord_m == pytest.approx(0.0, abs=1e-6)

    def test_area_matches(self, plan: PlanformResult) -> None:
        assert plan.wing_area_m2 == pytest.approx(16.0, rel=AREA_TOL)

    def test_mac_value(self, plan: PlanformResult) -> None:
        # MAC_elliptical = (π/4) * c_0
        expected = (math.pi / 4.0) * plan.root_chord_m
        assert plan.mac_m == pytest.approx(expected, rel=REL_TOL)

    def test_chord_decreases(self, plan: PlanformResult) -> None:
        chords = [s.chord_m for s in plan.stations]
        diffs = np.diff(chords)
        assert np.all(diffs <= 1e-10)

    def test_planform_type(self, plan: PlanformResult) -> None:
        assert plan.planform_type == "elliptical"

    def test_taper_ratio_zero(self, plan: PlanformResult) -> None:
        assert plan.taper_ratio == 0.0


# ===================================================================
# Standalone chord_at_eta
# ===================================================================
class TestChordAtEta:
    """Tests for the standalone chord lookup."""

    def test_root(self) -> None:
        assert chord_at_eta(2.0, 0.5, 0.0) == pytest.approx(2.0)

    def test_tip(self) -> None:
        assert chord_at_eta(2.0, 0.5, 1.0) == pytest.approx(1.0)

    def test_mid(self) -> None:
        assert chord_at_eta(2.0, 0.5, 0.5) == pytest.approx(1.5)

    def test_rectangular(self) -> None:
        assert chord_at_eta(1.6, 1.0, 0.7) == pytest.approx(1.6)

    def test_eta_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            chord_at_eta(2.0, 0.5, -0.1)

    def test_eta_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            chord_at_eta(2.0, 0.5, 1.1)

    def test_nan_eta_raises(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            chord_at_eta(2.0, 0.5, float("nan"))


# ===================================================================
# Standalone mac_tapered
# ===================================================================
class TestMacTapered:
    """Tests for the analytic MAC function."""

    def test_rectangular(self) -> None:
        """λ = 1 → MAC = c_r."""
        assert mac_tapered(1.6, 1.0) == pytest.approx(1.6, rel=REL_TOL)

    def test_pointed_tip(self) -> None:
        """λ = 0 → MAC = (2/3) c_r."""
        assert mac_tapered(3.0, 0.0) == pytest.approx(2.0, rel=REL_TOL)

    def test_typical_taper(self) -> None:
        lam = 0.45
        cr = 2.2
        expected = (2.0 / 3.0) * cr * (1 + lam + lam**2) / (1 + lam)
        assert mac_tapered(cr, lam) == pytest.approx(expected, rel=REL_TOL)

    def test_negative_chord_raises(self) -> None:
        with pytest.raises(ValueError, match="root_chord_m"):
            mac_tapered(-1.0, 0.5)

    def test_taper_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="taper_ratio"):
            mac_tapered(2.0, 1.5)


# ===================================================================
# Cosine vs uniform spacing
# ===================================================================
class TestSpacing:
    """Verify spacing options work correctly."""

    def test_cosine_clusters_tip(self) -> None:
        p = compute_planform(10.0, 16.0, n_stations=21, cosine_spacing=True)
        etas = [s.eta for s in p.stations]
        # Cosine: more stations near η = 1 (tip)
        mid_gap = etas[11] - etas[10]
        tip_gap = etas[-1] - etas[-2]
        assert tip_gap < mid_gap  # denser near tip

    def test_uniform_equal_gaps(self) -> None:
        p = compute_planform(
            10.0, 16.0, n_stations=11, cosine_spacing=False,
        )
        etas = [s.eta for s in p.stations]
        gaps = np.diff(etas)
        np.testing.assert_allclose(gaps, gaps[0], atol=1e-10)


# ===================================================================
# Validation
# ===================================================================
class TestValidation:
    """Input validation."""

    def test_zero_span_raises(self) -> None:
        with pytest.raises(ValueError, match="wing_span_m"):
            compute_planform(0.0, 16.0)

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError, match="wing_area_m2"):
            compute_planform(10.0, -1.0)

    def test_taper_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="taper_ratio"):
            compute_planform(10.0, 16.0, taper_ratio=1.5)

    def test_negative_taper_raises(self) -> None:
        with pytest.raises(ValueError, match="taper_ratio"):
            compute_planform(10.0, 16.0, taper_ratio=-0.1)

    def test_one_station_raises(self) -> None:
        with pytest.raises(ValueError, match="n_stations"):
            compute_planform(10.0, 16.0, n_stations=1)

    def test_nan_span_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compute_planform(float("nan"), 16.0)

    def test_elliptical_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="wing_area_m2"):
            compute_elliptical_planform(10.0, 0.0)
