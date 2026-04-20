"""
Tests for airfoil_config.scoring — airfoil ranking from polar data.

Covers:
    - Polar metrics extraction
    - Sub-score computation
    - Single airfoil scoring
    - Multi-airfoil ranking
    - Comparison helper
    - Custom weights
    - Edge cases (low points, extreme values)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from airfoil_config.requirements import (
    AircraftSpecs,
    WingRequirements,
    compute_wing_requirements,
)
from airfoil_config.scoring import (
    DEFAULT_WEIGHTS,
    PolarMetrics,
    ScoredAirfoil,
    ScoringWeights,
    compare_polars,
    compute_polar_metrics,
    rank_airfoils,
    score_airfoil,
)
from airfoil_config.xfoil_runner import XfoilPolar

REL_TOL = 0.05
ABS_TOL = 1e-6


# ===================================================================
# Helpers
# ===================================================================
def _ga_reqs() -> WingRequirements:
    """Typical GA requirements (CL_cruise ≈ 0.28)."""
    specs = AircraftSpecs(
        weight_n=10000.0, wing_span_m=10.0, wing_area_m2=16.0,
        cruise_altitude_m=3000.0, cruise_velocity_ms=60.0,
        purpose="general_aviation",
    )
    return compute_wing_requirements(specs)


def _make_polar(
    designation: str = "NACA 0012",
    n: int = 15,
    cl_slope: float = 0.11,
    cd_base: float = 0.006,
    cm_val: float = -0.02,
    cl_offset: float = 0.0,
) -> XfoilPolar:
    """Create a synthetic polar with controllable characteristics."""
    alpha = np.linspace(-4.0, 16.0, n)
    cl_linear = cl_offset + cl_slope * alpha
    # Add stall-like roll-off above α ≈ 12°
    cl = np.where(alpha < 12.0, cl_linear, cl_linear - 0.05 * (alpha - 12.0) ** 2)
    cd = cd_base + 0.0002 * alpha ** 2
    cdp = cd * 0.4
    cm = cm_val * np.ones(n)

    return XfoilPolar(
        alpha_deg=alpha, cl=cl, cd=cd, cdp=cdp, cm=cm,
        top_xtr=0.5 * np.ones(n), bot_xtr=0.6 * np.ones(n),
        reynolds=1e6, mach=0.0, n_crit=9.0,
        designation=designation,
        converged_count=n, total_count=n,
    )


# ===================================================================
# Polar metrics
# ===================================================================
class TestPolarMetrics:
    """Tests for compute_polar_metrics."""

    def test_returns_metrics(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert isinstance(m, PolarMetrics)

    def test_designation_preserved(self) -> None:
        m = compute_polar_metrics(_make_polar("NACA 2412"), _ga_reqs())
        assert m.designation == "NACA 2412"

    def test_ld_positive(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert m.ld_at_cruise > 0
        assert m.ld_max > 0

    def test_ld_max_ge_ld_cruise(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert m.ld_max >= m.ld_at_cruise - 0.1  # small tolerance

    def test_cl_max_positive(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert m.cl_max > 0

    def test_cd_min_positive(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert m.cd_min > 0
        assert m.cd_min <= m.cd_at_cruise

    def test_stall_margin(self) -> None:
        reqs = _ga_reqs()
        m = compute_polar_metrics(_make_polar(), reqs)
        assert m.stall_margin_cl == pytest.approx(
            m.cl_max - reqs.required_cl_cruise, abs=ABS_TOL,
        )

    def test_cl_alpha_reasonable(self) -> None:
        """Lift-curve slope should be near 2π ≈ 6.28 for a thin airfoil."""
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert 4.0 < m.cl_alpha_rad < 8.0

    def test_endurance_positive(self) -> None:
        m = compute_polar_metrics(_make_polar(), _ga_reqs())
        assert m.endurance_param > 0

    def test_too_few_points_raises(self) -> None:
        alpha = np.array([0.0])
        polar = XfoilPolar(
            alpha_deg=alpha, cl=np.array([0.5]),
            cd=np.array([0.01]), cdp=np.array([0.005]),
            cm=np.array([-0.02]),
            top_xtr=np.array([0.5]), bot_xtr=np.array([0.5]),
            reynolds=1e6, mach=0.0, n_crit=9.0,
            designation="test", converged_count=1, total_count=1,
        )
        with pytest.raises(ValueError, match="need >= 2"):
            compute_polar_metrics(polar, _ga_reqs())


# ===================================================================
# Cambered vs symmetric
# ===================================================================
class TestCamberedVsSymmetric:
    """Cambered airfoil should rate differently than symmetric."""

    def test_cambered_more_cl_at_zero(self) -> None:
        reqs = _ga_reqs()
        m_sym = compute_polar_metrics(_make_polar(cl_offset=0.0), reqs)
        m_cam = compute_polar_metrics(_make_polar(cl_offset=0.25), reqs)
        # Cambered → higher CL_max
        assert m_cam.cl_max > m_sym.cl_max


# ===================================================================
# Single scoring
# ===================================================================
class TestScoreAirfoil:
    """Tests for score_airfoil."""

    def test_returns_scored(self) -> None:
        sa = score_airfoil(_make_polar(), _ga_reqs())
        assert isinstance(sa, ScoredAirfoil)

    def test_score_in_0_1(self) -> None:
        sa = score_airfoil(_make_polar(), _ga_reqs())
        assert 0.0 <= sa.total_score <= 1.0

    def test_sub_scores_present(self) -> None:
        sa = score_airfoil(_make_polar(), _ga_reqs())
        expected_keys = {
            "ld_cruise", "cl_match", "cd_level",
            "stall_margin", "cm_magnitude", "endurance",
        }
        assert set(sa.sub_scores.keys()) == expected_keys

    def test_sub_scores_in_0_1(self) -> None:
        sa = score_airfoil(_make_polar(), _ga_reqs())
        for k, v in sa.sub_scores.items():
            assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"

    def test_rank_unset(self) -> None:
        """Single scoring should have rank 0 (unranked)."""
        sa = score_airfoil(_make_polar(), _ga_reqs())
        assert sa.rank == 0

    def test_has_metrics(self) -> None:
        sa = score_airfoil(_make_polar(), _ga_reqs())
        assert isinstance(sa.metrics, PolarMetrics)


# ===================================================================
# Ranking
# ===================================================================
class TestRankAirfoils:
    """Tests for rank_airfoils."""

    def test_returns_list(self) -> None:
        polars = {
            "A": _make_polar("A"),
            "B": _make_polar("B"),
        }
        ranked = rank_airfoils(polars, _ga_reqs())
        assert isinstance(ranked, list)
        assert len(ranked) == 2

    def test_sorted_descending(self) -> None:
        polars = {
            "A": _make_polar("A", cd_base=0.010),
            "B": _make_polar("B", cd_base=0.005),
        }
        ranked = rank_airfoils(polars, _ga_reqs())
        scores = [r.total_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_assigned(self) -> None:
        polars = {
            "A": _make_polar("A"),
            "B": _make_polar("B"),
            "C": _make_polar("C"),
        }
        ranked = rank_airfoils(polars, _ga_reqs())
        ranks = [r.rank for r in ranked]
        assert ranks == [1, 2, 3]

    def test_best_has_rank_1(self) -> None:
        polars = {"X": _make_polar("X")}
        ranked = rank_airfoils(polars, _ga_reqs())
        assert ranked[0].rank == 1

    def test_low_drag_ranks_higher(self) -> None:
        """Lower drag airfoil should rank above higher drag."""
        polars = {
            "high_drag": _make_polar("high_drag", cd_base=0.015),
            "low_drag": _make_polar("low_drag", cd_base=0.004),
        }
        ranked = rank_airfoils(polars, _ga_reqs())
        assert ranked[0].designation == "low_drag"

    def test_skips_short_polars(self) -> None:
        short = XfoilPolar(
            alpha_deg=np.array([0.0]), cl=np.array([0.5]),
            cd=np.array([0.01]), cdp=np.array([0.005]),
            cm=np.array([-0.02]),
            top_xtr=np.array([0.5]), bot_xtr=np.array([0.5]),
            reynolds=1e6, mach=0.0, n_crit=9.0,
            designation="short", converged_count=1, total_count=1,
        )
        polars = {
            "short": short,
            "ok": _make_polar("ok"),
        }
        ranked = rank_airfoils(polars, _ga_reqs())
        assert len(ranked) == 1
        assert ranked[0].designation == "ok"

    def test_empty_dict(self) -> None:
        ranked = rank_airfoils({}, _ga_reqs())
        assert ranked == []


# ===================================================================
# Custom weights
# ===================================================================
class TestCustomWeights:
    """Scoring with custom weights."""

    def test_ld_only(self) -> None:
        """When only L/D matters, higher L/D should dominate."""
        w = ScoringWeights(
            ld_cruise=1.0, cl_match=0.0, cd_level=0.0,
            stall_margin=0.0, cm_magnitude=0.0, endurance=0.0,
        )
        polars = {
            "low_ld": _make_polar("low_ld", cd_base=0.020),
            "high_ld": _make_polar("high_ld", cd_base=0.003),
        }
        ranked = rank_airfoils(polars, _ga_reqs(), weights=w)
        assert ranked[0].designation == "high_ld"

    def test_cm_only(self) -> None:
        """When only Cm matters, lower |Cm| should win."""
        w = ScoringWeights(
            ld_cruise=0.0, cl_match=0.0, cd_level=0.0,
            stall_margin=0.0, cm_magnitude=1.0, endurance=0.0,
        )
        polars = {
            "high_cm": _make_polar("high_cm", cm_val=-0.08),
            "low_cm": _make_polar("low_cm", cm_val=-0.01),
        }
        ranked = rank_airfoils(polars, _ga_reqs(), weights=w)
        assert ranked[0].designation == "low_cm"


# ===================================================================
# Comparison helper
# ===================================================================
class TestComparePolars:
    """Tests for compare_polars."""

    def test_returns_metrics_list(self) -> None:
        polars = {
            "A": _make_polar("A"),
            "B": _make_polar("B"),
        }
        results = compare_polars(polars, _ga_reqs())
        assert len(results) == 2
        assert all(isinstance(r, PolarMetrics) for r in results)


# ===================================================================
# Scoring weights
# ===================================================================
class TestScoringWeights:
    """ScoringWeights dataclass."""

    def test_defaults(self) -> None:
        w = DEFAULT_WEIGHTS
        total = (w.ld_cruise + w.cl_match + w.cd_level
                 + w.stall_margin + w.cm_magnitude + w.endurance)
        assert total == pytest.approx(1.0)

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            DEFAULT_WEIGHTS.ld_cruise = 0.5  # type: ignore[misc]
