"""
Tests for airfoil_config.loft — wing lofting.

Most tests use ``dry_run=True`` so CadQuery is not required.
CadQuery-dependent tests are behind ``@pytest.mark.skipif``.

Covers:
    - Section ordering (the bug fix)
    - build_sections_from_planform
    - build_sections_varying_airfoil
    - validate_sections
    - loft_wing dry_run
    - loft_wing with CadQuery (skipif)
    - LoftResult metadata
    - Edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from airfoil_config.cq_airfoil import AirfoilSection, CQ_AVAILABLE
from airfoil_config.loft import (
    LoftResult,
    build_sections_from_planform,
    build_sections_varying_airfoil,
    loft_wing,
    validate_sections,
)
from airfoil_config.naca_geometry import AirfoilCoordinates, naca4
from airfoil_config.wing_planform import compute_planform, compute_elliptical_planform


# ===================================================================
# Helpers
# ===================================================================
def _naca0012(n: int = 30) -> AirfoilCoordinates:
    return naca4(0, 0, 12, n_points=n)


def _naca2412(n: int = 30) -> AirfoilCoordinates:
    return naca4(2, 4, 12, n_points=n)


def _tapered_planform():
    return compute_planform(10.0, 16.0, taper_ratio=0.45, n_stations=11)


def _simple_sections(n: int = 5, reverse: bool = False) -> list[AirfoilSection]:
    """Create n sections at evenly spaced y positions."""
    coords = _naca0012()
    y_vals = list(np.linspace(0.0, 5.0, n))
    chords = list(np.linspace(2.0, 1.0, n))
    if reverse:
        y_vals = y_vals[::-1]
        chords = chords[::-1]
    return [
        AirfoilSection(coords=coords, chord_m=c, y_m=y)
        for y, c in zip(y_vals, chords)
    ]


# ===================================================================
# Section ordering — the bug fix
# ===================================================================
class TestSectionOrdering:
    """Verify sections are sorted root → tip regardless of input order."""

    def test_already_sorted(self) -> None:
        secs = _simple_sections(5, reverse=False)
        result = loft_wing(secs, dry_run=True)
        y_vals = [s.y_m for s in result.sections]
        assert y_vals == sorted(y_vals)

    def test_reversed_input_gets_sorted(self) -> None:
        """THE BUG FIX: reversed input must be sorted before lofting."""
        secs = _simple_sections(5, reverse=True)
        result = loft_wing(secs, dry_run=True)
        y_vals = [s.y_m for s in result.sections]
        assert y_vals == sorted(y_vals)

    def test_random_order_gets_sorted(self) -> None:
        """Arbitrary section order should be sorted."""
        coords = _naca0012()
        secs = [
            AirfoilSection(coords=coords, chord_m=1.5, y_m=3.0),
            AirfoilSection(coords=coords, chord_m=2.0, y_m=0.0),
            AirfoilSection(coords=coords, chord_m=1.0, y_m=5.0),
            AirfoilSection(coords=coords, chord_m=1.8, y_m=1.0),
        ]
        result = loft_wing(secs, dry_run=True)
        y_vals = [s.y_m for s in result.sections]
        assert y_vals == [0.0, 1.0, 3.0, 5.0]

    def test_root_is_first(self) -> None:
        secs = _simple_sections(5, reverse=True)
        result = loft_wing(secs, dry_run=True)
        assert result.sections[0].y_m == 0.0

    def test_tip_is_last(self) -> None:
        secs = _simple_sections(5, reverse=True)
        result = loft_wing(secs, dry_run=True)
        assert result.sections[-1].y_m == 5.0


# ===================================================================
# build_sections_from_planform
# ===================================================================
class TestBuildSectionsFromPlanform:
    """Tests for the planform → sections builder."""

    def test_returns_sections(self) -> None:
        secs = build_sections_from_planform(_tapered_planform(), _naca0012())
        assert len(secs) > 0
        assert all(isinstance(s, AirfoilSection) for s in secs)

    def test_sorted_root_to_tip(self) -> None:
        secs = build_sections_from_planform(_tapered_planform(), _naca0012())
        y_vals = [s.y_m for s in secs]
        assert y_vals == sorted(y_vals)

    def test_root_chord_matches_planform(self) -> None:
        plan = _tapered_planform()
        secs = build_sections_from_planform(plan, _naca0012())
        assert secs[0].chord_m == pytest.approx(plan.root_chord_m, rel=0.01)

    def test_tip_chord_matches_planform(self) -> None:
        plan = _tapered_planform()
        secs = build_sections_from_planform(plan, _naca0012())
        assert secs[-1].chord_m == pytest.approx(plan.tip_chord_m, rel=0.01)

    def test_station_indices(self) -> None:
        plan = _tapered_planform()
        secs = build_sections_from_planform(plan, _naca0012(), station_indices=[0, 5, 10])
        assert len(secs) == 3

    def test_min_chord_filter(self) -> None:
        """Elliptical tip (chord=0) should be filtered out."""
        plan = compute_elliptical_planform(10.0, 16.0, n_stations=21)
        secs = build_sections_from_planform(plan, _naca0012(), min_chord_m=0.01)
        for s in secs:
            assert s.chord_m >= 0.01

    def test_twist_from_planform(self) -> None:
        plan = compute_planform(
            10.0, 16.0, taper_ratio=0.45,
            washout_deg=3.0, n_stations=11,
        )
        secs = build_sections_from_planform(plan, _naca0012())
        assert secs[0].twist_deg == pytest.approx(0.0, abs=0.01)
        assert secs[-1].twist_deg < 0  # washout


# ===================================================================
# build_sections_varying_airfoil
# ===================================================================
class TestBuildSectionsVaryingAirfoil:
    """Tests for varying-airfoil section builder."""

    def test_two_airfoils(self) -> None:
        plan = _tapered_planform()
        airfoils = {
            0.0: _naca2412(),  # root
            1.0: _naca0012(),  # tip
        }
        secs = build_sections_varying_airfoil(plan, airfoils)
        assert len(secs) > 0
        # Root section should use NACA 2412 coords
        assert secs[0].coords is airfoils[0.0]

    def test_sorted(self) -> None:
        plan = _tapered_planform()
        airfoils = {0.0: _naca0012(), 1.0: _naca0012()}
        secs = build_sections_varying_airfoil(plan, airfoils)
        y_vals = [s.y_m for s in secs]
        assert y_vals == sorted(y_vals)

    def test_empty_airfoils_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_sections_varying_airfoil(_tapered_planform(), {})


# ===================================================================
# validate_sections
# ===================================================================
class TestValidateSections:
    """Tests for the section validator."""

    def test_valid_returns_empty(self) -> None:
        secs = _simple_sections(5)
        warnings = validate_sections(secs)
        assert warnings == []

    def test_too_few_sections(self) -> None:
        secs = _simple_sections(1)
        warnings = validate_sections(secs)
        assert any("Need >= 2" in w for w in warnings)

    def test_duplicate_y(self) -> None:
        coords = _naca0012()
        secs = [
            AirfoilSection(coords=coords, chord_m=1.0, y_m=2.0),
            AirfoilSection(coords=coords, chord_m=1.0, y_m=2.0),
        ]
        warnings = validate_sections(secs)
        assert any("Duplicate" in w for w in warnings)

    def test_zero_chord_warning(self) -> None:
        coords = _naca0012()
        secs = [
            AirfoilSection(coords=coords, chord_m=1.0, y_m=0.0),
            AirfoilSection(coords=coords, chord_m=0.0, y_m=5.0),
        ]
        warnings = validate_sections(secs)
        assert any("zero chord" in w for w in warnings)

    def test_large_twist_warning(self) -> None:
        coords = _naca0012()
        secs = [
            AirfoilSection(coords=coords, chord_m=1.0, y_m=0.0),
            AirfoilSection(coords=coords, chord_m=1.0, y_m=5.0, twist_deg=25.0),
        ]
        warnings = validate_sections(secs)
        assert any("twist" in w for w in warnings)

    def test_inconsistent_points(self) -> None:
        secs = [
            AirfoilSection(coords=naca4(0, 0, 12, n_points=30), chord_m=1.0, y_m=0.0),
            AirfoilSection(coords=naca4(0, 0, 12, n_points=50), chord_m=1.0, y_m=5.0),
        ]
        warnings = validate_sections(secs)
        assert any("point count" in w.lower() for w in warnings)


# ===================================================================
# loft_wing dry_run
# ===================================================================
class TestLoftDryRun:
    """Tests for loft_wing with dry_run=True (no CQ needed)."""

    def test_returns_result(self) -> None:
        result = loft_wing(_simple_sections(5), dry_run=True)
        assert isinstance(result, LoftResult)

    def test_solid_is_none(self) -> None:
        result = loft_wing(_simple_sections(5), dry_run=True)
        assert result.solid is None

    def test_n_sections(self) -> None:
        result = loft_wing(_simple_sections(5), dry_run=True)
        assert result.n_sections == 5

    def test_span(self) -> None:
        result = loft_wing(_simple_sections(5), dry_run=True)
        assert result.span_m == pytest.approx(5.0)

    def test_is_ruled(self) -> None:
        r_smooth = loft_wing(_simple_sections(3), ruled=False, dry_run=True)
        r_ruled = loft_wing(_simple_sections(3), ruled=True, dry_run=True)
        assert r_smooth.is_ruled is False
        assert r_ruled.is_ruled is True

    def test_too_few_sections_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            loft_wing([_simple_sections(1)[0]], dry_run=True)

    def test_export_step_raises_on_dry_run(self) -> None:
        result = loft_wing(_simple_sections(3), dry_run=True)
        with pytest.raises(RuntimeError):
            result.export_step("test.step")


# ===================================================================
# loft_wing with CadQuery (skip if not installed)
# ===================================================================
@pytest.mark.skipif(not CQ_AVAILABLE, reason="CadQuery not installed")
class TestLoftWithCadQuery:
    """Integration tests requiring CadQuery."""

    def test_solid_created(self) -> None:
        result = loft_wing(_simple_sections(5))
        assert result.solid is not None

    def test_export_step(self, tmp_path) -> None:
        result = loft_wing(_simple_sections(5))
        path = str(tmp_path / "wing.step")
        result.export_step(path)
        import os
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 100

    def test_ruled_loft(self) -> None:
        result = loft_wing(_simple_sections(5), ruled=True)
        assert result.solid is not None

    def test_planform_loft(self) -> None:
        """Full pipeline: planform → sections → loft."""
        plan = _tapered_planform()
        secs = build_sections_from_planform(plan, _naca0012())
        # Use every other station to keep it fast
        secs_subset = secs[::2]
        if len(secs_subset) < 2:
            secs_subset = secs[:2]
        result = loft_wing(secs_subset)
        assert result.solid is not None
