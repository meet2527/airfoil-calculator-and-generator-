"""
Tests for airfoil_config.report_generator — PDF report output.

Tests that require matplotlib + ReportLab are skipped if either
dependency is missing.  Structure / data-class tests run always.

Covers:
    - ReportData construction
    - Generate report with all sections
    - Generate report with minimal data
    - Comparison report
    - Missing dependencies raise RuntimeError
    - Plot helpers produce valid images
    - Table styling
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from airfoil_config.naca_geometry import naca4
from airfoil_config.report_generator import (
    MPL_AVAILABLE,
    RL_AVAILABLE,
    ReportData,
    generate_report,
    generate_comparison_report,
)
from airfoil_config.requirements import AircraftSpecs, WingRequirements, compute_wing_requirements
from airfoil_config.scoring import PolarMetrics
from airfoil_config.xfoil_runner import XfoilPolar


# Skip all if either dep missing
DEPS_OK = MPL_AVAILABLE and RL_AVAILABLE
skip_no_deps = pytest.mark.skipif(not DEPS_OK, reason="matplotlib or ReportLab not installed")


# ===================================================================
# Helpers
# ===================================================================
def _specs() -> AircraftSpecs:
    return AircraftSpecs(
        weight_n=10000.0, wing_span_m=10.0, wing_area_m2=16.0,
        cruise_altitude_m=3000.0, cruise_velocity_ms=60.0,
        purpose="general_aviation",
    )


def _requirements() -> WingRequirements:
    return compute_wing_requirements(_specs())


def _polar() -> XfoilPolar:
    n = 10
    alpha = np.linspace(-2, 12, n)
    return XfoilPolar(
        alpha_deg=alpha,
        cl=0.11 * alpha + 0.2,
        cd=0.006 + 0.0002 * alpha ** 2,
        cdp=0.003 + 0.0001 * alpha ** 2,
        cm=-0.03 * np.ones(n),
        top_xtr=0.5 * np.ones(n),
        bot_xtr=0.6 * np.ones(n),
        reynolds=1e6, mach=0.0, n_crit=9.0,
        designation="NACA 2412",
        converged_count=n, total_count=n,
    )


def _metrics() -> PolarMetrics:
    return PolarMetrics(
        designation="NACA 2412",
        cl_at_cruise_alpha=0.4,
        cd_at_cruise=0.008,
        ld_at_cruise=50.0,
        ld_max=72.0,
        alpha_ld_max_deg=5.0,
        cl_max=1.3,
        alpha_cl_max_deg=12.0,
        cd_min=0.005,
        cm_at_cruise=-0.03,
        alpha_zl_deg=-2.1,
        cl_alpha_rad=6.1,
        stall_margin_cl=0.9,
        endurance_param=18.0,
    )


# ===================================================================
# ReportData
# ===================================================================
class TestReportData:
    """ReportData construction — no deps needed."""

    def test_minimal(self) -> None:
        d = ReportData(title="Test")
        assert d.title == "Test"
        assert d.specs is None

    def test_full(self) -> None:
        d = ReportData(
            title="Full Report",
            specs=_specs(),
            requirements=_requirements(),
            coords=naca4(2, 4, 12),
            polar=_polar(),
            metrics=_metrics(),
            notes="Test note",
            author="Engineer",
        )
        assert d.author == "Engineer"
        assert d.polar is not None


# ===================================================================
# Full report generation
# ===================================================================
@skip_no_deps
class TestGenerateReport:
    """Tests that actually produce PDF files."""

    def test_minimal_report(self, tmp_path) -> None:
        path = str(tmp_path / "minimal.pdf")
        data = ReportData(title="Minimal Report")
        result = generate_report(data, path)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 100

    def test_full_report(self, tmp_path) -> None:
        path = str(tmp_path / "full.pdf")
        data = ReportData(
            title="NACA 2412 — Full Analysis",
            specs=_specs(),
            requirements=_requirements(),
            coords=naca4(2, 4, 12),
            polar=_polar(),
            metrics=_metrics(),
            notes="This is a test report.",
            author="Test Suite",
            date="2026-01-01",
        )
        result = generate_report(data, path)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 1000

    def test_specs_only(self, tmp_path) -> None:
        path = str(tmp_path / "specs.pdf")
        data = ReportData(
            title="Specs Report",
            specs=_specs(),
        )
        generate_report(data, path)
        assert os.path.isfile(path)

    def test_geometry_only(self, tmp_path) -> None:
        path = str(tmp_path / "geom.pdf")
        data = ReportData(
            title="Geometry Report",
            coords=naca4(0, 0, 12),
        )
        generate_report(data, path)
        assert os.path.isfile(path)

    def test_polar_only(self, tmp_path) -> None:
        path = str(tmp_path / "polar.pdf")
        data = ReportData(
            title="Polar Report",
            polar=_polar(),
        )
        generate_report(data, path)
        assert os.path.isfile(path)

    def test_with_llt(self, tmp_path) -> None:
        path = str(tmp_path / "llt.pdf")
        data = ReportData(
            title="LLT Report",
            llt_eta=np.linspace(0, 1, 20),
            llt_cl=0.5 * np.sqrt(1 - np.linspace(0, 1, 20) ** 2),
        )
        generate_report(data, path)
        assert os.path.isfile(path)

    def test_returns_absolute_path(self, tmp_path) -> None:
        path = str(tmp_path / "abs.pdf")
        result = generate_report(ReportData(title="Test"), path)
        assert os.path.isabs(result)


# ===================================================================
# Comparison report
# ===================================================================
@skip_no_deps
class TestComparisonReport:
    """Multi-airfoil comparison report."""

    def test_creates_pdf(self, tmp_path) -> None:
        path = str(tmp_path / "compare.pdf")
        polars = {
            "NACA 0012": _polar(),
            "NACA 2412": _polar(),
        }
        result = generate_comparison_report(
            "Comparison", polars, _requirements(), path,
        )
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 500


# ===================================================================
# Plot helpers (with matplotlib)
# ===================================================================
@skip_no_deps
class TestPlotHelpers:
    """Verify plot functions produce files."""

    def test_airfoil_geometry_plot(self) -> None:
        from airfoil_config.report_generator import _plot_airfoil_geometry
        path = _plot_airfoil_geometry(naca4(2, 4, 12))
        assert os.path.isfile(path)
        os.unlink(path)

    def test_polar_plots(self) -> None:
        from airfoil_config.report_generator import _plot_polar
        paths = _plot_polar(_polar())
        assert len(paths) == 4
        for p in paths:
            assert os.path.isfile(p)
            os.unlink(p)

    def test_llt_plot(self) -> None:
        from airfoil_config.report_generator import _plot_llt_distribution
        path = _plot_llt_distribution(
            np.linspace(0, 1, 10),
            0.5 * np.ones(10),
        )
        assert os.path.isfile(path)
        os.unlink(path)


# ===================================================================
# Missing dependencies
# ===================================================================
@pytest.mark.skipif(DEPS_OK, reason="Dependencies ARE available")
class TestMissingDeps:
    """When matplotlib or ReportLab is missing, should raise."""

    def test_raises_runtime_error(self, tmp_path) -> None:
        with pytest.raises(RuntimeError):
            generate_report(
                ReportData(title="Test"),
                str(tmp_path / "test.pdf"),
            )


# ===================================================================
# Temp image cleanup
# ===================================================================
@skip_no_deps
class TestCleanup:
    """Verify temp images are cleaned up."""

    def test_no_leftover_files(self, tmp_path) -> None:
        import tempfile
        before = set(os.listdir(tempfile.gettempdir()))
        path = str(tmp_path / "cleanup.pdf")
        data = ReportData(
            title="Cleanup Test",
            coords=naca4(0, 0, 12),
            polar=_polar(),
        )
        generate_report(data, path)
        after = set(os.listdir(tempfile.gettempdir()))
        # Our rpt_*.png files should be gone
        new_files = after - before
        rpt_files = [f for f in new_files if f.startswith("rpt_")]
        assert len(rpt_files) == 0
