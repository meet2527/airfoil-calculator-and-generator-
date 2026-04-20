"""
Tests for airfoil_config.cq_airfoil — CadQuery wire builder.

CadQuery may not be installed, so tests that require it are guarded
behind ``pytest.mark.skipif``.  The 3-D point builder and validation
are tested unconditionally.

Covers:
    - 3-D point generation (scaling, twist, sweep, dihedral)
    - Selig-loop ordering
    - Chord validation
    - AirfoilSection dataclass
    - CadQuery wire/face (when CQ available)
    - DXF export (when ezdxf available)
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from airfoil_config.naca_geometry import AirfoilCoordinates, naca4

# Import the module — CQ may or may not be available
from airfoil_config.cq_airfoil import (
    CQ_AVAILABLE,
    AirfoilSection,
    get_3d_points,
    _validate_chord,
)

ABS_TOL = 1e-6
REL_TOL = 1e-4


# ===================================================================
# Helpers
# ===================================================================
def _naca0012() -> AirfoilCoordinates:
    return naca4(0, 0, 12, n_points=50)


def _naca2412() -> AirfoilCoordinates:
    return naca4(2, 4, 12, n_points=50)


# ===================================================================
# 3-D point generation (no CadQuery required)
# ===================================================================
class TestGet3dPoints:
    """Tests for get_3d_points — pure numpy, no CQ needed."""

    def test_shape(self) -> None:
        pts = get_3d_points(_naca0012())
        # Selig loop: 2*n_points - 1
        assert pts.shape == (2 * 50 - 1, 3)

    def test_y_constant(self) -> None:
        """All points should be at y = 0 by default."""
        pts = get_3d_points(_naca0012())
        np.testing.assert_allclose(pts[:, 1], 0.0, atol=ABS_TOL)

    def test_y_offset(self) -> None:
        pts = get_3d_points(_naca0012(), y_m=3.5)
        np.testing.assert_allclose(pts[:, 1], 3.5, atol=ABS_TOL)

    def test_chord_scaling(self) -> None:
        """Points should span [0, chord_m] in x approximately."""
        pts = get_3d_points(_naca0012(), chord_m=2.0)
        x_range = pts[:, 0].max() - pts[:, 0].min()
        assert x_range == pytest.approx(2.0, rel=0.02)

    def test_unit_chord(self) -> None:
        pts = get_3d_points(_naca0012(), chord_m=1.0)
        x_range = pts[:, 0].max() - pts[:, 0].min()
        assert x_range == pytest.approx(1.0, rel=0.02)

    def test_sweep_offset(self) -> None:
        """Sweep should shift all x by the offset."""
        pts_0 = get_3d_points(_naca0012(), le_sweep_offset_m=0.0)
        pts_s = get_3d_points(_naca0012(), le_sweep_offset_m=1.5)
        dx = pts_s[:, 0] - pts_0[:, 0]
        np.testing.assert_allclose(dx, 1.5, atol=ABS_TOL)

    def test_dihedral_offset(self) -> None:
        """Dihedral should shift all z by the offset."""
        pts_0 = get_3d_points(_naca0012(), le_dihedral_offset_m=0.0)
        pts_d = get_3d_points(_naca0012(), le_dihedral_offset_m=0.5)
        dz = pts_d[:, 2] - pts_0[:, 2]
        np.testing.assert_allclose(dz, 0.5, atol=ABS_TOL)

    def test_zero_twist_no_change(self) -> None:
        pts_0 = get_3d_points(_naca0012(), twist_deg=0.0)
        pts_t = get_3d_points(_naca0012(), twist_deg=0.0)
        np.testing.assert_allclose(pts_0, pts_t, atol=ABS_TOL)

    def test_twist_rotates_points(self) -> None:
        """Non-zero twist should move points in the x-z plane."""
        pts_0 = get_3d_points(_naca0012(), twist_deg=0.0, chord_m=1.0)
        pts_t = get_3d_points(_naca0012(), twist_deg=10.0, chord_m=1.0)
        # Points should differ
        assert not np.allclose(pts_0, pts_t, atol=1e-3)
        # y should be unchanged
        np.testing.assert_allclose(pts_0[:, 1], pts_t[:, 1], atol=ABS_TOL)

    def test_twist_preserves_le(self) -> None:
        """Twist about LE: the LE point (x≈0, z≈0) shouldn't move much."""
        pts = get_3d_points(_naca0012(), twist_deg=15.0, chord_m=1.0)
        # LE is in the middle of the Selig loop (index n_points - 1 = 49)
        le_x = pts[49, 0]
        le_z = pts[49, 2]
        assert abs(le_x) < 0.01
        assert abs(le_z) < 0.01

    def test_symmetric_z(self) -> None:
        """NACA 0012 should have symmetric z (upper vs lower)."""
        pts = get_3d_points(_naca0012(), chord_m=1.0)
        n = 50
        # Upper (indices 0..48) reversed = TE..LE
        # Lower (indices 49..97) = LE..TE
        z_upper = pts[:n, 2]  # TE → LE
        z_lower = pts[n - 1:, 2]  # LE → TE
        # Upper reversed should mirror lower
        np.testing.assert_allclose(z_upper[::-1], -z_lower, atol=1e-4)

    def test_cambered_not_symmetric(self) -> None:
        pts = get_3d_points(_naca2412(), chord_m=1.0)
        n = 50
        z_upper = pts[:n, 2]
        z_lower = pts[n - 1:, 2]
        assert not np.allclose(z_upper[::-1], -z_lower, atol=1e-3)


# ===================================================================
# Chord validation
# ===================================================================
class TestChordValidation:
    """Tests for _validate_chord."""

    def test_positive_ok(self) -> None:
        _validate_chord(1.5)  # should not raise

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="chord_m"):
            _validate_chord(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chord_m"):
            _validate_chord(-1.0)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="chord_m"):
            _validate_chord(float("nan"))


# ===================================================================
# AirfoilSection dataclass
# ===================================================================
class TestAirfoilSection:
    """Tests for the AirfoilSection dataclass."""

    def test_create(self) -> None:
        s = AirfoilSection(
            coords=_naca0012(), chord_m=1.6,
            twist_deg=-2.0, y_m=3.0,
        )
        assert s.chord_m == 1.6
        assert s.twist_deg == -2.0
        assert s.y_m == 3.0

    def test_frozen(self) -> None:
        s = AirfoilSection(coords=_naca0012(), chord_m=1.0)
        with pytest.raises(AttributeError):
            s.chord_m = 2.0  # type: ignore[misc]

    def test_defaults(self) -> None:
        s = AirfoilSection(coords=_naca0012(), chord_m=1.0)
        assert s.twist_deg == 0.0
        assert s.y_m == 0.0
        assert s.le_sweep_offset_m == 0.0
        assert s.le_dihedral_offset_m == 0.0


# ===================================================================
# CadQuery wire / face (skip if CQ not installed)
# ===================================================================
@pytest.mark.skipif(not CQ_AVAILABLE, reason="CadQuery not installed")
class TestCadQueryWire:
    """Tests requiring CadQuery."""

    def test_wire_created(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_wire
        wire = airfoil_wire(_naca0012(), chord_m=1.0)
        assert wire is not None
        assert wire.IsClosed()

    def test_face_created(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_face
        face = airfoil_face(_naca0012(), chord_m=1.0)
        assert face is not None
        assert face.Area() > 0

    def test_section_wire(self) -> None:
        from airfoil_config.cq_airfoil import section_wire
        s = AirfoilSection(coords=_naca0012(), chord_m=1.6, y_m=2.0)
        wire = section_wire(s)
        assert wire.IsClosed()

    def test_polyline_wire(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_wire
        wire = airfoil_wire(_naca0012(), chord_m=1.0, spline=False)
        assert wire.IsClosed()

    def test_workplane(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_workplane
        wp = airfoil_workplane(_naca2412(), chord_m=1.5)
        assert wp is not None


# ===================================================================
# CQ not available — RuntimeError
# ===================================================================
@pytest.mark.skipif(CQ_AVAILABLE, reason="CadQuery IS installed")
class TestCqNotAvailable:
    """When CadQuery is missing, wire/face should raise RuntimeError."""

    def test_wire_raises(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_wire
        with pytest.raises(RuntimeError, match="CadQuery"):
            airfoil_wire(_naca0012())

    def test_face_raises(self) -> None:
        from airfoil_config.cq_airfoil import airfoil_face
        with pytest.raises(RuntimeError, match="CadQuery"):
            airfoil_face(_naca0012())


# ===================================================================
# DXF export
# ===================================================================
class TestDxfExport:
    """Tests for export_wire_dxf (requires ezdxf)."""

    def test_creates_file(self, tmp_path) -> None:
        try:
            from airfoil_config.cq_airfoil import export_wire_dxf
        except ImportError:
            pytest.skip("ezdxf not installed")

        path = str(tmp_path / "test.dxf")
        export_wire_dxf(_naca0012(), path, chord_m=1.0)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 100

    def test_negative_chord_raises(self) -> None:
        try:
            from airfoil_config.cq_airfoil import export_wire_dxf
        except ImportError:
            pytest.skip("ezdxf not installed")

        with pytest.raises(ValueError, match="chord_m"):
            export_wire_dxf(_naca0012(), "dummy.dxf", chord_m=-1.0)
