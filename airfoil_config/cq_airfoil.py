"""
CadQuery airfoil wire and face builder.

Converts airfoil coordinate arrays (from ``naca_geometry``) into
CadQuery ``Wire`` and ``Face`` objects suitable for lofting into
3-D wing solids.

This is the **only** module that imports CadQuery (heavy dependency).
All other modules work with pure numpy arrays.

Typical usage:
    >>> from airfoil_config.naca_geometry import naca4, to_single_surface
    >>> from airfoil_config.cq_airfoil import airfoil_wire, airfoil_face
    >>> coords = naca4(2, 4, 12)
    >>> wire = airfoil_wire(coords, chord_m=1.6, twist_deg=0.0)
    >>> face = airfoil_face(coords, chord_m=1.6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cadquery as cq
    CQ_AVAILABLE = True
except ImportError:
    CQ_AVAILABLE = False

from airfoil_config.naca_geometry import AirfoilCoordinates, AirfoilSingleSurface


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AirfoilSection:
    """A positioned airfoil section in 3-D space.

    Attributes:
        coords: Source airfoil coordinates (normalised, chord = 1).
        chord_m: Chord length [m].
        twist_deg: Geometric twist about LE, positive = LE up [deg].
        y_m: Spanwise position (along wing span axis) [m].
        le_sweep_offset_m: Chordwise offset of LE from root LE [m].
        le_dihedral_offset_m: Vertical offset of LE [m].
    """

    coords: AirfoilCoordinates
    chord_m: float
    twist_deg: float = 0.0
    y_m: float = 0.0
    le_sweep_offset_m: float = 0.0
    le_dihedral_offset_m: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def airfoil_wire(
    coords: AirfoilCoordinates,
    chord_m: float = 1.0,
    twist_deg: float = 0.0,
    y_m: float = 0.0,
    le_sweep_offset_m: float = 0.0,
    le_dihedral_offset_m: float = 0.0,
    spline: bool = True,
) -> "cq.Wire":
    """Build a closed CadQuery Wire from airfoil coordinates.

    The airfoil is scaled by *chord_m*, rotated by *twist_deg* about
    the leading edge, and translated to the specified (x, y, z) position.

    Convention:
        - x-axis = chordwise (TE → LE is negative x)
        - y-axis = spanwise
        - z-axis = vertical (lift direction)

    Args:
        coords: Normalised airfoil coordinates (chord = 1).
        chord_m: Chord length [m].
        twist_deg: Geometric twist [deg].
        y_m: Spanwise position [m].
        le_sweep_offset_m: Chordwise LE offset [m].
        le_dihedral_offset_m: Vertical LE offset [m].
        spline: If True, use spline interpolation (default).
            If False, use polyline.

    Returns:
        Closed CadQuery ``Wire``.

    Raises:
        RuntimeError: If CadQuery is not installed.
        ValueError: If *chord_m* is non-positive.
    """
    _check_cq()
    _validate_chord(chord_m)

    pts = _build_3d_points(
        coords, chord_m, twist_deg,
        y_m, le_sweep_offset_m, le_dihedral_offset_m,
    )

    if spline:
        wire = cq.Wire.makeSpline(
            [cq.Vector(*p) for p in pts],
            periodic=True,
        )
    else:
        edges = []
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            edges.append(
                cq.Edge.makeLine(
                    cq.Vector(*pts[i]),
                    cq.Vector(*pts[j]),
                )
            )
        wire = cq.Wire.assembleEdges(edges)

    return wire


def airfoil_face(
    coords: AirfoilCoordinates,
    chord_m: float = 1.0,
    twist_deg: float = 0.0,
    y_m: float = 0.0,
    le_sweep_offset_m: float = 0.0,
    le_dihedral_offset_m: float = 0.0,
) -> "cq.Face":
    """Build a planar CadQuery Face from airfoil coordinates.

    Creates a closed wire and makes a face from it.

    Args:
        coords: Normalised airfoil coordinates.
        chord_m: Chord length [m].
        twist_deg: Twist [deg].
        y_m: Spanwise position [m].
        le_sweep_offset_m: Chordwise LE offset [m].
        le_dihedral_offset_m: Vertical LE offset [m].

    Returns:
        Planar CadQuery ``Face``.

    Raises:
        RuntimeError: If CadQuery is not installed.
    """
    _check_cq()
    wire = airfoil_wire(
        coords, chord_m, twist_deg,
        y_m, le_sweep_offset_m, le_dihedral_offset_m,
    )
    return cq.Face.makeFromWires(wire)


def airfoil_workplane(
    coords: AirfoilCoordinates,
    chord_m: float = 1.0,
    twist_deg: float = 0.0,
    y_m: float = 0.0,
    le_sweep_offset_m: float = 0.0,
    le_dihedral_offset_m: float = 0.0,
) -> "cq.Workplane":
    """Build a CadQuery Workplane with the airfoil wire on it.

    Useful for interactive CadQuery workflows and exporting.

    Args:
        coords: Normalised airfoil coordinates.
        chord_m: Chord [m].
        twist_deg: Twist [deg].
        y_m: Spanwise position [m].
        le_sweep_offset_m: Chordwise LE offset [m].
        le_dihedral_offset_m: Vertical LE offset [m].

    Returns:
        CadQuery ``Workplane`` with the airfoil wire.
    """
    _check_cq()
    wire = airfoil_wire(
        coords, chord_m, twist_deg,
        y_m, le_sweep_offset_m, le_dihedral_offset_m,
    )
    return cq.Workplane("XZ").add(wire)


def section_wire(section: AirfoilSection, spline: bool = True) -> "cq.Wire":
    """Build a wire from an :class:`AirfoilSection`.

    Args:
        section: Positioned airfoil section.
        spline: Use spline (True) or polyline (False).

    Returns:
        CadQuery ``Wire``.
    """
    return airfoil_wire(
        section.coords, section.chord_m, section.twist_deg,
        section.y_m, section.le_sweep_offset_m,
        section.le_dihedral_offset_m, spline=spline,
    )


def section_face(section: AirfoilSection) -> "cq.Face":
    """Build a face from an :class:`AirfoilSection`.

    Args:
        section: Positioned airfoil section.

    Returns:
        CadQuery ``Face``.
    """
    return airfoil_face(
        section.coords, section.chord_m, section.twist_deg,
        section.y_m, section.le_sweep_offset_m,
        section.le_dihedral_offset_m,
    )


def export_wire_dxf(
    coords: AirfoilCoordinates,
    filepath: str,
    chord_m: float = 1.0,
) -> None:
    """Export an airfoil wire as a DXF file (2-D profile).

    Uses *ezdxf* for export — does not require CadQuery.

    Args:
        coords: Normalised airfoil coordinates.
        filepath: Output DXF file path.
        chord_m: Chord scale [m].

    Raises:
        ImportError: If *ezdxf* is not installed.
        ValueError: If *chord_m* is non-positive.
    """
    import ezdxf

    _validate_chord(chord_m)

    # Build 2-D points (x, z) — no twist, no offset
    x_upper = coords.x_upper * chord_m
    y_upper = coords.y_upper * chord_m
    x_lower = coords.x_lower * chord_m
    y_lower = coords.y_lower * chord_m

    # Selig loop: TE → upper → LE → lower → TE
    x_loop = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_loop = np.concatenate([y_upper[::-1], y_lower[1:]])

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    points_2d = [(float(x), float(y)) for x, y in zip(x_loop, y_loop)]
    # Close the loop
    points_2d.append(points_2d[0])
    msp.add_lwpolyline(points_2d, close=True)

    doc.saveas(filepath)


# ---------------------------------------------------------------------------
# Public — coordinate utilities
# ---------------------------------------------------------------------------
def get_3d_points(
    coords: AirfoilCoordinates,
    chord_m: float = 1.0,
    twist_deg: float = 0.0,
    y_m: float = 0.0,
    le_sweep_offset_m: float = 0.0,
    le_dihedral_offset_m: float = 0.0,
) -> np.ndarray:
    """Get 3-D point array without creating CadQuery objects.

    Useful for visualisation or non-CadQuery processing.

    Args:
        coords: Normalised airfoil coordinates.
        chord_m: Chord [m].
        twist_deg: Twist [deg].
        y_m: Spanwise position [m].
        le_sweep_offset_m: Chordwise LE offset [m].
        le_dihedral_offset_m: Vertical LE offset [m].

    Returns:
        Array of shape ``(n, 3)`` — columns are (x, y, z).
    """
    _validate_chord(chord_m)
    return _build_3d_points(
        coords, chord_m, twist_deg,
        y_m, le_sweep_offset_m, le_dihedral_offset_m,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _build_3d_points(
    coords: AirfoilCoordinates,
    chord_m: float,
    twist_deg: float,
    y_m: float,
    le_sweep: float,
    le_dihedral: float,
) -> np.ndarray:
    """Build a closed-loop 3-D point array from airfoil coordinates.

    Args:
        coords: Normalised coordinates.
        chord_m: Chord scale.
        twist_deg: Twist angle.
        y_m: Span position.
        le_sweep: LE chordwise offset.
        le_dihedral: LE vertical offset.

    Returns:
        Array of shape ``(n, 3)``.
    """
    # Selig loop: TE → upper → LE → lower → TE (skip duplicate LE)
    x_2d = np.concatenate([coords.x_upper[::-1], coords.x_lower[1:]])
    z_2d = np.concatenate([coords.y_upper[::-1], coords.y_lower[1:]])

    # Scale by chord
    x_2d = x_2d * chord_m
    z_2d = z_2d * chord_m

    # Twist about LE (x=0, z=0): rotation in the x-z plane
    if abs(twist_deg) > 1e-10:
        twist_rad = math.radians(twist_deg)
        cos_t = math.cos(twist_rad)
        sin_t = math.sin(twist_rad)
        x_rot = x_2d * cos_t + z_2d * sin_t
        z_rot = -x_2d * sin_t + z_2d * cos_t
        x_2d = x_rot
        z_2d = z_rot

    # Translate: sweep offset along x, dihedral along z
    x_2d = x_2d + le_sweep
    z_2d = z_2d + le_dihedral

    # Build 3-D: (x=chordwise, y=spanwise, z=vertical)
    n = len(x_2d)
    pts = np.column_stack([x_2d, np.full(n, y_m), z_2d])

    return pts


def _check_cq() -> None:
    """Raise if CadQuery is not available.

    Raises:
        RuntimeError: If CadQuery cannot be imported.
    """
    if not CQ_AVAILABLE:
        raise RuntimeError(
            "CadQuery is not installed. Install with: "
            "conda install -c conda-forge cadquery"
        )


def _validate_chord(chord_m: float) -> None:
    """Validate chord length.

    Args:
        chord_m: Chord in metres.

    Raises:
        ValueError: If non-positive or NaN.
    """
    if math.isnan(chord_m) or chord_m <= 0.0:
        raise ValueError(f"chord_m must be positive, got {chord_m}")
