"""
Wing loft — 3-D solid from spanwise airfoil sections.

Takes a sequence of :class:`AirfoilSection` objects (from ``cq_airfoil``)
and lofts them into a CadQuery ``Solid``, with optional STEP export.

This module uses CadQuery internally (via ``cq_airfoil``).

**Loft ordering**: sections are always sorted **root → tip** (ascending
``y_m``) before lofting.  Earlier versions had a bug where sections
could be passed in arbitrary order, producing a twisted or self-
intersecting solid.  The sort is now enforced automatically.

Typical usage:
    >>> from airfoil_config.loft import loft_wing, build_sections_from_planform
    >>> sections = build_sections_from_planform(planform, coords)
    >>> result = loft_wing(sections)
    >>> result.export_step("wing.step")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cadquery as cq
    CQ_AVAILABLE = True
except ImportError:
    CQ_AVAILABLE = False

from airfoil_config.cq_airfoil import (
    AirfoilSection,
    section_wire,
)
from airfoil_config.naca_geometry import AirfoilCoordinates
from airfoil_config.wing_planform import PlanformResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoftResult:
    """Result of a wing loft operation.

    Attributes:
        solid: CadQuery ``Solid`` (or None if CQ unavailable and
            ``dry_run=True``).
        sections: Ordered list of sections used (root → tip).
        n_sections: Number of sections lofted.
        span_m: Wing semi-span covered [m].
        is_ruled: Whether a ruled (linear) loft was used.
    """

    solid: object  # cq.Solid or None
    sections: list[AirfoilSection]
    n_sections: int
    span_m: float
    is_ruled: bool

    def export_step(self, filepath: str) -> None:
        """Export the lofted solid to a STEP file.

        Args:
            filepath: Output path (e.g. ``"wing.step"``).

        Raises:
            RuntimeError: If CadQuery is not available or solid is None.
        """
        if not CQ_AVAILABLE:
            raise RuntimeError("CadQuery required for STEP export")
        if self.solid is None:
            raise RuntimeError("No solid to export (dry_run was used)")
        cq.exporters.export(
            cq.Workplane().add(self.solid),
            filepath,
            exportType="STEP",
        )

    def export_stl(self, filepath: str, tolerance: float = 0.01) -> None:
        """Export the lofted solid to an STL file.

        Args:
            filepath: Output path.
            tolerance: Mesh tolerance [m].

        Raises:
            RuntimeError: If CadQuery is not available or solid is None.
        """
        if not CQ_AVAILABLE:
            raise RuntimeError("CadQuery required for STL export")
        if self.solid is None:
            raise RuntimeError("No solid to export (dry_run was used)")
        cq.exporters.export(
            cq.Workplane().add(self.solid),
            filepath,
            exportType="STL",
            tolerance=tolerance,
        )


# ---------------------------------------------------------------------------
# Public API — section builder
# ---------------------------------------------------------------------------
def build_sections_from_planform(
    planform: PlanformResult,
    coords: AirfoilCoordinates,
    station_indices: Optional[list[int]] = None,
    min_chord_m: float = 0.01,
) -> list[AirfoilSection]:
    """Build positioned :class:`AirfoilSection` list from a planform.

    Uses the same airfoil shape at every station, scaled by the local
    chord and twisted per the planform twist distribution.

    Args:
        planform: Wing planform (from ``wing_planform``).
        coords: Normalised airfoil coordinates (chord = 1).
        station_indices: If given, use only these planform station
            indices.  If *None*, use all stations.
        min_chord_m: Skip stations with chord below this value [m]
            (avoids degenerate tip sections).

    Returns:
        List of :class:`AirfoilSection`, sorted root → tip.
    """
    stations = planform.stations
    if station_indices is not None:
        stations = [stations[i] for i in station_indices]

    sections: list[AirfoilSection] = []
    for st in stations:
        if st.chord_m < min_chord_m:
            continue
        sections.append(AirfoilSection(
            coords=coords,
            chord_m=st.chord_m,
            twist_deg=st.twist_deg,
            y_m=st.y_m,
            le_sweep_offset_m=st.le_sweep_offset_m,
            le_dihedral_offset_m=0.0,
        ))

    # CRITICAL: sort root → tip to avoid twisted loft
    sections.sort(key=lambda s: s.y_m)

    return sections


def build_sections_varying_airfoil(
    planform: PlanformResult,
    airfoils: dict[float, AirfoilCoordinates],
    min_chord_m: float = 0.01,
) -> list[AirfoilSection]:
    """Build sections with varying airfoil shapes along the span.

    Airfoils are assigned to the nearest planform station by η
    (normalised span).  Stations without a nearby airfoil use the
    closest available one.

    Args:
        planform: Wing planform.
        airfoils: Mapping of η (0 = root, 1 = tip) → airfoil coords.
        min_chord_m: Minimum chord to include.

    Returns:
        Sorted list of :class:`AirfoilSection` (root → tip).
    """
    eta_keys = sorted(airfoils.keys())
    if not eta_keys:
        raise ValueError("airfoils dict must not be empty")

    sections: list[AirfoilSection] = []
    for st in planform.stations:
        if st.chord_m < min_chord_m:
            continue
        # Find closest η key
        closest_eta = min(eta_keys, key=lambda e: abs(e - st.eta))
        coords = airfoils[closest_eta]

        sections.append(AirfoilSection(
            coords=coords,
            chord_m=st.chord_m,
            twist_deg=st.twist_deg,
            y_m=st.y_m,
            le_sweep_offset_m=st.le_sweep_offset_m,
            le_dihedral_offset_m=0.0,
        ))

    sections.sort(key=lambda s: s.y_m)
    return sections


# ---------------------------------------------------------------------------
# Public API — lofting
# ---------------------------------------------------------------------------
def loft_wing(
    sections: list[AirfoilSection],
    ruled: bool = False,
    cap: bool = True,
    dry_run: bool = False,
) -> LoftResult:
    """Loft airfoil sections into a 3-D wing solid.

    Sections are automatically sorted by spanwise position (``y_m``)
    to guarantee correct loft ordering.

    Args:
        sections: List of :class:`AirfoilSection` (>= 2).
        ruled: If True, use ruled (linear) loft instead of smooth.
        cap: If True, cap the root and tip faces (default True).
        dry_run: If True, skip CadQuery operations and return a
            result with ``solid=None``.  Useful for testing the
            section ordering without CQ installed.

    Returns:
        :class:`LoftResult`.

    Raises:
        ValueError: If fewer than 2 sections are provided.
        RuntimeError: If CadQuery is not installed (and not dry_run).
    """
    if len(sections) < 2:
        raise ValueError(
            f"Need >= 2 sections for loft, got {len(sections)}"
        )

    # Sort root → tip (ascending y_m) — THE FIX for the ordering bug
    sorted_sections = sorted(sections, key=lambda s: s.y_m)

    span = sorted_sections[-1].y_m - sorted_sections[0].y_m

    if dry_run:
        return LoftResult(
            solid=None,
            sections=sorted_sections,
            n_sections=len(sorted_sections),
            span_m=span,
            is_ruled=ruled,
        )

    if not CQ_AVAILABLE:
        raise RuntimeError(
            "CadQuery is required for lofting. "
            "Install with: conda install -c conda-forge cadquery"
        )

    # Build wires in order
    wires = [section_wire(s) for s in sorted_sections]

    # Loft
    solid = _loft_wires(wires, ruled=ruled, cap=cap)

    return LoftResult(
        solid=solid,
        sections=sorted_sections,
        n_sections=len(sorted_sections),
        span_m=span,
        is_ruled=ruled,
    )


def loft_full_wing(
    sections: list[AirfoilSection],
    ruled: bool = False,
) -> LoftResult:
    """Loft a full wing by mirroring the semi-span.

    Creates the right semi-span from the sections, then mirrors
    about y = 0 to create the full wing.

    Args:
        sections: Semi-span sections (y_m >= 0).
        ruled: Ruled loft.

    Returns:
        :class:`LoftResult` with the full wing solid.

    Raises:
        RuntimeError: If CadQuery is not available.
    """
    if not CQ_AVAILABLE:
        raise RuntimeError("CadQuery required for full-wing loft")

    right = loft_wing(sections, ruled=ruled, cap=True)
    if right.solid is None:
        raise RuntimeError("Right semi-span loft failed")

    # Mirror about the XZ plane (y = 0)
    right_wp = cq.Workplane().add(right.solid)
    left_wp = right_wp.mirror("XZ")

    # Fuse the two halves
    full = right_wp.union(left_wp)

    return LoftResult(
        solid=full.val(),
        sections=right.sections,
        n_sections=right.n_sections * 2,
        span_m=right.span_m * 2.0,
        is_ruled=ruled,
    )


# ---------------------------------------------------------------------------
# Public API — validation
# ---------------------------------------------------------------------------
def validate_sections(sections: list[AirfoilSection]) -> list[str]:
    """Check a section list for common problems.

    Does **not** require CadQuery.

    Args:
        sections: List of sections to validate.

    Returns:
        List of warning strings (empty if all OK).
    """
    warnings: list[str] = []

    if len(sections) < 2:
        warnings.append(f"Need >= 2 sections, got {len(sections)}")
        return warnings

    sorted_secs = sorted(sections, key=lambda s: s.y_m)

    # Check for duplicate y_m
    y_vals = [s.y_m for s in sorted_secs]
    for i in range(1, len(y_vals)):
        if abs(y_vals[i] - y_vals[i - 1]) < 1e-6:
            warnings.append(
                f"Duplicate y_m={y_vals[i]:.4f} at indices {i - 1} and {i}"
            )

    # Check for zero-length chords
    for i, s in enumerate(sorted_secs):
        if s.chord_m < 1e-6:
            warnings.append(f"Section {i} has near-zero chord ({s.chord_m:.6f} m)")

    # Check for very large twist
    for i, s in enumerate(sorted_secs):
        if abs(s.twist_deg) > 20.0:
            warnings.append(
                f"Section {i} has large twist ({s.twist_deg:.1f}°)"
            )

    # Check consistent point counts
    n_pts = [len(s.coords.x_upper) for s in sorted_secs]
    if len(set(n_pts)) > 1:
        warnings.append(
            f"Inconsistent point counts across sections: {set(n_pts)}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Private — CadQuery loft engine
# ---------------------------------------------------------------------------
def _loft_wires(
    wires: list,
    ruled: bool = False,
    cap: bool = True,
) -> object:
    """Loft a list of CadQuery wires into a solid.

    Args:
        wires: Ordered list of CadQuery Wire objects.
        ruled: Ruled (linear) loft.
        cap: Cap end faces.

    Returns:
        CadQuery Solid.
    """
    builder = cq.Workplane("XY")

    # Add first wire
    builder = builder.add(wires[0])

    # Use the CadQuery loft API
    # Build a compound of wires, then loft
    wire_list = wires

    # Use OCC directly through CadQuery's Shell.loft
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

    loft_builder = BRepOffsetAPI_ThruSections(True, ruled)

    for w in wire_list:
        loft_builder.AddWire(w.wrapped)

    loft_builder.Build()

    if not loft_builder.IsDone():
        raise RuntimeError("OCC ThruSections loft failed")

    shape = loft_builder.Shape()

    from OCP.TopoDS import TopoDS

    return cq.Shape(shape)
