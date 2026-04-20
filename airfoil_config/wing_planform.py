"""
Wing planform geometry — chord and twist distributions.

Computes spanwise chord length, geometric twist, and mean aerodynamic
chord (MAC) for common planform shapes: rectangular, linearly-tapered,
and elliptical.  Also produces the station data needed by downstream
modules (``lifting_line.py``, ``loft.py``) to build the 3-D wing.

References:
    Anderson, J.D., *Fundamentals of Aerodynamics*, 6th ed., Ch. 5.
    Raymer, D.P., *Aircraft Design: A Conceptual Approach*, Ch. 4.

Typical usage:
    >>> plan = compute_planform(
    ...     wing_span_m=10.0, wing_area_m2=16.0,
    ...     taper_ratio=0.45, washout_deg=3.0, n_stations=21,
    ... )
    >>> plan.mac_m          # mean aerodynamic chord
    1.738...
    >>> plan.stations[0].chord_m   # root chord
    2.206...
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WingStation:
    """Geometry at a single spanwise station.

    Attributes:
        y_m: Spanwise coordinate from centreline [m] (0 = root).
        eta: Normalised span station y/(b/2), in [0, 1].
        chord_m: Local chord length [m].
        twist_deg: Local geometric twist angle [deg] (positive = LE up).
        le_sweep_offset_m: Chordwise offset of the local LE from the
            root LE due to sweep [m].
    """

    y_m: float
    eta: float
    chord_m: float
    twist_deg: float
    le_sweep_offset_m: float


@dataclass(frozen=True)
class PlanformResult:
    """Complete planform description.

    Attributes:
        stations: Ordered list of :class:`WingStation` from root to tip.
        mac_m: Mean aerodynamic chord [m].
        mac_y_m: Spanwise location of the MAC [m from centreline].
        wing_area_m2: Reference area (integrated) [m²].
        wing_span_m: Full span [m].
        root_chord_m: Root chord [m].
        tip_chord_m: Tip chord [m].
        taper_ratio: Tip / root chord ratio λ.
        aspect_ratio: b² / S.
        planform_type: ``"tapered"`` or ``"elliptical"``.
    """

    stations: list[WingStation]
    mac_m: float
    mac_y_m: float
    wing_area_m2: float
    wing_span_m: float
    root_chord_m: float
    tip_chord_m: float
    taper_ratio: float
    aspect_ratio: float
    planform_type: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_planform(
    wing_span_m: float,
    wing_area_m2: float,
    taper_ratio: float = 1.0,
    washout_deg: float = 0.0,
    le_sweep_deg: float = 0.0,
    n_stations: int = 21,
    cosine_spacing: bool = True,
) -> PlanformResult:
    """Compute a linearly-tapered wing planform.

    The planform is defined for a single semi-span (root → tip).
    A ``taper_ratio`` of 1.0 gives a rectangular wing; values in
    [0.3, 0.6] are typical for light aircraft.

    Args:
        wing_span_m: Full wing span [m].
        wing_area_m2: Wing reference area [m²].
        taper_ratio: Tip-to-root chord ratio λ ∈ (0, 1].
        washout_deg: Linear geometric washout (twist reduction root → tip)
            [deg].  Positive washout means the tip is twisted LE-down.
        le_sweep_deg: Leading-edge sweep angle [deg].  Positive = swept
            back.
        n_stations: Number of spanwise stations (>= 2).
        cosine_spacing: Use half-cosine spacing for better tip
            resolution (default True).

    Returns:
        :class:`PlanformResult` with station list and derived properties.

    Raises:
        ValueError: On invalid geometry parameters.
    """
    _validate_planform_inputs(
        wing_span_m, wing_area_m2, taper_ratio, n_stations,
    )

    semi_span = wing_span_m / 2.0
    eta = _spanwise_stations(n_stations, cosine_spacing)

    # Root chord from area and taper:  S = (b/2) · c_r · (1 + λ)
    root_chord = wing_area_m2 / (semi_span * (1.0 + taper_ratio))
    tip_chord = root_chord * taper_ratio

    stations: list[WingStation] = []
    for e in eta:
        y = e * semi_span
        chord = root_chord * (1.0 - e * (1.0 - taper_ratio))
        twist = -washout_deg * e  # linear washout: 0 at root, -washout at tip
        le_offset = y * math.tan(math.radians(le_sweep_deg))
        stations.append(WingStation(
            y_m=y, eta=e, chord_m=chord,
            twist_deg=twist, le_sweep_offset_m=le_offset,
        ))

    mac, mac_y = _mac_tapered(root_chord, taper_ratio, semi_span)
    area_check = _integrate_area(stations, semi_span)

    return PlanformResult(
        stations=stations,
        mac_m=mac,
        mac_y_m=mac_y,
        wing_area_m2=area_check,
        wing_span_m=wing_span_m,
        root_chord_m=root_chord,
        tip_chord_m=tip_chord,
        taper_ratio=taper_ratio,
        aspect_ratio=wing_span_m ** 2 / wing_area_m2,
        planform_type="tapered",
    )


def compute_elliptical_planform(
    wing_span_m: float,
    wing_area_m2: float,
    washout_deg: float = 0.0,
    n_stations: int = 21,
    cosine_spacing: bool = True,
) -> PlanformResult:
    """Compute an elliptical wing planform.

    The chord at each station follows c(η) = c_0 √(1 − η²) where
    c_0 is the root chord derived from the area constraint.

    Args:
        wing_span_m: Full wing span [m].
        wing_area_m2: Wing reference area [m²].
        washout_deg: Linear washout [deg].
        n_stations: Spanwise stations (>= 2).
        cosine_spacing: Half-cosine spacing (default True).

    Returns:
        :class:`PlanformResult`.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_planform_inputs(wing_span_m, wing_area_m2, 1.0, n_stations)

    semi_span = wing_span_m / 2.0
    eta = _spanwise_stations(n_stations, cosine_spacing)

    # Elliptical area:  S = (π/4) · b · c_0  →  c_0 = 4S / (π·b)
    root_chord = (4.0 * wing_area_m2) / (math.pi * wing_span_m)

    stations: list[WingStation] = []
    for e in eta:
        y = e * semi_span
        chord = root_chord * math.sqrt(max(1.0 - e ** 2, 0.0))
        twist = -washout_deg * e
        stations.append(WingStation(
            y_m=y, eta=e, chord_m=chord,
            twist_deg=twist, le_sweep_offset_m=0.0,
        ))

    mac = _mac_elliptical(root_chord)
    mac_y = _mac_y_elliptical(semi_span)
    area_check = _integrate_area(stations, semi_span)

    return PlanformResult(
        stations=stations,
        mac_m=mac,
        mac_y_m=mac_y,
        wing_area_m2=area_check,
        wing_span_m=wing_span_m,
        root_chord_m=root_chord,
        tip_chord_m=0.0,
        taper_ratio=0.0,
        aspect_ratio=wing_span_m ** 2 / wing_area_m2,
        planform_type="elliptical",
    )


def chord_at_eta(
    root_chord_m: float,
    taper_ratio: float,
    eta: float,
) -> float:
    """Compute local chord for a linearly-tapered wing.

    Args:
        root_chord_m: Root chord [m].
        taper_ratio: Tip / root chord ratio λ.
        eta: Normalised span station y/(b/2), in [0, 1].

    Returns:
        Local chord [m].

    Raises:
        ValueError: If *eta* is outside [0, 1] or inputs are invalid.
    """
    if math.isnan(eta) or eta < 0.0 or eta > 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    _validate_positive("root_chord_m", root_chord_m)
    if math.isnan(taper_ratio) or taper_ratio < 0.0 or taper_ratio > 1.0:
        raise ValueError(f"taper_ratio must be in [0, 1], got {taper_ratio}")
    return root_chord_m * (1.0 - eta * (1.0 - taper_ratio))


def mac_tapered(
    root_chord_m: float,
    taper_ratio: float,
) -> float:
    """Compute MAC for a linearly-tapered planform (analytic).

    MAC = (2/3) · c_r · (1 + λ + λ²) / (1 + λ)

    Args:
        root_chord_m: Root chord [m].
        taper_ratio: Tip / root chord ratio λ ∈ [0, 1].

    Returns:
        Mean aerodynamic chord [m].

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_positive("root_chord_m", root_chord_m)
    if math.isnan(taper_ratio) or taper_ratio < 0.0 or taper_ratio > 1.0:
        raise ValueError(f"taper_ratio must be in [0, 1], got {taper_ratio}")
    lam = taper_ratio
    return (2.0 / 3.0) * root_chord_m * (1 + lam + lam ** 2) / (1 + lam)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _spanwise_stations(n: int, use_cosine: bool) -> np.ndarray:
    """Generate normalised span stations η ∈ [0, 1].

    Args:
        n: Number of stations.
        use_cosine: Half-cosine distribution for tip clustering.

    Returns:
        1-D array of η values.  Shape ``(n,)``.
    """
    if use_cosine:
        theta = np.linspace(0.0, np.pi / 2.0, n)
        return np.sin(theta)  # half-cosine: clusters near tip (η = 1)
    return np.linspace(0.0, 1.0, n)


def _mac_tapered(
    root_chord: float, taper_ratio: float, semi_span: float,
) -> tuple[float, float]:
    """Analytic MAC and its spanwise location for tapered planform.

    Args:
        root_chord: Root chord [m].
        taper_ratio: λ.
        semi_span: Semi-span [m].

    Returns:
        ``(mac, y_mac)`` in metres.
    """
    lam = taper_ratio
    mac = (2.0 / 3.0) * root_chord * (1 + lam + lam ** 2) / (1 + lam)
    y_mac = (semi_span / 3.0) * (1 + 2 * lam) / (1 + lam)
    return mac, y_mac


def _mac_elliptical(root_chord: float) -> float:
    """MAC for an elliptical planform.

    MAC = (π/4) · c_0 ≈ 0.7854 · c_0

    Args:
        root_chord: Root (maximum) chord [m].

    Returns:
        MAC [m].
    """
    return (math.pi / 4.0) * root_chord


def _mac_y_elliptical(semi_span: float) -> float:
    """Spanwise location of MAC for an elliptical planform.

    y_MAC = (4/(3π)) · (b/2)

    Args:
        semi_span: Semi-span [m].

    Returns:
        y_MAC [m].
    """
    return (4.0 / (3.0 * math.pi)) * semi_span


def _integrate_area(
    stations: list[WingStation], semi_span: float,
) -> float:
    """Numerically integrate the planform area (full wing = 2 × semi-span).

    Uses the trapezoidal rule on the station chords.

    Args:
        stations: Ordered root-to-tip stations.
        semi_span: Semi-span [m].

    Returns:
        Full-wing reference area [m²].
    """
    y_arr = np.array([s.y_m for s in stations])
    c_arr = np.array([s.chord_m for s in stations])
    # Calculate area via trapezoidal integration
    # (replaces np.trapz removed in NumPy 2.0)
    semi_area = float(np.sum((c_arr[1:] + c_arr[:-1]) * np.diff(y_arr)) / 2.0)
    return 2.0 * semi_area


def _validate_planform_inputs(
    span: float, area: float, taper: float, n: int,
) -> None:
    """Validate common planform parameters.

    Args:
        span: Wing span [m].
        area: Wing area [m²].
        taper: Taper ratio.
        n: Number of stations.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_positive("wing_span_m", span)
    _validate_positive("wing_area_m2", area)
    if math.isnan(taper) or taper < 0.0 or taper > 1.0:
        raise ValueError(f"taper_ratio must be in [0, 1], got {taper}")
    if n < 2:
        raise ValueError(f"n_stations must be >= 2, got {n}")


def _validate_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` if *value* is not a positive finite number.

    Args:
        name: Parameter name.
        value: Value to check.

    Raises:
        ValueError: On NaN, inf, zero, or negative.
    """
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite, got inf")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
