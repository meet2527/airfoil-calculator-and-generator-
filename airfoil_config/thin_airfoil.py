"""
Thin airfoil theory — fast aerodynamic estimates.

Computes lift coefficient, pitching-moment coefficient, zero-lift angle,
and centre-of-pressure from the mean camber line using classical
thin-airfoil-theory Fourier analysis.

Supports both analytic NACA 4-digit camber and arbitrary camber lines
supplied as arrays.  All angles are in **degrees** at the public API
and converted internally to radians.

References:
    Anderson, J.D., *Fundamentals of Aerodynamics*, 6th ed., Ch. 4.
    Abbott & Von Doenhoff, *Theory of Wing Sections*, Dover 1959.

Typical usage:
    >>> r = thin_airfoil_naca4(m_pct=2, p_pct=4, alpha_deg=4.0)
    >>> r.cl
    0.8...
    >>> r.alpha_zl_deg
    -2.0...
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThinAirfoilResult:
    """Results from thin-airfoil-theory analysis.

    Attributes:
        cl: Section lift coefficient.
        cm_le: Pitching-moment coefficient about the leading edge.
        cm_c4: Pitching-moment coefficient about the quarter-chord.
        alpha_zl_deg: Zero-lift angle of attack [deg].
        cl_alpha: Lift-curve slope [1/rad] (= 2π for thin airfoil).
        alpha_deg: Angle of attack used [deg].
        a0: Fourier coefficient A₀.
        a1: Fourier coefficient A₁.
        a2: Fourier coefficient A₂.
        xcp: Centre-of-pressure location x/c (from LE).
    """

    cl: float
    cm_le: float
    cm_c4: float
    alpha_zl_deg: float
    cl_alpha: float
    alpha_deg: float
    a0: float
    a1: float
    a2: float
    xcp: float


# ---------------------------------------------------------------------------
# Numerical Helpers
# ---------------------------------------------------------------------------
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration (replaces np.trapz removed in NumPy 2.0)."""
    # Standard trapezoidal rule implementation
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


# ---------------------------------------------------------------------------
# Public API — NACA 4-digit (analytic camber)
# ---------------------------------------------------------------------------
def thin_airfoil_naca4(
    m_pct: int,
    p_pct: int,
    alpha_deg: float = 0.0,
    n_quad: int = 200,
) -> ThinAirfoilResult:
    """Thin-airfoil analysis for a NACA 4-digit camber line.

    Args:
        m_pct: Maximum camber in percent chord (0–9).
        p_pct: Position of max camber in tenths chord (0–9).
        alpha_deg: Angle of attack [deg].
        n_quad: Number of quadrature points (default 200).

    Returns:
        :class:`ThinAirfoilResult` with all aerodynamic coefficients.

    Raises:
        ValueError: On invalid parameters.
    """
    _validate_naca4(m_pct, p_pct)

    m = m_pct / 100.0
    p = p_pct / 10.0

    # Build dyc/dx on θ-stations
    theta = np.linspace(1e-8, np.pi, n_quad)
    x = 0.5 * (1.0 - np.cos(theta))

    if m == 0.0:
        dyc = np.zeros_like(x)
    else:
        dyc = np.where(
            x <= p,
            (2.0 * m / p ** 2) * (p - x),
            (2.0 * m / (1.0 - p) ** 2) * (p - x),
        )

    return _analyse_camber_slope(dyc, theta, alpha_deg)


# ---------------------------------------------------------------------------
# Public API — arbitrary camber line
# ---------------------------------------------------------------------------
def thin_airfoil_camber(
    x_camber: np.ndarray,
    y_camber: np.ndarray,
    alpha_deg: float = 0.0,
    n_quad: int = 200,
) -> ThinAirfoilResult:
    """Thin-airfoil analysis for an arbitrary camber line.

    The camber line is resampled onto cosine-spaced stations for
    Fourier integration.

    Args:
        x_camber: Chordwise stations of the camber line, normalised
            to [0, 1].  Shape ``(n,)``.
        y_camber: Camber ordinates (y/c).  Shape ``(n,)``.
        alpha_deg: Angle of attack [deg].
        n_quad: Quadrature points (default 200).

    Returns:
        :class:`ThinAirfoilResult`.

    Raises:
        ValueError: If arrays are mismatched, too short, or not
            monotonically increasing in x.
    """
    _validate_camber_arrays(x_camber, y_camber)

    # Resample onto cosine-spaced θ
    theta = np.linspace(1e-8, np.pi, n_quad)
    x_q = 0.5 * (1.0 - np.cos(theta))

    # Interpolate camber and compute slope via finite differences
    y_q = np.interp(x_q, x_camber, y_camber)
    dyc = np.gradient(y_q, x_q)

    return _analyse_camber_slope(dyc, theta, alpha_deg)


# ---------------------------------------------------------------------------
# Public API — quick helpers
# ---------------------------------------------------------------------------
def cl_at_alpha(alpha_zl_deg: float, alpha_deg: float) -> float:
    """Compute CL = 2π(α − α_L0).

    Args:
        alpha_zl_deg: Zero-lift angle [deg].
        alpha_deg: Angle of attack [deg].

    Returns:
        Section lift coefficient.
    """
    return 2.0 * math.pi * math.radians(alpha_deg - alpha_zl_deg)


def alpha_for_cl(cl: float, alpha_zl_deg: float) -> float:
    """Compute angle of attack required for a given CL.

    α = α_L0 + CL / (2π)

    Args:
        cl: Desired lift coefficient.
        alpha_zl_deg: Zero-lift angle [deg].

    Returns:
        Required angle of attack [deg].
    """
    return alpha_zl_deg + math.degrees(cl / (2.0 * math.pi))


def fourier_coefficients(
    x_camber: np.ndarray,
    y_camber: np.ndarray,
    n_quad: int = 200,
    n_coeffs: int = 5,
) -> list[float]:
    """Compute the first N Fourier coefficients of the camber slope.

    Args:
        x_camber: Camber x-coordinates [0, 1].  Shape ``(n,)``.
        y_camber: Camber y-coordinates.  Shape ``(n,)``.
        n_quad: Quadrature points.
        n_coeffs: Number of coefficients to return (A₀ … A_{n-1}).

    Returns:
        List of Fourier coefficients ``[A0_camber, A1, ..., A_{n_coeffs-1}]``.
    """
    _validate_camber_arrays(x_camber, y_camber)

    theta = np.linspace(1e-8, np.pi, n_quad)
    x_q = 0.5 * (1.0 - np.cos(theta))
    y_q = np.interp(x_q, x_camber, y_camber)
    dyc = np.gradient(y_q, x_q)

    coeffs: list[float] = []

    # A0_camber = -(1/π) ∫ dyc/dx dθ
    a0_c = -(1.0 / np.pi) * _trapz(dyc, theta)
    coeffs.append(a0_c)

    for n in range(1, n_coeffs):
        an = (2.0 / np.pi) * _trapz(dyc * np.cos(n * theta), theta)
        coeffs.append(an)

    return coeffs


# ---------------------------------------------------------------------------
# Core analysis engine (private)
# ---------------------------------------------------------------------------
def _analyse_camber_slope(
    dyc: np.ndarray,
    theta: np.ndarray,
    alpha_deg: float,
) -> ThinAirfoilResult:
    """Run thin-airfoil Fourier analysis on a camber-slope distribution."""
    alpha_rad = math.radians(alpha_deg)

    # Fourier coefficients (camber-only parts)
    a0_c = -(1.0 / np.pi) * _trapz(dyc, theta)
    a1 = (2.0 / np.pi) * _trapz(dyc * np.cos(theta), theta)
    a2 = (2.0 / np.pi) * _trapz(dyc * np.cos(2 * theta), theta)

    # Full A0 includes angle of attack
    a0 = alpha_rad + a0_c

    # Zero-lift angle: α_L0 = -(a0_c + a1/2)
    alpha_zl_rad = -(1.0 / np.pi) * _trapz(dyc * (np.cos(theta) - 1.0), theta)
    alpha_zl_deg = math.degrees(alpha_zl_rad)

    # Aerodynamic coefficients
    cl = 2.0 * math.pi * (alpha_rad - alpha_zl_rad)
    cm_le = -(cl / 4.0) - (math.pi / 4.0) * (a1 - a2)
    cm_c4 = (math.pi / 4.0) * (a2 - a1)

    # Centre of pressure
    if abs(cl) > 1e-12:
        xcp = -cm_le / cl
    else:
        xcp = 0.25

    return ThinAirfoilResult(
        cl=cl,
        cm_le=cm_le,
        cm_c4=cm_c4,
        alpha_zl_deg=alpha_zl_deg,
        cl_alpha=2.0 * math.pi,
        alpha_deg=alpha_deg,
        a0=a0,
        a1=a1,
        a2=a2,
        xcp=xcp,
    )


# ---------------------------------------------------------------------------
# Validation (private)
# ---------------------------------------------------------------------------
def _validate_naca4(m: int, p: int) -> None:
    if not isinstance(m, int) or not 0 <= m <= 9:
        raise ValueError(f"m_pct must be int 0–9, got {m!r}")
    if not isinstance(p, int) or not 0 <= p <= 9:
        raise ValueError(f"p_pct must be int 0–9, got {p!r}")
    if m > 0 and p == 0:
        raise ValueError(f"Cambered (m_pct={m}) requires p_pct > 0")
    if m == 0 and p > 0:
        raise ValueError(f"Symmetric (m_pct=0) must have p_pct=0, got {p}")


def _validate_camber_arrays(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes must match: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"Arrays must be 1-D, got ndim={x.ndim}")
    if len(x) < 3:
        raise ValueError(f"Need >= 3 camber points, got {len(x)}")
    if not np.all(np.diff(x) > -1e-12):
        raise ValueError("x_camber must be monotonically non-decreasing")
