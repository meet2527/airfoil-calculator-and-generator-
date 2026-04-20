"""
NACA 4-digit and 5-digit airfoil coordinate generator.

Generates normalised (chord = 1) airfoil coordinates using standard
NACA parameterisation.  Supports cosine point spacing, open/closed
trailing edges, and both split-surface and single-loop output formats.

References:
    Abbott & Von Doenhoff, *Theory of Wing Sections*, Dover 1959.
    Jacobs, Ward & Pinkerton, NACA Report 460, 1933.

Typical usage:
    >>> coords = naca4(m_pct=2, p_pct=4, t_pct=12)
    >>> coords.designation
    'NACA 2412'
    >>> coords.x_upper.shape
    (100,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AirfoilCoordinates:
    """Split-surface airfoil coordinates (chord = 1).

    Upper and lower surfaces are ordered from leading edge (x = 0) to
    trailing edge (x ≈ 1).

    Attributes:
        x_upper: Upper-surface x-coordinates.  Shape ``(n,)``.
        y_upper: Upper-surface y-coordinates.  Shape ``(n,)``.
        x_lower: Lower-surface x-coordinates.  Shape ``(n,)``.
        y_lower: Lower-surface y-coordinates.  Shape ``(n,)``.
        x_camber: Mean camber line x-coordinates.  Shape ``(n,)``.
        y_camber: Mean camber line y-coordinates.  Shape ``(n,)``.
        designation: NACA designation string, e.g. ``"NACA 2412"``.
    """

    x_upper: np.ndarray
    y_upper: np.ndarray
    x_lower: np.ndarray
    y_lower: np.ndarray
    x_camber: np.ndarray
    y_camber: np.ndarray
    designation: str


@dataclass(frozen=True)
class AirfoilSingleSurface:
    """Closed-loop coordinates in Selig order.

    Points run TE → upper → LE → lower → TE.  The leading-edge point
    appears once (not duplicated).

    Attributes:
        x: x-coordinates of the closed loop.  Shape ``(2n-1,)``.
        y: y-coordinates.  Shape matches *x*.
        designation: NACA designation string.
    """

    x: np.ndarray
    y: np.ndarray
    designation: str


# ---------------------------------------------------------------------------
# NACA 5-digit camber-line coefficient tables
# ---------------------------------------------------------------------------
# Standard (S = 0): key = P digit → (m, k1)
_NACA5_STD: dict[int, tuple[float, float]] = {
    1: (0.0580, 361.400),
    2: (0.1260, 51.640),
    3: (0.2025, 15.957),
    4: (0.2900, 6.643),
    5: (0.3910, 3.230),
}

# Reflex (S = 1): key = P digit → (m, k1, k2/k1)
_NACA5_RFX: dict[int, tuple[float, float, float]] = {
    2: (0.1300, 51.990, 0.000764),
    3: (0.2170, 15.793, 0.00677),
    4: (0.3180, 6.520, 0.0303),
    5: (0.4410, 3.191, 0.1355),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cosine_spacing(n_points: int) -> np.ndarray:
    """Generate cosine-spaced x on [0, 1] for better LE/TE resolution.

    Args:
        n_points: Number of points (>= 2).

    Returns:
        1-D array of x from 0 to 1.  Shape ``(n_points,)``.

    Raises:
        ValueError: If *n_points* < 2.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def naca4(
    m_pct: int,
    p_pct: int,
    t_pct: int,
    n_points: int = 100,
    closed_te: bool = True,
) -> AirfoilCoordinates:
    """Generate NACA 4-digit airfoil coordinates.

    Args:
        m_pct: Maximum camber in percent chord (0–9, first digit).
        p_pct: Position of max camber in tenths chord (0–9, second digit).
        t_pct: Max thickness in percent chord (1–99, last two digits).
        n_points: Points per surface side (default 100).
        closed_te: Sharp closed trailing edge (default True).

    Returns:
        :class:`AirfoilCoordinates` with upper / lower / camber arrays.

    Raises:
        ValueError: On invalid parameters.
    """
    _validate_naca4(m_pct, p_pct, t_pct, n_points)

    m = m_pct / 100.0
    p = p_pct / 10.0
    t = t_pct / 100.0

    x = cosine_spacing(n_points)
    yt = _thickness(x, t, closed_te)
    yc, dyc = _camber4(x, m, p)
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return AirfoilCoordinates(
        x_upper=xu, y_upper=yu,
        x_lower=xl, y_lower=yl,
        x_camber=x.copy(), y_camber=yc,
        designation=f"NACA {m_pct}{p_pct}{t_pct:02d}",
    )


def naca5(
    l_digit: int,
    p_digit: int,
    s_digit: int,
    t_pct: int,
    n_points: int = 100,
    closed_te: bool = True,
) -> AirfoilCoordinates:
    """Generate NACA 5-digit airfoil coordinates.

    Designation digits LPSTT encode:
    L × 0.15 = design CL, P × 0.05 = max-camber position,
    S = 0 standard / 1 reflex, TT = thickness %.

    Args:
        l_digit: First digit (1–6).
        p_digit: Second digit (1–5).
        s_digit: 0 (standard) or 1 (reflex).
        t_pct: Thickness percent (1–99).
        n_points: Points per side.
        closed_te: Close trailing edge.

    Returns:
        :class:`AirfoilCoordinates`.

    Raises:
        ValueError: On invalid parameters.
    """
    _validate_naca5(l_digit, p_digit, s_digit, t_pct, n_points)

    t = t_pct / 100.0
    cl_d = l_digit * 0.15

    x = cosine_spacing(n_points)
    yt = _thickness(x, t, closed_te)

    if s_digit == 0:
        yc, dyc = _camber5_std(x, p_digit, cl_d)
    else:
        yc, dyc = _camber5_rfx(x, p_digit, cl_d)

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return AirfoilCoordinates(
        x_upper=xu, y_upper=yu,
        x_lower=xl, y_lower=yl,
        x_camber=x.copy(), y_camber=yc,
        designation=f"NACA {l_digit}{p_digit}{s_digit}{t_pct:02d}",
    )


def naca_from_string(
    designation: str,
    n_points: int = 100,
    closed_te: bool = True,
) -> AirfoilCoordinates:
    """Parse a NACA designation string and generate coordinates.

    Accepts ``"NACA 2412"``, ``"naca23012"``, ``"0012"``, etc.

    Args:
        designation: NACA designation string.
        n_points: Points per side.
        closed_te: Close the trailing edge.

    Returns:
        :class:`AirfoilCoordinates`.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    digits = _parse_designation(designation)

    if len(digits) == 4:
        return naca4(
            int(digits[0]), int(digits[1]), int(digits[2:]),
            n_points=n_points, closed_te=closed_te,
        )
    if len(digits) == 5:
        return naca5(
            int(digits[0]), int(digits[1]), int(digits[2]),
            int(digits[3:]),
            n_points=n_points, closed_te=closed_te,
        )
    raise ValueError(
        f"Expected 4 or 5 digits in {designation!r}, got {len(digits)}"
    )


def to_single_surface(coords: AirfoilCoordinates) -> AirfoilSingleSurface:
    """Merge split surfaces into a single Selig-format loop.

    TE → upper (reversed) → LE → lower → TE.  The LE point is shared.

    Args:
        coords: Split-surface coordinates.

    Returns:
        :class:`AirfoilSingleSurface`.
    """
    x = np.concatenate([coords.x_upper[::-1], coords.x_lower[1:]])
    y = np.concatenate([coords.y_upper[::-1], coords.y_lower[1:]])
    return AirfoilSingleSurface(x=x, y=y, designation=coords.designation)


def max_thickness_at(coords: AirfoilCoordinates) -> tuple[float, float]:
    """Find chordwise location and value of maximum thickness.

    Thickness is the vertical distance between upper and lower surfaces.

    Args:
        coords: Airfoil coordinates.

    Returns:
        ``(x_location, thickness)`` both normalised by chord.
    """
    yl_interp = np.interp(coords.x_upper, coords.x_lower, coords.y_lower)
    thick = coords.y_upper - yl_interp
    idx = int(np.argmax(thick))
    return float(coords.x_upper[idx]), float(thick[idx])


# ---------------------------------------------------------------------------
# Private — thickness distribution
# ---------------------------------------------------------------------------
def _thickness(x: np.ndarray, t: float, closed_te: bool) -> np.ndarray:
    """NACA thickness distribution yt(x).

    Args:
        x: Chordwise stations.  Shape ``(n,)``.
        t: Max thickness as chord fraction.
        closed_te: Use modified last coefficient for closed TE.

    Returns:
        Half-thickness array.  Shape ``(n,)``.
    """
    a4 = -0.1036 if closed_te else -0.1015
    xs = np.maximum(x, 0.0)  # guard against tiny negatives
    return 5.0 * t * (
        0.2969 * np.sqrt(xs)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        + a4 * x ** 4
    )


# ---------------------------------------------------------------------------
# Private — NACA 4-digit camber
# ---------------------------------------------------------------------------
def _camber4(
    x: np.ndarray, m: float, p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line and slope for NACA 4-digit.

    Args:
        x: Chordwise stations.  Shape ``(n,)``.
        m: Max camber (chord fraction).
        p: Max camber position (chord fraction).

    Returns:
        ``(yc, dyc_dx)`` each shape ``(n,)``.
    """
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if m == 0.0 or p == 0.0:
        return yc, dyc

    fore = x <= p
    aft = ~fore

    yc[fore] = (m / p ** 2) * (2.0 * p * x[fore] - x[fore] ** 2)
    dyc[fore] = (2.0 * m / p ** 2) * (p - x[fore])

    denom = (1.0 - p) ** 2
    yc[aft] = (m / denom) * ((1.0 - 2.0 * p) + 2.0 * p * x[aft] - x[aft] ** 2)
    dyc[aft] = (2.0 * m / denom) * (p - x[aft])

    return yc, dyc


# ---------------------------------------------------------------------------
# Private — NACA 5-digit camber (standard + reflex)
# ---------------------------------------------------------------------------
def _camber5_std(
    x: np.ndarray, p_digit: int, cl_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard (non-reflex) NACA 5-digit camber line.

    Args:
        x: Chordwise stations.  Shape ``(n,)``.
        p_digit: Second designation digit (1–5).
        cl_d: Design lift coefficient.

    Returns:
        ``(yc, dyc_dx)`` each shape ``(n,)``.
    """
    m, k1 = _NACA5_STD[p_digit]
    k = k1 * (cl_d / 0.3)  # table is for CL = 0.3

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    fore = x <= m
    aft = ~fore
    xf = x[fore]

    yc[fore] = (k / 6.0) * (xf ** 3 - 3.0 * m * xf ** 2 + m ** 2 * (3.0 - m) * xf)
    dyc[fore] = (k / 6.0) * (3.0 * xf ** 2 - 6.0 * m * xf + m ** 2 * (3.0 - m))

    yc[aft] = (k * m ** 3 / 6.0) * (1.0 - x[aft])
    dyc[aft] = -(k * m ** 3 / 6.0)

    return yc, dyc


def _camber5_rfx(
    x: np.ndarray, p_digit: int, cl_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflex NACA 5-digit camber line.

    Args:
        x: Chordwise stations.  Shape ``(n,)``.
        p_digit: Second designation digit (2–5).
        cl_d: Design lift coefficient.

    Returns:
        ``(yc, dyc_dx)`` each shape ``(n,)``.
    """
    m, k1, k2k1 = _NACA5_RFX[p_digit]
    k = k1 * (cl_d / 0.3)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    fore = x <= m
    aft = ~fore

    xf = x[fore]
    yc[fore] = (k / 6.0) * (
        (xf - m) ** 3 - k2k1 * (1.0 - m) ** 3 * xf - m ** 3 * xf + m ** 3
    )
    dyc[fore] = (k / 6.0) * (
        3.0 * (xf - m) ** 2 - k2k1 * (1.0 - m) ** 3 - m ** 3
    )

    xa = x[aft]
    yc[aft] = (k / 6.0) * (
        k2k1 * (xa - m) ** 3 - k2k1 * (1.0 - m) ** 3 * xa
        - m ** 3 * xa + m ** 3
    )
    dyc[aft] = (k / 6.0) * (
        3.0 * k2k1 * (xa - m) ** 2 - k2k1 * (1.0 - m) ** 3 - m ** 3
    )

    return yc, dyc


# ---------------------------------------------------------------------------
# Private — parsing & validation
# ---------------------------------------------------------------------------
def _parse_designation(designation: str) -> str:
    """Extract digit string from a NACA designation.

    Args:
        designation: e.g. ``"NACA 2412"``, ``"0012"``.

    Returns:
        Digit-only string, e.g. ``"2412"``.

    Raises:
        ValueError: On unparseable input.
    """
    s = designation.strip().upper().replace(" ", "").replace("-", "")
    if s.startswith("NACA"):
        s = s[4:]
    if not s.isdigit() or len(s) not in (4, 5):
        raise ValueError(
            f"Invalid NACA designation {designation!r}: "
            f"expected 4 or 5 digits, got {s!r}"
        )
    return s


def _validate_naca4(m: int, p: int, t: int, n: int) -> None:
    """Validate NACA 4-digit parameters.

    Args:
        m: Max camber percent (0–9).
        p: Max camber position tenths (0–9).
        t: Thickness percent (1–99).
        n: Number of points.

    Raises:
        ValueError: On invalid parameters.
    """
    if not isinstance(m, int) or not 0 <= m <= 9:
        raise ValueError(f"m_pct must be int 0–9, got {m!r}")
    if not isinstance(p, int) or not 0 <= p <= 9:
        raise ValueError(f"p_pct must be int 0–9, got {p!r}")
    if not isinstance(t, int) or not 1 <= t <= 99:
        raise ValueError(f"t_pct must be int 1–99, got {t!r}")
    if m > 0 and p == 0:
        raise ValueError(f"Cambered airfoil (m_pct={m}) requires p_pct > 0")
    if m == 0 and p > 0:
        raise ValueError(f"Symmetric airfoil (m_pct=0) must have p_pct=0, got {p}")
    if n < 2:
        raise ValueError(f"n_points must be >= 2, got {n}")


def _validate_naca5(l: int, p: int, s: int, t: int, n: int) -> None:
    """Validate NACA 5-digit parameters.

    Args:
        l: First digit (1–6).
        p: Second digit (1–5).
        s: Third digit (0 or 1).
        t: Thickness percent (1–99).
        n: Number of points.

    Raises:
        ValueError: On invalid parameters.
    """
    if not isinstance(l, int) or not 1 <= l <= 6:
        raise ValueError(f"l_digit must be int 1–6, got {l!r}")
    if not isinstance(p, int) or not 1 <= p <= 5:
        raise ValueError(f"p_digit must be int 1–5, got {p!r}")
    if not isinstance(s, int) or s not in (0, 1):
        raise ValueError(f"s_digit must be 0 or 1, got {s!r}")
    if not isinstance(t, int) or not 1 <= t <= 99:
        raise ValueError(f"t_pct must be int 1–99, got {t!r}")
    if s == 1 and p == 1:
        raise ValueError("Reflex camber (s_digit=1) not defined for p_digit=1")
    if n < 2:
        raise ValueError(f"n_points must be >= 2, got {n}")
