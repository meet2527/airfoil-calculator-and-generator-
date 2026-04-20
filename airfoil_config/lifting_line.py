"""
Prandtl lifting-line theory (LLT) solver.

Solves the monoplane equation for the spanwise circulation
distribution Γ(y) of a finite wing, yielding:

* spanwise lift distribution
* total lift coefficient CL
* induced drag coefficient CDi
* Oswald / span efficiency factor e

Supports arbitrary planforms (via :class:`WingStation` from
``wing_planform``) and per-station section aerodynamic data
(cl_alpha, alpha_zl from ``thin_airfoil``).

References:
    Anderson, J.D., *Fundamentals of Aerodynamics*, 6th ed., Ch. 5.
    Phillips, W.F., *Mechanics of Flight*, 2nd ed., Ch. 1.

Typical usage:
    >>> result = solve_lifting_line(
    ...     planform=planform_result,
    ...     alpha_root_deg=5.0,
    ...     cl_alpha=2*math.pi,
    ...     alpha_zl_deg=-2.0,
    ... )
    >>> result.cl_wing
    0.45...
    >>> result.cdi
    0.008...
    >>> result.span_efficiency
    0.97...
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from airfoil_config.wing_planform import PlanformResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiftingLineResult:
    """Results from the lifting-line solver.

    Attributes:
        cl_wing: Total wing lift coefficient.
        cdi: Induced drag coefficient.
        span_efficiency: Oswald span efficiency factor e (1.0 = elliptic).
        cl_distribution: Section CL at each θ station.  Shape ``(n,)``.
        gamma_distribution: Circulation Γ/(V∞·b/2) at each station.
            Shape ``(n,)``.
        alpha_i_distribution: Induced angle of attack [rad] at each
            station.  Shape ``(n,)``.
        theta_stations: θ stations used (0, π).  Shape ``(n,)``.
        eta_stations: Normalised span η = cos(θ) at each station.
            Shape ``(n,)``.
        fourier_coefficients: Fourier sine-series coefficients A_n.
            Shape ``(n,)``.
        alpha_root_deg: Root geometric angle of attack [deg].
        wing_area_m2: Reference area used [m²].
        wing_span_m: Wing span used [m].
        aspect_ratio: Aspect ratio b²/S.
    """

    cl_wing: float
    cdi: float
    span_efficiency: float
    cl_distribution: np.ndarray
    gamma_distribution: np.ndarray
    alpha_i_distribution: np.ndarray
    theta_stations: np.ndarray
    eta_stations: np.ndarray
    fourier_coefficients: np.ndarray
    alpha_root_deg: float
    wing_area_m2: float
    wing_span_m: float
    aspect_ratio: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_lifting_line(
    planform: PlanformResult,
    alpha_root_deg: float,
    cl_alpha: float | np.ndarray = 2.0 * math.pi,
    alpha_zl_deg: float | np.ndarray = 0.0,
    n_terms: int = 30,
) -> LiftingLineResult:
    """Solve the Prandtl lifting-line equation.

    Uses N Fourier sine-series terms to represent the circulation
    distribution Γ(θ) = 2bV∞ Σ Aₙ sin(nθ).

    The geometric angle of attack at each station is computed from the
    root angle and the planform twist distribution.

    Args:
        planform: Wing planform from :func:`compute_planform`.
        alpha_root_deg: Root angle of attack [deg].
        cl_alpha: Section lift-curve slope [1/rad].  Scalar for uniform,
            or array of shape ``(n_terms,)`` for per-station values.
        alpha_zl_deg: Section zero-lift angle [deg].  Scalar or array.
        n_terms: Number of Fourier terms (default 30, must be >= 3).

    Returns:
        :class:`LiftingLineResult`.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(planform, alpha_root_deg, n_terms)

    b = planform.wing_span_m
    s = planform.wing_area_m2
    ar = planform.aspect_ratio
    semi = b / 2.0

    # --- Build θ stations (exclude 0 and π where sin = 0) ---------------
    theta = np.linspace(np.pi / (2 * n_terms), np.pi * (1 - 1 / (2 * n_terms)), n_terms)
    eta = np.cos(theta)  # η = cos(θ), 1 at root, 0 at tip

    # --- Chord and twist at each station ---------------------------------
    chord = np.array([
        _interp_chord(planform, abs(e)) for e in eta
    ])
    twist_deg = np.array([
        _interp_twist(planform, abs(e)) for e in eta
    ])

    # Geometric angle of attack at each station [rad]
    alpha_geom = np.radians(alpha_root_deg + twist_deg)

    # Section properties (broadcast scalar → array)
    if isinstance(cl_alpha, (int, float)):
        a_sec = np.full(n_terms, float(cl_alpha))
    else:
        a_sec = np.asarray(cl_alpha, dtype=np.float64)
        if len(a_sec) != n_terms:
            raise ValueError(
                f"cl_alpha array length {len(a_sec)} != n_terms {n_terms}"
            )

    if isinstance(alpha_zl_deg, (int, float)):
        azl = np.full(n_terms, math.radians(float(alpha_zl_deg)))
    else:
        azl = np.radians(np.asarray(alpha_zl_deg, dtype=np.float64))
        if len(azl) != n_terms:
            raise ValueError(
                f"alpha_zl_deg array length {len(azl)} != n_terms {n_terms}"
            )

    # --- Assemble the linear system  A · x = b --------------------------
    # The monoplane equation at station i:
    #   Σ_n Aₙ [ sin(nθᵢ) (n·μᵢ + sin(θᵢ)) ] = μᵢ sin(θᵢ) (α - α_L0)ᵢ
    # where μᵢ = chord_i · a_sec_i / (4 · semi)

    mu = chord * a_sec / (4.0 * semi)

    mat = np.zeros((n_terms, n_terms))
    rhs = np.zeros(n_terms)

    for i in range(n_terms):
        sin_th = math.sin(theta[i])
        for n in range(n_terms):
            nn = n + 1  # Fourier index starts at 1
            sin_n_th = math.sin(nn * theta[i])
            mat[i, n] = sin_n_th * (nn * mu[i] + sin_th)
        rhs[i] = mu[i] * sin_th * (alpha_geom[i] - azl[i])

    # --- Solve for Fourier coefficients ----------------------------------
    coeffs = np.linalg.solve(mat, rhs)  # Aₙ, n = 1..N

    # --- Post-process ----------------------------------------------------
    # CL = π · AR · A₁
    cl_wing = math.pi * ar * coeffs[0]

    # CDi = π · AR · Σ n · Aₙ²
    ns = np.arange(1, n_terms + 1, dtype=np.float64)
    cdi = math.pi * ar * float(np.sum(ns * coeffs ** 2))

    # Span efficiency: CDi = CL² / (π · AR · e)
    if abs(cdi) > 1e-15 and abs(cl_wing) > 1e-15:
        e_span = cl_wing ** 2 / (math.pi * ar * cdi)
    else:
        e_span = 1.0

    # Circulation distribution: Γ/(V∞·b/2) = 2 Σ Aₙ sin(nθ)
    gamma = np.zeros(n_terms)
    for i in range(n_terms):
        gamma[i] = 2.0 * float(
            np.sum(coeffs * np.sin(ns * theta[i]))
        )

    # Section CL: cl_section = 2·Γ / c  (where Γ is scaled by V∞·semi)
    cl_section = np.where(chord > 1e-12, 2.0 * gamma * semi / chord, 0.0)

    # Induced angle: αᵢ = Σ n·Aₙ sin(nθ) / sin(θ)
    alpha_i = np.zeros(n_terms)
    for i in range(n_terms):
        sin_th = math.sin(theta[i])
        if abs(sin_th) > 1e-12:
            alpha_i[i] = float(
                np.sum(ns * coeffs * np.sin(ns * theta[i]))
            ) / sin_th

    return LiftingLineResult(
        cl_wing=cl_wing,
        cdi=cdi,
        span_efficiency=e_span,
        cl_distribution=cl_section,
        gamma_distribution=gamma,
        alpha_i_distribution=alpha_i,
        theta_stations=theta,
        eta_stations=eta,
        fourier_coefficients=coeffs,
        alpha_root_deg=alpha_root_deg,
        wing_area_m2=s,
        wing_span_m=b,
        aspect_ratio=ar,
    )


def compute_cdi_from_cl(
    cl: float,
    aspect_ratio: float,
    span_efficiency: float = 1.0,
) -> float:
    """Quick induced-drag estimate from CL, AR, and e.

    CDi = CL² / (π · AR · e)

    Args:
        cl: Wing lift coefficient.
        aspect_ratio: Geometric aspect ratio b²/S.
        span_efficiency: Oswald span efficiency (default 1.0).

    Returns:
        Induced drag coefficient.

    Raises:
        ValueError: If inputs are non-positive where required.
    """
    if aspect_ratio <= 0.0 or math.isnan(aspect_ratio):
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}")
    if span_efficiency <= 0.0 or math.isnan(span_efficiency):
        raise ValueError(
            f"span_efficiency must be positive, got {span_efficiency}"
        )
    return cl ** 2 / (math.pi * aspect_ratio * span_efficiency)


def compute_oswald_factor(
    cl: float,
    cdi: float,
    aspect_ratio: float,
) -> float:
    """Compute the Oswald span efficiency from CL and CDi.

    e = CL² / (π · AR · CDi)

    Args:
        cl: Wing lift coefficient.
        cdi: Induced drag coefficient.
        aspect_ratio: Aspect ratio.

    Returns:
        Span efficiency factor.

    Raises:
        ValueError: If CDi or AR is non-positive.
    """
    if cdi <= 0.0 or math.isnan(cdi):
        raise ValueError(f"cdi must be positive, got {cdi}")
    if aspect_ratio <= 0.0 or math.isnan(aspect_ratio):
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}")
    return cl ** 2 / (math.pi * aspect_ratio * cdi)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _interp_chord(planform: PlanformResult, eta: float) -> float:
    """Interpolate chord at a normalised span station.

    Args:
        planform: Planform data.
        eta: Normalised station in [0, 1].

    Returns:
        Local chord [m].
    """
    etas = np.array([s.eta for s in planform.stations])
    chords = np.array([s.chord_m for s in planform.stations])
    return float(np.interp(eta, etas, chords))


def _interp_twist(planform: PlanformResult, eta: float) -> float:
    """Interpolate twist at a normalised span station.

    Args:
        planform: Planform data.
        eta: Normalised station in [0, 1].

    Returns:
        Local twist [deg].
    """
    etas = np.array([s.eta for s in planform.stations])
    twists = np.array([s.twist_deg for s in planform.stations])
    return float(np.interp(eta, etas, twists))


def _validate_inputs(
    planform: PlanformResult,
    alpha_root_deg: float,
    n_terms: int,
) -> None:
    """Validate LLT inputs.

    Args:
        planform: Planform data.
        alpha_root_deg: Root α.
        n_terms: Number of Fourier terms.

    Raises:
        ValueError: On invalid inputs.
    """
    if math.isnan(alpha_root_deg):
        raise ValueError("alpha_root_deg must not be NaN")
    if n_terms < 3:
        raise ValueError(f"n_terms must be >= 3, got {n_terms}")
    if len(planform.stations) < 2:
        raise ValueError(
            f"Planform must have >= 2 stations, got {len(planform.stations)}"
        )
    if planform.wing_span_m <= 0:
        raise ValueError(f"wing_span must be positive, got {planform.wing_span_m}")
