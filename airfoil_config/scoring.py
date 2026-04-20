"""
Airfoil ranking and scoring from polar data.

Scores candidate airfoils against :class:`WingRequirements` using
XFOIL polar data.  Evaluates multiple criteria — CL match, L/D at
cruise, drag level, Cm behaviour, stall margin — and computes a
weighted composite score.

Designed to consume :class:`XfoilPolar` objects either fresh from
``xfoil_runner`` or cached via ``polar_db``.

Typical usage:
    >>> metrics = compute_polar_metrics(polar, reqs)
    >>> metrics.ld_max
    85.3
    >>> ranked = rank_airfoils(polars_dict, reqs)
    >>> ranked[0].designation
    'NACA 2412'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from airfoil_config.requirements import WingRequirements
from airfoil_config.xfoil_runner import XfoilPolar


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarMetrics:
    """Aerodynamic performance metrics extracted from a polar.

    Attributes:
        designation: Airfoil identifier.
        cl_at_cruise_alpha: CL at the cruise angle of attack.
        cd_at_cruise: CD at cruise conditions.
        ld_at_cruise: Lift-to-drag ratio at cruise.
        ld_max: Maximum L/D across the polar.
        alpha_ld_max_deg: α at maximum L/D [deg].
        cl_max: Maximum CL in the polar (proxy for stall CL).
        alpha_cl_max_deg: α at CL_max [deg].
        cd_min: Minimum drag coefficient.
        cm_at_cruise: Pitching moment at cruise α.
        alpha_zl_deg: Estimated zero-lift angle [deg] (linear interpolation).
        cl_alpha_rad: Estimated lift-curve slope [1/rad].
        stall_margin_cl: CL_max − CL_cruise.
        endurance_param: CL^1.5 / CD at cruise (endurance figure of merit).
    """

    designation: str
    cl_at_cruise_alpha: float
    cd_at_cruise: float
    ld_at_cruise: float
    ld_max: float
    alpha_ld_max_deg: float
    cl_max: float
    alpha_cl_max_deg: float
    cd_min: float
    cm_at_cruise: float
    alpha_zl_deg: float
    cl_alpha_rad: float
    stall_margin_cl: float
    endurance_param: float


@dataclass(frozen=True)
class ScoredAirfoil:
    """A scored and ranked airfoil.

    Attributes:
        designation: Airfoil identifier.
        total_score: Composite score in [0, 1], higher = better.
        sub_scores: Dict of sub-criterion name → score in [0, 1].
        metrics: Full :class:`PolarMetrics`.
        rank: 1-based rank (1 = best).
    """

    designation: str
    total_score: float
    sub_scores: dict[str, float]
    metrics: PolarMetrics
    rank: int


# ---------------------------------------------------------------------------
# Scoring weights (module-level defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringWeights:
    """Relative weights for each scoring criterion.

    All weights are normalised internally so they need not sum to 1.

    Attributes:
        ld_cruise: Weight for L/D at cruise.
        cl_match: Weight for CL match to required cruise CL.
        cd_level: Weight for low absolute drag.
        stall_margin: Weight for CL_max headroom above cruise CL.
        cm_magnitude: Weight for low |Cm| (pitch stability / trim drag).
        endurance: Weight for CL^1.5/CD endurance parameter.
    """

    ld_cruise: float = 0.25
    cl_match: float = 0.25
    cd_level: float = 0.15
    stall_margin: float = 0.15
    cm_magnitude: float = 0.10
    endurance: float = 0.10


DEFAULT_WEIGHTS = ScoringWeights()


# ---------------------------------------------------------------------------
# Public API — metrics extraction
# ---------------------------------------------------------------------------
def compute_polar_metrics(
    polar: XfoilPolar,
    reqs: WingRequirements,
) -> PolarMetrics:
    """Extract performance metrics from an XFOIL polar.

    The cruise angle of attack is found by interpolating the polar's
    CL vs α curve to the required cruise CL.

    Args:
        polar: XFOIL polar data (must have >= 2 converged points).
        reqs: Wing requirements (provides ``required_cl_cruise``).

    Returns:
        :class:`PolarMetrics`.

    Raises:
        ValueError: If the polar has fewer than 2 points.
    """
    if polar.converged_count < 2:
        raise ValueError(
            f"Polar for {polar.designation!r} has {polar.converged_count} "
            f"point(s) — need >= 2 for metrics"
        )

    alpha = polar.alpha_deg
    cl = polar.cl
    cd = polar.cd
    cm = polar.cm
    target_cl = reqs.required_cl_cruise

    # --- Cruise-point interpolation --------------------------------------
    alpha_cruise = _interp_alpha_for_cl(alpha, cl, target_cl)
    cd_cruise = float(np.interp(alpha_cruise, alpha, cd))
    cm_cruise = float(np.interp(alpha_cruise, alpha, cm))
    ld_cruise = target_cl / cd_cruise if cd_cruise > 1e-12 else 0.0

    # --- L/D envelope ----------------------------------------------------
    ld = cl / np.maximum(cd, 1e-12)
    idx_ldmax = int(np.argmax(ld))
    ld_max = float(ld[idx_ldmax])
    alpha_ldmax = float(alpha[idx_ldmax])

    # --- CL_max ----------------------------------------------------------
    idx_clmax = int(np.argmax(cl))
    cl_max = float(cl[idx_clmax])
    alpha_clmax = float(alpha[idx_clmax])

    # --- CD_min ----------------------------------------------------------
    cd_min = float(np.min(cd))

    # --- Zero-lift angle (linear interpolation) --------------------------
    alpha_zl = _interp_alpha_for_cl(alpha, cl, 0.0)

    # --- Lift-curve slope (linear fit over low-α range) ------------------
    cl_alpha = _estimate_cl_alpha(alpha, cl)

    # --- Derived ---------------------------------------------------------
    stall_margin = cl_max - target_cl
    endurance = (
        (target_cl ** 1.5) / cd_cruise if cd_cruise > 1e-12 else 0.0
    )

    return PolarMetrics(
        designation=polar.designation,
        cl_at_cruise_alpha=target_cl,
        cd_at_cruise=cd_cruise,
        ld_at_cruise=ld_cruise,
        ld_max=ld_max,
        alpha_ld_max_deg=alpha_ldmax,
        cl_max=cl_max,
        alpha_cl_max_deg=alpha_clmax,
        cd_min=cd_min,
        cm_at_cruise=cm_cruise,
        alpha_zl_deg=alpha_zl,
        cl_alpha_rad=cl_alpha,
        stall_margin_cl=stall_margin,
        endurance_param=endurance,
    )


# ---------------------------------------------------------------------------
# Public API — scoring
# ---------------------------------------------------------------------------
def score_airfoil(
    polar: XfoilPolar,
    reqs: WingRequirements,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> ScoredAirfoil:
    """Score a single airfoil polar against wing requirements.

    Args:
        polar: XFOIL polar data.
        reqs: Wing requirements.
        weights: Criterion weights (default :data:`DEFAULT_WEIGHTS`).

    Returns:
        :class:`ScoredAirfoil` (unranked — rank = 0).

    Raises:
        ValueError: If the polar is too small for metrics.
    """
    metrics = compute_polar_metrics(polar, reqs)
    sub_scores = _compute_sub_scores(metrics, reqs)
    total = _weighted_total(sub_scores, weights)

    return ScoredAirfoil(
        designation=polar.designation,
        total_score=total,
        sub_scores=sub_scores,
        metrics=metrics,
        rank=0,
    )


def rank_airfoils(
    polars: dict[str, XfoilPolar],
    reqs: WingRequirements,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> list[ScoredAirfoil]:
    """Score and rank multiple airfoils.

    Args:
        polars: Mapping of designation → XFOIL polar.
        reqs: Wing requirements.
        weights: Criterion weights.

    Returns:
        List of :class:`ScoredAirfoil` sorted best-first (rank 1 = best).
        Airfoils with fewer than 2 polar points are excluded.
    """
    scored: list[ScoredAirfoil] = []
    for designation, polar in polars.items():
        if polar.converged_count < 2:
            continue
        sa = score_airfoil(polar, reqs, weights)
        scored.append(sa)

    scored.sort(key=lambda s: s.total_score, reverse=True)

    ranked = [
        ScoredAirfoil(
            designation=s.designation,
            total_score=s.total_score,
            sub_scores=s.sub_scores,
            metrics=s.metrics,
            rank=i + 1,
        )
        for i, s in enumerate(scored)
    ]
    return ranked


# ---------------------------------------------------------------------------
# Public API — comparison helpers
# ---------------------------------------------------------------------------
def compare_polars(
    polars: dict[str, XfoilPolar],
    reqs: WingRequirements,
) -> list[PolarMetrics]:
    """Extract metrics for multiple airfoils without scoring.

    Args:
        polars: Mapping of designation → XFOIL polar.
        reqs: Wing requirements.

    Returns:
        List of :class:`PolarMetrics`, one per airfoil.
    """
    results: list[PolarMetrics] = []
    for polar in polars.values():
        if polar.converged_count < 2:
            continue
        results.append(compute_polar_metrics(polar, reqs))
    return results


# ---------------------------------------------------------------------------
# Private — sub-score computation
# ---------------------------------------------------------------------------
def _compute_sub_scores(
    m: PolarMetrics,
    reqs: WingRequirements,
) -> dict[str, float]:
    """Compute individual criterion scores in [0, 1].

    Args:
        m: Polar metrics.
        reqs: Wing requirements.

    Returns:
        Dict of criterion name → score.
    """
    scores: dict[str, float] = {}

    # L/D at cruise: normalise against a reference L/D of 100
    scores["ld_cruise"] = min(m.ld_at_cruise / 100.0, 1.0)

    # CL match: Gaussian penalty for deviation from required CL
    cl_err = abs(m.cl_at_cruise_alpha - reqs.required_cl_cruise)
    scores["cl_match"] = math.exp(-(cl_err ** 2) / (2 * 0.1 ** 2))

    # CD level: lower is better — normalise against Cd = 0.020 as "bad"
    scores["cd_level"] = max(1.0 - m.cd_at_cruise / 0.020, 0.0)

    # Stall margin: want at least 0.3 CL headroom
    scores["stall_margin"] = min(m.stall_margin_cl / 0.5, 1.0)
    scores["stall_margin"] = max(scores["stall_margin"], 0.0)

    # Cm magnitude: lower |Cm| is better for trim
    scores["cm_magnitude"] = max(1.0 - abs(m.cm_at_cruise) / 0.10, 0.0)

    # Endurance parameter: normalise against reference of 30
    scores["endurance"] = min(m.endurance_param / 30.0, 1.0)

    return scores


def _weighted_total(
    sub_scores: dict[str, float],
    weights: ScoringWeights,
) -> float:
    """Compute the composite weighted score.

    Args:
        sub_scores: Criterion → score mapping.
        weights: Weight object.

    Returns:
        Composite score in [0, 1].
    """
    w = {
        "ld_cruise": weights.ld_cruise,
        "cl_match": weights.cl_match,
        "cd_level": weights.cd_level,
        "stall_margin": weights.stall_margin,
        "cm_magnitude": weights.cm_magnitude,
        "endurance": weights.endurance,
    }
    total_w = sum(w.values())
    if total_w < 1e-12:
        return 0.0

    return sum(
        w[k] * sub_scores.get(k, 0.0) for k in w
    ) / total_w


# ---------------------------------------------------------------------------
# Private — interpolation helpers
# ---------------------------------------------------------------------------
def _interp_alpha_for_cl(
    alpha: np.ndarray, cl: np.ndarray, target_cl: float,
) -> float:
    """Interpolate α for a given CL value.

    Uses linear interpolation on the CL(α) curve.  If the target CL
    is outside the polar range, extrapolates linearly from the
    nearest two points.

    Args:
        alpha: α values [deg].  Shape ``(n,)``.
        cl: CL values.  Shape ``(n,)``.
        target_cl: Desired CL.

    Returns:
        Interpolated α [deg].
    """
    # np.interp expects xp to be increasing — CL is generally increasing
    # with α in the linear range, so we can use CL as the "x" axis
    # Ensure sorted by CL for interp
    sort_idx = np.argsort(cl)
    cl_sorted = cl[sort_idx]
    alpha_sorted = alpha[sort_idx]

    return float(np.interp(target_cl, cl_sorted, alpha_sorted))


def _estimate_cl_alpha(
    alpha: np.ndarray, cl: np.ndarray,
) -> float:
    """Estimate lift-curve slope from the linear portion of the polar.

    Fits a line to α ∈ [−2°, 8°] (or available range).

    Args:
        alpha: α values [deg].  Shape ``(n,)``.
        cl: CL values.  Shape ``(n,)``.

    Returns:
        dCL/dα [1/rad].
    """
    # Select linear region
    mask = (alpha >= -2.0) & (alpha <= 8.0)
    if np.sum(mask) < 2:
        mask = np.ones(len(alpha), dtype=bool)
    if np.sum(mask) < 2:
        return 2.0 * math.pi  # fallback to thin airfoil

    a_lin = alpha[mask]
    cl_lin = cl[mask]

    # Linear fit: CL = slope * α + intercept
    coeffs = np.polyfit(a_lin, cl_lin, 1)
    slope_per_deg = coeffs[0]

    return float(slope_per_deg * 180.0 / math.pi)  # convert to per-radian
