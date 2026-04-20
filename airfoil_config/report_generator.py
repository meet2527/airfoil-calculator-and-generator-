"""
PDF report generator for airfoil analysis.

Produces a formatted PDF document containing:

* aircraft specifications summary
* wing requirements table
* airfoil geometry plot
* polar data plots (CL vs α, CD vs α, CL vs CD, L/D vs α)
* scoring summary table
* lifting-line results (if provided)

Uses **matplotlib** for plots and **ReportLab** for PDF composition.

Typical usage:
    >>> from airfoil_config.report_generator import generate_report, ReportData
    >>> data = ReportData(
    ...     title="Wing Analysis — NACA 2412",
    ...     specs=specs, requirements=reqs,
    ...     coords=coords, polar=polar, metrics=metrics,
    ... )
    >>> generate_report(data, "report.pdf")
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from airfoil_config.naca_geometry import AirfoilCoordinates
from airfoil_config.requirements import AircraftSpecs, WingRequirements
from airfoil_config.scoring import PolarMetrics
from airfoil_config.xfoil_runner import XfoilPolar


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ReportData:
    """All data needed for a report.

    Only ``title`` is required — all other fields are optional and
    will be included in the report if present.

    Attributes:
        title: Report title.
        specs: Aircraft specifications.
        requirements: Computed wing requirements.
        coords: NACA airfoil coordinates.
        polar: XFOIL polar data.
        metrics: Scored polar metrics.
        llt_cl: Spanwise CL distribution from LLT.
        llt_eta: Normalised span stations for LLT data.
        notes: Free-form notes to include at the end.
        author: Author name for the title page.
        date: Report date (defaults to today).
    """

    title: str
    specs: Optional[AircraftSpecs] = None
    requirements: Optional[WingRequirements] = None
    coords: Optional[AirfoilCoordinates] = None
    polar: Optional[XfoilPolar] = None
    metrics: Optional[PolarMetrics] = None
    llt_cl: Optional[np.ndarray] = None
    llt_eta: Optional[np.ndarray] = None
    notes: str = ""
    author: str = ""
    date: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_report(
    data: ReportData,
    filepath: str,
    page_size: tuple[float, float] = A4 if RL_AVAILABLE else (595.27, 841.89),
) -> str:
    """Generate a PDF report.

    Args:
        data: Report data (see :class:`ReportData`).
        filepath: Output PDF path.
        page_size: Page dimensions (default A4).

    Returns:
        Absolute path to the generated PDF.

    Raises:
        RuntimeError: If ReportLab or matplotlib is not installed.
    """
    _check_deps()

    elements: list = []
    styles = getSampleStyleSheet()
    tmp_images: list[str] = []

    try:
        # --- Title section ------------------------------------------------
        elements.extend(_build_title(data, styles))

        # --- Specs table --------------------------------------------------
        if data.specs is not None:
            elements.extend(_build_specs_table(data.specs, styles))

        # --- Requirements table -------------------------------------------
        if data.requirements is not None:
            elements.extend(_build_requirements_table(data.requirements, styles))

        # --- Airfoil geometry plot ----------------------------------------
        if data.coords is not None:
            img_path = _plot_airfoil_geometry(data.coords)
            tmp_images.append(img_path)
            elements.extend(_build_image_section(
                "Airfoil Geometry", img_path, styles,
            ))

        # --- Polar plots --------------------------------------------------
        if data.polar is not None:
            paths = _plot_polar(data.polar)
            tmp_images.extend(paths)
            for label, path in zip(
                ["CL vs α", "CD vs α", "CL vs CD (Drag Polar)", "L/D vs α"],
                paths,
            ):
                elements.extend(_build_image_section(label, path, styles))

        # --- Metrics table ------------------------------------------------
        if data.metrics is not None:
            elements.extend(_build_metrics_table(data.metrics, styles))

        # --- LLT distribution plot ----------------------------------------
        if data.llt_cl is not None and data.llt_eta is not None:
            img = _plot_llt_distribution(data.llt_eta, data.llt_cl)
            tmp_images.append(img)
            elements.extend(_build_image_section(
                "Spanwise CL Distribution (LLT)", img, styles,
            ))

        # --- Notes --------------------------------------------------------
        if data.notes:
            elements.append(Spacer(1, 12 * mm))
            elements.append(Paragraph("Notes", styles["Heading2"]))
            elements.append(Paragraph(data.notes, styles["BodyText"]))

        # --- Build PDF ----------------------------------------------------
        doc = SimpleDocTemplate(
            filepath,
            pagesize=page_size,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        doc.build(elements)

    finally:
        # Clean up temp images
        for p in tmp_images:
            try:
                os.unlink(p)
            except OSError:
                pass

    return str(Path(filepath).resolve())


def generate_comparison_report(
    title: str,
    polars: dict[str, XfoilPolar],
    requirements: WingRequirements,
    filepath: str,
) -> str:
    """Generate a comparison report for multiple airfoils.

    Args:
        title: Report title.
        polars: Mapping of designation → polar.
        requirements: Wing requirements.
        filepath: Output PDF path.

    Returns:
        Absolute path to the PDF.

    Raises:
        RuntimeError: If dependencies are missing.
    """
    _check_deps()

    elements: list = []
    styles = getSampleStyleSheet()
    tmp_images: list[str] = []

    try:
        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Spacer(1, 8 * mm))

        # Overlay CL vs α
        img = _plot_comparison_cl(polars)
        tmp_images.append(img)
        elements.extend(_build_image_section("CL vs α Comparison", img, styles))

        # Overlay drag polars
        img2 = _plot_comparison_drag(polars)
        tmp_images.append(img2)
        elements.extend(_build_image_section("Drag Polar Comparison", img2, styles))

        doc = SimpleDocTemplate(filepath, pagesize=A4)
        doc.build(elements)

    finally:
        for p in tmp_images:
            try:
                os.unlink(p)
            except OSError:
                pass

    return str(Path(filepath).resolve())


# ---------------------------------------------------------------------------
# Private — PDF building helpers
# ---------------------------------------------------------------------------
def _build_title(data: ReportData, styles) -> list:
    """Build title section elements."""
    els = []
    els.append(Paragraph(data.title, styles["Title"]))
    date_str = data.date or datetime.now().strftime("%Y-%m-%d")
    if data.author:
        els.append(Paragraph(f"Author: {data.author}", styles["Normal"]))
    els.append(Paragraph(f"Date: {date_str}", styles["Normal"]))
    els.append(Spacer(1, 10 * mm))
    return els


def _build_specs_table(specs: AircraftSpecs, styles) -> list:
    """Build aircraft specs table."""
    els = []
    els.append(Paragraph("Aircraft Specifications", styles["Heading2"]))

    rows = [
        ["Parameter", "Value"],
        ["Weight", f"{specs.weight_n:.1f} N"],
        ["Wing Span", f"{specs.wing_span_m:.2f} m"],
        ["Purpose", specs.purpose],
        ["Cruise Altitude", f"{specs.cruise_altitude_m:.0f} m"],
        ["Cruise Velocity", f"{specs.cruise_velocity_ms:.1f} m/s"],
    ]
    if specs.wing_area_m2 is not None:
        rows.append(["Wing Area", f"{specs.wing_area_m2:.2f} m²"])
    if specs.aspect_ratio is not None:
        rows.append(["Aspect Ratio", f"{specs.aspect_ratio:.2f}"])

    els.append(_styled_table(rows))
    els.append(Spacer(1, 8 * mm))
    return els


def _build_requirements_table(reqs: WingRequirements, styles) -> list:
    """Build wing requirements table."""
    els = []
    els.append(Paragraph("Wing Requirements", styles["Heading2"]))

    rows = [
        ["Parameter", "Value"],
        ["Wing Loading", f"{reqs.wing_loading_pa:.1f} Pa"],
        ["Reynolds Number", f"{reqs.reynolds_number:.2e}"],
        ["Required CL (cruise)", f"{reqs.required_cl_cruise:.4f}"],
        ["MAC", f"{reqs.mean_aero_chord_m:.3f} m"],
        ["Stall Speed", f"{reqs.stall_speed_ms:.1f} m/s"],
        ["CL_max Estimate", f"{reqs.cl_max_estimate:.2f}"],
        ["Mach Number", f"{reqs.mach_number:.4f}"],
        ["Dynamic Pressure", f"{reqs.dynamic_pressure_pa:.1f} Pa"],
        ["Aspect Ratio", f"{reqs.aspect_ratio:.2f}"],
    ]

    els.append(_styled_table(rows))
    els.append(Spacer(1, 8 * mm))
    return els


def _build_metrics_table(metrics: PolarMetrics, styles) -> list:
    """Build polar metrics table."""
    els = []
    els.append(Paragraph(
        f"Performance Metrics — {metrics.designation}", styles["Heading2"],
    ))

    rows = [
        ["Metric", "Value"],
        ["L/D at Cruise", f"{metrics.ld_at_cruise:.1f}"],
        ["L/D Max", f"{metrics.ld_max:.1f}"],
        ["α at L/D Max", f"{metrics.alpha_ld_max_deg:.1f}°"],
        ["CL Max", f"{metrics.cl_max:.3f}"],
        ["α at CL Max", f"{metrics.alpha_cl_max_deg:.1f}°"],
        ["CD Min", f"{metrics.cd_min:.5f}"],
        ["CD at Cruise", f"{metrics.cd_at_cruise:.5f}"],
        ["Cm at Cruise", f"{metrics.cm_at_cruise:.4f}"],
        ["α Zero-Lift", f"{metrics.alpha_zl_deg:.2f}°"],
        ["CL_α", f"{metrics.cl_alpha_rad:.3f} /rad"],
        ["Stall Margin (ΔCL)", f"{metrics.stall_margin_cl:.3f}"],
        ["Endurance Param", f"{metrics.endurance_param:.2f}"],
    ]

    els.append(_styled_table(rows))
    els.append(Spacer(1, 8 * mm))
    return els


def _build_image_section(title: str, img_path: str, styles) -> list:
    """Build a section with a heading and an image."""
    els = []
    els.append(Paragraph(title, styles["Heading3"]))
    els.append(Spacer(1, 3 * mm))
    els.append(Image(img_path, width=160 * mm, height=100 * mm))
    els.append(Spacer(1, 6 * mm))
    return els


def _styled_table(rows: list[list[str]]) -> Table:
    """Create a styled ReportLab table with header row."""
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------------
# Private — matplotlib plots
# ---------------------------------------------------------------------------
_FIG_DPI = 150
_FIG_SIZE = (8, 5)


def _save_fig(fig) -> str:
    """Save a matplotlib figure to a temp PNG and return its path."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="rpt_")
    os.close(fd)
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_airfoil_geometry(coords: AirfoilCoordinates) -> str:
    """Plot airfoil profile and camber line."""
    fig, ax = plt.subplots(1, 1, figsize=_FIG_SIZE)

    ax.plot(coords.x_upper, coords.y_upper, "b-", lw=1.5, label="Upper")
    ax.plot(coords.x_lower, coords.y_lower, "b-", lw=1.5, label="Lower")
    ax.plot(coords.x_camber, coords.y_camber, "r--", lw=1.0, label="Camber")

    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(coords.designation)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig)


def _plot_polar(polar: XfoilPolar) -> list[str]:
    """Plot the four standard polar charts. Returns list of image paths."""
    paths: list[str] = []

    # CL vs α
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(polar.alpha_deg, polar.cl, "b-o", ms=3)
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("CL")
    ax.set_title(f"{polar.designation} — CL vs α (Re={polar.reynolds:.0e})")
    ax.grid(True, alpha=0.3)
    paths.append(_save_fig(fig))

    # CD vs α
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(polar.alpha_deg, polar.cd, "r-o", ms=3)
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("CD")
    ax.set_title(f"{polar.designation} — CD vs α")
    ax.grid(True, alpha=0.3)
    paths.append(_save_fig(fig))

    # Drag polar (CL vs CD)
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(polar.cd, polar.cl, "g-o", ms=3)
    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title(f"{polar.designation} — Drag Polar")
    ax.grid(True, alpha=0.3)
    paths.append(_save_fig(fig))

    # L/D vs α
    ld = polar.cl / np.maximum(polar.cd, 1e-12)
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(polar.alpha_deg, ld, "m-o", ms=3)
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("L/D")
    ax.set_title(f"{polar.designation} — L/D vs α")
    ax.grid(True, alpha=0.3)
    paths.append(_save_fig(fig))

    return paths


def _plot_llt_distribution(eta: np.ndarray, cl: np.ndarray) -> str:
    """Plot spanwise CL distribution."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(eta, cl, "b-o", ms=3)
    ax.set_xlabel("η = y / (b/2)")
    ax.set_ylabel("Section CL")
    ax.set_title("Spanwise Lift Distribution (LLT)")
    ax.grid(True, alpha=0.3)
    return _save_fig(fig)


def _plot_comparison_cl(polars: dict[str, XfoilPolar]) -> str:
    """Overlay CL vs α for multiple airfoils."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    for name, polar in polars.items():
        ax.plot(polar.alpha_deg, polar.cl, "-o", ms=2, label=name)
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("CL")
    ax.set_title("CL vs α — Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return _save_fig(fig)


def _plot_comparison_drag(polars: dict[str, XfoilPolar]) -> str:
    """Overlay drag polars for multiple airfoils."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    for name, polar in polars.items():
        ax.plot(polar.cd, polar.cl, "-o", ms=2, label=name)
    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title("Drag Polar — Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return _save_fig(fig)


# ---------------------------------------------------------------------------
# Private — dependency check
# ---------------------------------------------------------------------------
def _check_deps() -> None:
    """Raise if matplotlib or ReportLab is missing."""
    if not MPL_AVAILABLE:
        raise RuntimeError(
            "matplotlib is required for report generation. "
            "Install with: pip install matplotlib"
        )
    if not RL_AVAILABLE:
        raise RuntimeError(
            "ReportLab is required for report generation. "
            "Install with: pip install reportlab"
        )
