"""
XFOIL subprocess wrapper.

Drives the XFOIL executable via ``subprocess.Popen`` stdin/stdout to
compute viscous (and inviscid) airfoil polars.  Handles coordinate
loading, Reynolds / Mach setup, alpha sweeps, and polar parsing.

The wrapper is intentionally defensive: it validates inputs, enforces
a timeout, and maps common XFOIL failure modes (convergence failure,
missing binary, NaN in output) to explicit Python exceptions.

Typical usage:
    >>> polar = run_xfoil(
    ...     coords_x=[1.0, 0.5, 0.0, 0.5, 1.0],
    ...     coords_y=[0.0, 0.06, 0.0, -0.06, 0.0],
    ...     reynolds=1e6,
    ...     alpha_start=-2.0, alpha_end=12.0, alpha_step=0.5,
    ... )
    >>> polar.alpha_deg   # np.ndarray of converged angles
    >>> polar.cl          # corresponding CL values

Note:
    Requires the ``xfoil`` (or ``xfoil.exe`` on Windows) binary on
    ``$PATH`` or supplied via the *xfoil_path* argument.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class XfoilError(RuntimeError):
    """Base exception for XFOIL-related failures."""


class XfoilNotFoundError(XfoilError):
    """Raised when the XFOIL binary cannot be located."""


class XfoilConvergenceError(XfoilError):
    """Raised when XFOIL fails to converge at one or more α stations."""


class XfoilTimeoutError(XfoilError):
    """Raised when XFOIL exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XfoilPolar:
    """Parsed XFOIL polar data.

    All arrays share the same length — only converged α stations are
    included.

    Attributes:
        alpha_deg: Angle of attack [deg].  Shape ``(n,)``.
        cl: Lift coefficient.  Shape ``(n,)``.
        cd: Drag coefficient.  Shape ``(n,)``.
        cdp: Pressure drag coefficient.  Shape ``(n,)``.
        cm: Pitching-moment coefficient (about c/4).  Shape ``(n,)``.
        top_xtr: Upper-surface transition location x/c.  Shape ``(n,)``.
        bot_xtr: Lower-surface transition location x/c.  Shape ``(n,)``.
        reynolds: Reynolds number used.
        mach: Mach number used.
        n_crit: Amplification factor (e^n) used.
        designation: Airfoil identifier string.
        converged_count: Number of converged α stations.
        total_count: Total α stations attempted.
    """

    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cdp: np.ndarray
    cm: np.ndarray
    top_xtr: np.ndarray
    bot_xtr: np.ndarray
    reynolds: float
    mach: float
    n_crit: float
    designation: str
    converged_count: int
    total_count: int


@dataclass(frozen=True)
class XfoilConfig:
    """Configuration for an XFOIL run.

    Attributes:
        reynolds: Reynolds number.
        mach: Mach number (default 0).
        alpha_start: Start of α sweep [deg].
        alpha_end: End of α sweep [deg].
        alpha_step: α increment [deg].
        n_crit: e^n transition criterion (default 9 = clean wind tunnel).
        max_iter: XFOIL viscous iterations per point (default 100).
        timeout_s: Subprocess timeout [seconds] (default 60).
        xfoil_path: Path to XFOIL binary (None = auto-detect on PATH).
        repanel: If True, run PANE to repanel the airfoil (default True).
        n_panels: Number of panels for repaneling (default 200).
    """

    reynolds: float
    mach: float = 0.0
    alpha_start: float = -2.0
    alpha_end: float = 12.0
    alpha_step: float = 0.5
    n_crit: float = 9.0
    max_iter: int = 100
    timeout_s: float = 60.0
    xfoil_path: Optional[str] = None
    repanel: bool = True
    n_panels: int = 200


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_xfoil(
    coords_x: list[float] | np.ndarray,
    coords_y: list[float] | np.ndarray,
    reynolds: float,
    mach: float = 0.0,
    alpha_start: float = -2.0,
    alpha_end: float = 12.0,
    alpha_step: float = 0.5,
    n_crit: float = 9.0,
    max_iter: int = 100,
    timeout_s: float = 60.0,
    xfoil_path: Optional[str] = None,
    designation: str = "airfoil",
    repanel: bool = True,
    n_panels: int = 200,
) -> XfoilPolar:
    """Run XFOIL on given airfoil coordinates and return the polar.

    Coordinates should be in Selig format (single loop, TE → upper →
    LE → lower → TE) or at minimum as matching x, y arrays defining
    the airfoil surface.

    Args:
        coords_x: x-coordinates of the airfoil surface.
        coords_y: y-coordinates of the airfoil surface.
        reynolds: Reynolds number (must be positive).
        mach: Free-stream Mach number (default 0).
        alpha_start: First angle of attack [deg].
        alpha_end: Last angle of attack [deg].
        alpha_step: α step size [deg] (must be positive).
        n_crit: e^n amplification factor (default 9).
        max_iter: Max viscous iterations per α (default 100).
        timeout_s: Subprocess timeout [s] (default 60).
        xfoil_path: Path to XFOIL binary (None = auto-detect).
        designation: Label for the airfoil (stored in result).
        repanel: Repanel the airfoil in XFOIL (default True).
        n_panels: Number of panels if repaneling.

    Returns:
        :class:`XfoilPolar` with converged data.

    Raises:
        XfoilNotFoundError: If the binary is not found.
        XfoilTimeoutError: If the subprocess times out.
        XfoilConvergenceError: If zero α stations converge.
        ValueError: On invalid input parameters.
    """
    config = XfoilConfig(
        reynolds=reynolds, mach=mach,
        alpha_start=alpha_start, alpha_end=alpha_end,
        alpha_step=alpha_step, n_crit=n_crit, max_iter=max_iter,
        timeout_s=timeout_s, xfoil_path=xfoil_path,
        repanel=repanel, n_panels=n_panels,
    )
    return run_xfoil_config(
        coords_x=coords_x, coords_y=coords_y,
        config=config, designation=designation,
    )


def run_xfoil_config(
    coords_x: list[float] | np.ndarray,
    coords_y: list[float] | np.ndarray,
    config: XfoilConfig,
    designation: str = "airfoil",
) -> XfoilPolar:
    """Run XFOIL with an :class:`XfoilConfig` object.

    Args:
        coords_x: x-coordinates.
        coords_y: y-coordinates.
        config: XFOIL configuration.
        designation: Airfoil label.

    Returns:
        :class:`XfoilPolar`.

    Raises:
        XfoilNotFoundError: Binary not found.
        XfoilTimeoutError: Subprocess timeout.
        XfoilConvergenceError: No converged points.
        ValueError: Bad inputs.
    """
    # --- Validate ----------------------------------------------------------
    x_arr = np.asarray(coords_x, dtype=np.float64)
    y_arr = np.asarray(coords_y, dtype=np.float64)
    _validate_inputs(x_arr, y_arr, config)

    # --- Locate binary -----------------------------------------------------
    xfoil_bin = _find_xfoil(config.xfoil_path)

    # --- Write temporary files ---------------------------------------------
    work_dir = tempfile.mkdtemp(prefix="xfoil_")
    try:
        coord_file = os.path.join(work_dir, "airfoil.dat")
        polar_file = os.path.join(work_dir, "polar.dat")

        _write_coordinate_file(coord_file, x_arr, y_arr, designation)
        script = _build_xfoil_script(coord_file, polar_file, config)

        # --- Run XFOIL -----------------------------------------------------
        _execute_xfoil(xfoil_bin, script, config.timeout_s, work_dir)

        # --- Parse polar ---------------------------------------------------
        polar = _parse_polar_file(polar_file, config, designation)

    finally:
        # Clean up temp files
        shutil.rmtree(work_dir, ignore_errors=True)

    return polar


def find_xfoil_binary(hint: Optional[str] = None) -> str:
    """Locate the XFOIL executable.

    Args:
        hint: Optional explicit path or name.

    Returns:
        Absolute path to the XFOIL binary.

    Raises:
        XfoilNotFoundError: If not found.
    """
    return _find_xfoil(hint)


# ---------------------------------------------------------------------------
# Private — binary location
# ---------------------------------------------------------------------------
_XFOIL_NAMES = ("xfoil", "xfoil.exe", "Xfoil", "XFOIL")


def _find_xfoil(hint: Optional[str]) -> str:
    """Resolve the XFOIL binary path.

    Args:
        hint: User-supplied path or None.

    Returns:
        Absolute path string.

    Raises:
        XfoilNotFoundError: If the binary cannot be found.
    """
    if hint is not None:
        path = Path(hint)
        if path.is_file():
            return str(path.resolve())
        # Maybe it's just a name on PATH
        found = shutil.which(hint)
        if found:
            return found
        raise XfoilNotFoundError(f"XFOIL binary not found at {hint!r}")

    for name in _XFOIL_NAMES:
        found = shutil.which(name)
        if found:
            return found

    raise XfoilNotFoundError(
        "XFOIL binary not found on PATH. Install XFOIL or pass "
        "xfoil_path= explicitly."
    )


# ---------------------------------------------------------------------------
# Private — temp file I/O
# ---------------------------------------------------------------------------
def _write_coordinate_file(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
) -> None:
    """Write airfoil coordinates in Selig/Lednicer .dat format.

    Args:
        path: Output file path.
        x: x-coordinates.  Shape ``(n,)``.
        y: y-coordinates.  Shape ``(n,)``.
        name: Header name line.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x, y):
            f.write(f" {xi: .8f}  {yi: .8f}\n")


def _build_xfoil_script(
    coord_file: str,
    polar_file: str,
    cfg: XfoilConfig,
) -> str:
    """Build the XFOIL command sequence.

    Args:
        coord_file: Path to the .dat coordinate file.
        polar_file: Path for polar output.
        cfg: Run configuration.

    Returns:
        Multi-line string of XFOIL commands.
    """
    lines: list[str] = []

    # Load airfoil
    lines.append(f"LOAD {coord_file}")
    lines.append("")  # accept default name

    # Repanel
    if cfg.repanel:
        lines.append("PPAR")
        lines.append(f"N {cfg.n_panels}")
        lines.append("")  # accept
        lines.append("")  # exit PPAR

    # Enter OPER mode
    lines.append("OPER")

    # Viscous mode
    lines.append(f"VISC {cfg.reynolds:.0f}")
    lines.append(f"MACH {cfg.mach:.4f}")

    # Iteration limit
    lines.append(f"ITER {cfg.max_iter}")

    # Transition criterion
    lines.append("VPAR")
    lines.append(f"N {cfg.n_crit:.1f}")
    lines.append("")  # exit VPAR

    # Set up polar accumulation
    lines.append("PACC")
    lines.append(polar_file)
    lines.append("")  # no dump file

    # Alpha sweep
    lines.append(f"ASEQ {cfg.alpha_start:.4f} {cfg.alpha_end:.4f} {cfg.alpha_step:.4f}")

    # Close polar accumulation and exit
    lines.append("PACC")
    lines.append("")
    lines.append("QUIT")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Private — subprocess execution
# ---------------------------------------------------------------------------
def _execute_xfoil(
    binary: str,
    script: str,
    timeout_s: float,
    cwd: str,
) -> None:
    """Run XFOIL with the given script via stdin.

    Args:
        binary: Path to XFOIL executable.
        script: Command string to pipe to stdin.
        timeout_s: Timeout in seconds.
        cwd: Working directory.

    Raises:
        XfoilTimeoutError: On timeout.
        XfoilError: On non-zero exit code.
    """
    try:
        proc = subprocess.run(
            [binary],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise XfoilNotFoundError(f"Cannot execute {binary!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise XfoilTimeoutError(
            f"XFOIL timed out after {timeout_s:.0f}s"
        ) from exc

    # XFOIL often returns 0 even on partial failure; we check polar content later.
    # But a truly bad exit code should still be reported.
    if proc.returncode not in (0, None):
        stderr_head = (proc.stderr or "")[:500]
        raise XfoilError(
            f"XFOIL exited with code {proc.returncode}: {stderr_head}"
        )


# ---------------------------------------------------------------------------
# Private — polar parsing
# ---------------------------------------------------------------------------
def _parse_polar_file(
    path: str,
    cfg: XfoilConfig,
    designation: str,
) -> XfoilPolar:
    """Read and parse an XFOIL polar accumulation file.

    The file has a multi-line header (lines starting with ``-`` or text)
    followed by data columns::

        alpha   CL      CD      CDp     CM    Top_Xtr  Bot_Xtr

    Args:
        path: Path to the polar file.
        cfg: Configuration (for metadata).
        designation: Airfoil label.

    Returns:
        :class:`XfoilPolar`.

    Raises:
        XfoilConvergenceError: If the file is missing or contains
            zero data rows.
    """
    if not os.path.isfile(path):
        raise XfoilConvergenceError(
            "XFOIL produced no polar file — possible total convergence failure"
        )

    data_rows: list[list[float]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # The header ends with a dashed separator line
            if stripped.startswith("---"):
                in_data = True
                continue
            if not in_data:
                continue
            # Parse numeric row
            parts = stripped.split()
            if len(parts) < 7:
                continue
            try:
                row = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            # Skip rows with NaN
            if any(math.isnan(v) for v in row):
                continue
            data_rows.append(row)

    # Total α stations attempted
    total = int(
        round(abs(cfg.alpha_end - cfg.alpha_start) / cfg.alpha_step) + 1
    ) if cfg.alpha_step > 0 else 0

    if not data_rows:
        raise XfoilConvergenceError(
            f"XFOIL converged at 0/{total} alpha stations"
        )

    data = np.array(data_rows)

    return XfoilPolar(
        alpha_deg=data[:, 0],
        cl=data[:, 1],
        cd=data[:, 2],
        cdp=data[:, 3],
        cm=data[:, 4],
        top_xtr=data[:, 5],
        bot_xtr=data[:, 6],
        reynolds=cfg.reynolds,
        mach=cfg.mach,
        n_crit=cfg.n_crit,
        designation=designation,
        converged_count=len(data_rows),
        total_count=total,
    )


# ---------------------------------------------------------------------------
# Private — input validation
# ---------------------------------------------------------------------------
def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    cfg: XfoilConfig,
) -> None:
    """Validate coordinate arrays and configuration.

    Args:
        x: x-coordinates.
        y: y-coordinates.
        cfg: Run configuration.

    Raises:
        ValueError: On bad inputs.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Coordinates must be 1-D, got x:{x.ndim}D y:{y.ndim}D")
    if len(x) != len(y):
        raise ValueError(f"x and y lengths must match: {len(x)} vs {len(y)}")
    if len(x) < 5:
        raise ValueError(f"Need >= 5 coordinate points, got {len(x)}")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Coordinates contain NaN")
    if cfg.reynolds <= 0.0 or math.isnan(cfg.reynolds):
        raise ValueError(f"reynolds must be positive, got {cfg.reynolds}")
    if cfg.mach < 0.0 or math.isnan(cfg.mach):
        raise ValueError(f"mach must be >= 0, got {cfg.mach}")
    if cfg.alpha_step <= 0.0:
        raise ValueError(f"alpha_step must be positive, got {cfg.alpha_step}")
    if cfg.timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be positive, got {cfg.timeout_s}")
    if cfg.max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {cfg.max_iter}")
