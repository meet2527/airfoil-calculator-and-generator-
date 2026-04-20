"""
Tests for airfoil_config.xfoil_runner — XFOIL subprocess wrapper.

Since XFOIL is unlikely to be installed in CI / dev environments,
these tests focus on:
    - Input validation
    - Coordinate file writing
    - Script generation
    - Polar file parsing (with synthetic data)
    - Exception hierarchy
    - Binary-finding logic

Integration tests that actually call XFOIL are guarded behind a
``@pytest.mark.skipif`` that checks for the binary.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from airfoil_config.xfoil_runner import (
    XfoilConfig,
    XfoilConvergenceError,
    XfoilError,
    XfoilNotFoundError,
    XfoilPolar,
    XfoilTimeoutError,
    _build_xfoil_script,
    _parse_polar_file,
    _validate_inputs,
    _write_coordinate_file,
    find_xfoil_binary,
    run_xfoil,
)

ABS_TOL = 1e-6

# Whether XFOIL is available for integration tests
_XFOIL_AVAILABLE = shutil.which("xfoil") is not None or shutil.which("xfoil.exe") is not None


def _dummy_coords() -> tuple[np.ndarray, np.ndarray]:
    """Simple symmetric airfoil coords (10 points)."""
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = 0.5 * (1.0 + np.cos(theta))
    y = 0.06 * np.sin(theta)
    return x, y


def _default_config(**overrides) -> XfoilConfig:
    """Create a config with sensible defaults."""
    defaults = dict(
        reynolds=1e6, mach=0.0,
        alpha_start=-2.0, alpha_end=10.0, alpha_step=0.5,
    )
    defaults.update(overrides)
    return XfoilConfig(**defaults)


# ===================================================================
# Input validation
# ===================================================================
class TestValidation:
    """Tests for _validate_inputs."""

    def test_valid_passes(self) -> None:
        x, y = _dummy_coords()
        _validate_inputs(x, y, _default_config())

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="lengths"):
            _validate_inputs(
                np.array([0, 0.5, 1.0]),
                np.array([0, 0.05]),
                _default_config(),
            )

    def test_too_few_points(self) -> None:
        with pytest.raises(ValueError, match=">= 5"):
            _validate_inputs(
                np.array([0, 0.5, 1.0, 0.5]),
                np.array([0, 0.05, 0, -0.05]),
                _default_config(),
            )

    def test_nan_in_coords(self) -> None:
        x = np.array([0, 0.5, float("nan"), 0.5, 1.0])
        y = np.zeros(5)
        with pytest.raises(ValueError, match="NaN"):
            _validate_inputs(x, y, _default_config())

    def test_negative_reynolds(self) -> None:
        x, y = _dummy_coords()
        with pytest.raises(ValueError, match="reynolds"):
            _validate_inputs(x, y, _default_config(reynolds=-1.0))

    def test_negative_mach(self) -> None:
        x, y = _dummy_coords()
        with pytest.raises(ValueError, match="mach"):
            _validate_inputs(x, y, _default_config(mach=-0.1))

    def test_zero_alpha_step(self) -> None:
        x, y = _dummy_coords()
        with pytest.raises(ValueError, match="alpha_step"):
            _validate_inputs(x, y, _default_config(alpha_step=0.0))

    def test_zero_timeout(self) -> None:
        x, y = _dummy_coords()
        with pytest.raises(ValueError, match="timeout"):
            _validate_inputs(x, y, _default_config(timeout_s=0.0))

    def test_zero_max_iter(self) -> None:
        x, y = _dummy_coords()
        with pytest.raises(ValueError, match="max_iter"):
            _validate_inputs(x, y, _default_config(max_iter=0))

    def test_2d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            _validate_inputs(
                np.zeros((5, 2)), np.zeros((5, 2)), _default_config(),
            )


# ===================================================================
# Coordinate file writing
# ===================================================================
class TestCoordinateFile:
    """Tests for _write_coordinate_file."""

    def test_writes_header_and_points(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.dat")
            x = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
            y = np.array([0.0, 0.06, 0.0, -0.06, 0.0])
            _write_coordinate_file(path, x, y, "NACA 0012")

            lines = Path(path).read_text().strip().splitlines()
            assert lines[0] == "NACA 0012"
            assert len(lines) == 6  # header + 5 points

    def test_values_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.dat")
            x = np.array([1.0, 0.0])
            y = np.array([0.001, -0.001])
            _write_coordinate_file(path, x, y, "test")

            lines = Path(path).read_text().strip().splitlines()
            vals = lines[1].split()
            assert float(vals[0]) == pytest.approx(1.0)
            assert float(vals[1]) == pytest.approx(0.001, abs=1e-7)


# ===================================================================
# Script generation
# ===================================================================
class TestScriptGeneration:
    """Tests for _build_xfoil_script."""

    def test_contains_load(self) -> None:
        s = _build_xfoil_script("a.dat", "p.dat", _default_config())
        assert "LOAD a.dat" in s

    def test_contains_visc(self) -> None:
        s = _build_xfoil_script("a.dat", "p.dat", _default_config(reynolds=2e6))
        assert "VISC 2000000" in s

    def test_contains_aseq(self) -> None:
        cfg = _default_config(alpha_start=-5.0, alpha_end=15.0, alpha_step=1.0)
        s = _build_xfoil_script("a.dat", "p.dat", cfg)
        assert "ASEQ -5.0000 15.0000 1.0000" in s

    def test_contains_quit(self) -> None:
        s = _build_xfoil_script("a.dat", "p.dat", _default_config())
        assert "QUIT" in s

    def test_contains_pacc(self) -> None:
        s = _build_xfoil_script("a.dat", "p.dat", _default_config())
        assert "PACC" in s

    def test_repanel_included(self) -> None:
        cfg = _default_config(repanel=True, n_panels=250)
        s = _build_xfoil_script("a.dat", "p.dat", cfg)
        assert "PPAR" in s
        assert "N 250" in s

    def test_repanel_skipped(self) -> None:
        cfg = XfoilConfig(
            reynolds=1e6, repanel=False,
        )
        s = _build_xfoil_script("a.dat", "p.dat", cfg)
        assert "PPAR" not in s

    def test_mach_in_script(self) -> None:
        cfg = _default_config(mach=0.3)
        s = _build_xfoil_script("a.dat", "p.dat", cfg)
        assert "MACH 0.3000" in s

    def test_ncrit_in_script(self) -> None:
        cfg = _default_config(n_crit=11.0)
        s = _build_xfoil_script("a.dat", "p.dat", cfg)
        assert "N 11.0" in s


# ===================================================================
# Polar file parsing
# ===================================================================
class TestPolarParsing:
    """Tests for _parse_polar_file with synthetic data."""

    @pytest.fixture()
    def polar_path(self, tmp_path: Path) -> Path:
        """Write a synthetic XFOIL polar file."""
        content = """\
 Calculated polar for: NACA 0012

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000

  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
 ------- -------- --------- --------- -------- -------- --------
  -2.000  -0.2190   0.00573   0.00201  -0.0025  0.6912   0.0719
   0.000   0.0000   0.00489   0.00176   0.0000  0.5977   0.5977
   2.000   0.2190   0.00573   0.00201   0.0025  0.0719   0.6912
   4.000   0.4382   0.00694   0.00245   0.0052  0.0412   0.7654
   6.000   0.6528   0.00878   0.00321  -0.0079  0.0298   0.8123
"""
        p = tmp_path / "polar.dat"
        p.write_text(content)
        return p

    def test_parses_correct_count(self, polar_path: Path) -> None:
        cfg = _default_config(alpha_start=-2.0, alpha_end=6.0, alpha_step=2.0)
        polar = _parse_polar_file(str(polar_path), cfg, "NACA 0012")
        assert polar.converged_count == 5

    def test_alpha_values(self, polar_path: Path) -> None:
        cfg = _default_config()
        polar = _parse_polar_file(str(polar_path), cfg, "test")
        np.testing.assert_allclose(
            polar.alpha_deg, [-2.0, 0.0, 2.0, 4.0, 6.0], atol=0.001,
        )

    def test_cl_values(self, polar_path: Path) -> None:
        cfg = _default_config()
        polar = _parse_polar_file(str(polar_path), cfg, "test")
        assert polar.cl[1] == pytest.approx(0.0, abs=0.001)  # α=0 symmetric
        assert polar.cl[2] > 0  # α=2

    def test_cd_positive(self, polar_path: Path) -> None:
        cfg = _default_config()
        polar = _parse_polar_file(str(polar_path), cfg, "test")
        assert np.all(polar.cd > 0)

    def test_metadata(self, polar_path: Path) -> None:
        cfg = _default_config(reynolds=1e6, mach=0.0, n_crit=9.0)
        polar = _parse_polar_file(str(polar_path), cfg, "NACA 0012")
        assert polar.reynolds == 1e6
        assert polar.mach == 0.0
        assert polar.n_crit == 9.0
        assert polar.designation == "NACA 0012"

    def test_array_shapes(self, polar_path: Path) -> None:
        cfg = _default_config()
        polar = _parse_polar_file(str(polar_path), cfg, "test")
        n = polar.converged_count
        assert polar.alpha_deg.shape == (n,)
        assert polar.cl.shape == (n,)
        assert polar.cd.shape == (n,)
        assert polar.cm.shape == (n,)
        assert polar.top_xtr.shape == (n,)
        assert polar.bot_xtr.shape == (n,)


class TestPolarParsingEdgeCases:
    """Edge cases in polar parsing."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        cfg = _default_config()
        with pytest.raises(XfoilConvergenceError, match="no polar file"):
            _parse_polar_file(
                str(tmp_path / "nonexistent.dat"), cfg, "test",
            )

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.dat"
        p.write_text("header\n------\n")
        cfg = _default_config()
        with pytest.raises(XfoilConvergenceError, match="0/"):
            _parse_polar_file(str(p), cfg, "test")

    def test_nan_rows_skipped(self, tmp_path: Path) -> None:
        content = """\
 header
 ------- --------
   0.000   0.5000   0.00500   0.00200   0.0000  0.5000  0.5000
   2.000   NaN      NaN       NaN       NaN     NaN     NaN
   4.000   1.0000   0.00700   0.00300   0.0020  0.3000  0.7000
"""
        p = tmp_path / "nan.dat"
        p.write_text(content)
        cfg = _default_config()
        polar = _parse_polar_file(str(p), cfg, "test")
        assert polar.converged_count == 2  # NaN row skipped


# ===================================================================
# Exception hierarchy
# ===================================================================
class TestExceptions:
    """Exception class relationships."""

    def test_not_found_is_xfoil_error(self) -> None:
        assert issubclass(XfoilNotFoundError, XfoilError)

    def test_convergence_is_xfoil_error(self) -> None:
        assert issubclass(XfoilConvergenceError, XfoilError)

    def test_timeout_is_xfoil_error(self) -> None:
        assert issubclass(XfoilTimeoutError, XfoilError)

    def test_xfoil_error_is_runtime(self) -> None:
        assert issubclass(XfoilError, RuntimeError)


# ===================================================================
# Binary finding
# ===================================================================
class TestFindBinary:
    """Tests for find_xfoil_binary."""

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(XfoilNotFoundError):
            find_xfoil_binary("/no/such/path/xfoil_fake_1234")

    @pytest.mark.skipif(not _XFOIL_AVAILABLE, reason="XFOIL not installed")
    def test_auto_detect(self) -> None:
        path = find_xfoil_binary()
        assert os.path.isfile(path)


# ===================================================================
# Full run_xfoil (integration — requires XFOIL)
# ===================================================================
@pytest.mark.skipif(not _XFOIL_AVAILABLE, reason="XFOIL not installed")
class TestRunXfoilIntegration:
    """Integration tests — only run if XFOIL is on PATH."""

    def test_symmetric_polar(self) -> None:
        # NACA 0012 coordinates (Selig format, simplified)
        theta = np.linspace(0, np.pi, 50)
        x_upper = 0.5 * (1.0 - np.cos(theta))
        t = 0.12
        y_upper = 5 * t * (
            0.2969 * np.sqrt(x_upper)
            - 0.1260 * x_upper
            - 0.3516 * x_upper ** 2
            + 0.2843 * x_upper ** 3
            - 0.1036 * x_upper ** 4
        )
        x = np.concatenate([x_upper[::-1], x_upper[1:]])
        y = np.concatenate([y_upper[::-1], -y_upper[1:]])

        polar = run_xfoil(
            x, y, reynolds=1e6,
            alpha_start=0.0, alpha_end=6.0, alpha_step=2.0,
            designation="NACA 0012",
        )
        assert isinstance(polar, XfoilPolar)
        assert polar.converged_count >= 1
        assert np.all(polar.cl[polar.alpha_deg > 0] > 0)


# ===================================================================
# XfoilPolar dataclass
# ===================================================================
class TestXfoilPolar:
    """XfoilPolar structure tests."""

    def test_frozen(self) -> None:
        polar = XfoilPolar(
            alpha_deg=np.array([0.0]),
            cl=np.array([0.5]),
            cd=np.array([0.01]),
            cdp=np.array([0.005]),
            cm=np.array([-0.02]),
            top_xtr=np.array([0.5]),
            bot_xtr=np.array([0.5]),
            reynolds=1e6,
            mach=0.0,
            n_crit=9.0,
            designation="test",
            converged_count=1,
            total_count=1,
        )
        with pytest.raises(AttributeError):
            polar.reynolds = 2e6  # type: ignore[misc]
