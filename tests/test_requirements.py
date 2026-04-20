"""
Tests for airfoil_config.requirements — W01 Wing loading calculator.

Covers:
    - ISA atmosphere model (sea-level, mid-altitude, tropopause, edge cases)
    - Wing loading computation
    - Reynolds number computation
    - Required CL computation
    - Stall speed computation
    - Wing area from aspect ratio
    - Dynamic pressure
    - Mach number
    - Purpose resolution
    - Full compute_wing_requirements integration
    - Edge-case / error handling (NaN, zero, negative, missing area)
"""

from __future__ import annotations

import math

import pytest

from airfoil_config.requirements import (
    AircraftPurpose,
    AircraftSpecs,
    AtmosphereProperties,
    WingRequirements,
    cl_max_for_purpose,
    compute_mach_number,
    compute_mean_aero_chord,
    compute_required_cl,
    compute_reynolds_number,
    compute_stall_speed,
    compute_wing_loading,
    compute_wing_requirements,
    dynamic_pressure,
    isa_atmosphere,
    resolve_purpose,
    wing_area_from_aspect_ratio,
)


# ===================================================================
# Helpers
# ===================================================================
def _sea_level_atm() -> AtmosphereProperties:
    """Return ISA sea-level atmosphere for reuse in tests."""
    return isa_atmosphere(0.0)


# Tolerance for floating-point comparisons
REL_TOL = 1e-4   # 0.01 %
ABS_TOL = 1e-8


# ===================================================================
# ISA Atmosphere
# ===================================================================
class TestIsaAtmosphere:
    """Tests for the ISA standard-atmosphere function."""

    def test_sea_level_temperature(self) -> None:
        """Sea-level temperature should be 288.15 K."""
        atm = isa_atmosphere(0.0)
        assert atm.temperature_k == pytest.approx(288.15, rel=REL_TOL)

    def test_sea_level_pressure(self) -> None:
        """Sea-level pressure should be 101325 Pa."""
        atm = isa_atmosphere(0.0)
        assert atm.pressure_pa == pytest.approx(101325.0, rel=REL_TOL)

    def test_sea_level_density(self) -> None:
        """Sea-level density should be ~1.225 kg/m³."""
        atm = isa_atmosphere(0.0)
        assert atm.density_kg_m3 == pytest.approx(1.225, rel=REL_TOL)

    def test_sea_level_speed_of_sound(self) -> None:
        """Sea-level speed of sound should be ~340.3 m/s."""
        atm = isa_atmosphere(0.0)
        assert atm.speed_of_sound_ms == pytest.approx(340.3, rel=1e-3)

    def test_sea_level_viscosity(self) -> None:
        """Sea-level viscosity should be ~1.789e-5 Pa·s."""
        atm = isa_atmosphere(0.0)
        assert atm.dynamic_viscosity_pa_s == pytest.approx(
            1.789e-5, rel=1e-2
        )

    def test_mid_altitude_temperature(self) -> None:
        """At 5000 m, T should be 288.15 - 0.0065*5000 = 255.65 K."""
        atm = isa_atmosphere(5000.0)
        assert atm.temperature_k == pytest.approx(255.65, rel=REL_TOL)

    def test_mid_altitude_density_decreases(self) -> None:
        """Density at 5000 m should be less than at sea level."""
        atm_sl = isa_atmosphere(0.0)
        atm_5k = isa_atmosphere(5000.0)
        assert atm_5k.density_kg_m3 < atm_sl.density_kg_m3

    def test_tropopause(self) -> None:
        """At 11000 m the function should still work (boundary)."""
        atm = isa_atmosphere(11000.0)
        expected_t = 288.15 - 0.0065 * 11000.0  # 216.65 K
        assert atm.temperature_k == pytest.approx(expected_t, rel=REL_TOL)

    def test_negative_altitude_raises(self) -> None:
        """Negative altitudes are not physical — should raise."""
        with pytest.raises(ValueError, match="must be >= 0"):
            isa_atmosphere(-100.0)

    def test_above_troposphere_raises(self) -> None:
        """Above 11 km we don't model the stratosphere yet."""
        with pytest.raises(ValueError, match="troposphere"):
            isa_atmosphere(12000.0)

    def test_nan_altitude_raises(self) -> None:
        """NaN altitude should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            isa_atmosphere(float("nan"))


# ===================================================================
# Wing Loading
# ===================================================================
class TestWingLoading:
    """Tests for wing loading W/S."""

    def test_basic(self) -> None:
        """10000 N / 16 m² = 625 Pa."""
        assert compute_wing_loading(10000.0, 16.0) == pytest.approx(625.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="wing_area_m2"):
            compute_wing_loading(10000.0, 0.0)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_n"):
            compute_wing_loading(-1.0, 16.0)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compute_wing_loading(float("nan"), 16.0)


# ===================================================================
# Reynolds Number
# ===================================================================
class TestReynoldsNumber:
    """Tests for Reynolds number Re = ρVc/μ."""

    def test_sea_level_typical(self) -> None:
        """Typical GA: V=60 m/s, c=1.5 m at sea level → Re ≈ 6.15e6."""
        atm = _sea_level_atm()
        re = compute_reynolds_number(60.0, 1.5, atm)
        assert re == pytest.approx(6.15e6, rel=2e-2)

    def test_increases_with_velocity(self) -> None:
        """Re should scale linearly with velocity."""
        atm = _sea_level_atm()
        re_30 = compute_reynolds_number(30.0, 1.0, atm)
        re_60 = compute_reynolds_number(60.0, 1.0, atm)
        assert re_60 == pytest.approx(2.0 * re_30, rel=REL_TOL)

    def test_zero_chord_raises(self) -> None:
        with pytest.raises(ValueError, match="chord_m"):
            compute_reynolds_number(60.0, 0.0, _sea_level_atm())


# ===================================================================
# Required CL
# ===================================================================
class TestRequiredCL:
    """Tests for required CL = 2W / (ρV²S)."""

    def test_level_flight(self) -> None:
        """Known case: W=10000 N, V=60 m/s, S=16 m², sea-level."""
        atm = _sea_level_atm()
        cl = compute_required_cl(10000.0, 60.0, 16.0, atm)
        # q = 0.5 * 1.225 * 3600 = 2205, CL = 10000 / (2205 * 16) ≈ 0.2834
        expected = 10000.0 / (0.5 * atm.density_kg_m3 * 60.0**2 * 16.0)
        assert cl == pytest.approx(expected, rel=REL_TOL)

    def test_increases_with_weight(self) -> None:
        """Doubling weight should double CL."""
        atm = _sea_level_atm()
        cl_1 = compute_required_cl(5000.0, 60.0, 16.0, atm)
        cl_2 = compute_required_cl(10000.0, 60.0, 16.0, atm)
        assert cl_2 == pytest.approx(2.0 * cl_1, rel=REL_TOL)

    def test_zero_velocity_raises(self) -> None:
        with pytest.raises(ValueError, match="velocity_ms"):
            compute_required_cl(10000.0, 0.0, 16.0, _sea_level_atm())


# ===================================================================
# Mean Aerodynamic Chord
# ===================================================================
class TestMeanAeroChord:
    """Tests for simplified MAC = S/b."""

    def test_basic(self) -> None:
        """16 m² / 10 m = 1.6 m."""
        assert compute_mean_aero_chord(16.0, 10.0) == pytest.approx(1.6)

    def test_zero_span_raises(self) -> None:
        with pytest.raises(ValueError, match="wing_span_m"):
            compute_mean_aero_chord(16.0, 0.0)


# ===================================================================
# Stall Speed
# ===================================================================
class TestStallSpeed:
    """Tests for V_s = sqrt(2W / (ρ S CL_max))."""

    def test_sea_level_ga(self) -> None:
        """Typical GA: W=10000, S=16, CL_max=1.5, sea-level."""
        atm = _sea_level_atm()
        vs = compute_stall_speed(10000.0, 16.0, 1.5, atm)
        expected = math.sqrt(2.0 * 10000.0 / (1.225 * 16.0 * 1.5))
        assert vs == pytest.approx(expected, rel=REL_TOL)

    def test_higher_cl_max_lowers_stall_speed(self) -> None:
        """Higher CL_max → lower stall speed."""
        atm = _sea_level_atm()
        vs_low = compute_stall_speed(10000.0, 16.0, 1.2, atm)
        vs_high = compute_stall_speed(10000.0, 16.0, 1.8, atm)
        assert vs_high < vs_low

    def test_zero_cl_max_raises(self) -> None:
        with pytest.raises(ValueError, match="cl_max"):
            compute_stall_speed(10000.0, 16.0, 0.0, _sea_level_atm())


# ===================================================================
# Wing Area from Aspect Ratio
# ===================================================================
class TestWingAreaFromAR:
    """Tests for S = b²/AR."""

    def test_basic(self) -> None:
        """b=10, AR=6.25 → S = 100/6.25 = 16."""
        assert wing_area_from_aspect_ratio(10.0, 6.25) == pytest.approx(16.0)

    def test_negative_ar_raises(self) -> None:
        with pytest.raises(ValueError, match="aspect_ratio"):
            wing_area_from_aspect_ratio(10.0, -1.0)


# ===================================================================
# Dynamic Pressure
# ===================================================================
class TestDynamicPressure:
    """Tests for q = ½ρV²."""

    def test_sea_level(self) -> None:
        """V=60 m/s, sea-level → q = 0.5*1.225*3600 = 2205 Pa."""
        atm = _sea_level_atm()
        q = dynamic_pressure(60.0, atm)
        assert q == pytest.approx(2205.0, rel=REL_TOL)


# ===================================================================
# Mach Number
# ===================================================================
class TestMachNumber:
    """Tests for M = V / a."""

    def test_subsonic(self) -> None:
        """V=60 m/s, a≈340.3 m/s → M ≈ 0.176."""
        atm = _sea_level_atm()
        m = compute_mach_number(60.0, atm)
        assert m == pytest.approx(60.0 / 340.3, rel=1e-2)


# ===================================================================
# Purpose Resolution
# ===================================================================
class TestResolvePurpose:
    """Tests for purpose string → enum conversion."""

    def test_valid(self) -> None:
        assert resolve_purpose("general_aviation") is AircraftPurpose.GENERAL_AVIATION

    def test_case_insensitive(self) -> None:
        assert resolve_purpose("  UAV  ") is AircraftPurpose.UAV

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aircraft purpose"):
            resolve_purpose("fighter_jet")


class TestClMaxForPurpose:
    """Tests for purpose → CL_max lookup."""

    def test_ga(self) -> None:
        assert cl_max_for_purpose(AircraftPurpose.GENERAL_AVIATION) == 1.5

    def test_glider(self) -> None:
        assert cl_max_for_purpose(AircraftPurpose.GLIDER) == 1.4


# ===================================================================
# Integration — compute_wing_requirements
# ===================================================================
class TestComputeWingRequirements:
    """Integration tests for the main entry point."""

    @pytest.fixture()
    def ga_specs(self) -> AircraftSpecs:
        """General-aviation aircraft specs with explicit area."""
        return AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            wing_area_m2=16.0,
            cruise_altitude_m=3000.0,
            cruise_velocity_ms=60.0,
            purpose="general_aviation",
        )

    def test_wing_loading(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.wing_loading_pa == pytest.approx(625.0)

    def test_mac(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.mean_aero_chord_m == pytest.approx(1.6)

    def test_aspect_ratio(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.aspect_ratio == pytest.approx(6.25)

    def test_reynolds_positive(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.reynolds_number > 1e5

    def test_required_cl_reasonable(self, ga_specs: AircraftSpecs) -> None:
        """CL should be between 0 and CL_max for a sensible design."""
        reqs = compute_wing_requirements(ga_specs)
        assert 0.0 < reqs.required_cl_cruise < reqs.cl_max_estimate

    def test_stall_speed_below_cruise(self, ga_specs: AircraftSpecs) -> None:
        """Stall speed should be below cruise speed for a stable design."""
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.stall_speed_ms < ga_specs.cruise_velocity_ms

    def test_mach_subsonic(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert reqs.mach_number < 1.0

    def test_atmosphere_returned(self, ga_specs: AircraftSpecs) -> None:
        reqs = compute_wing_requirements(ga_specs)
        assert isinstance(reqs.atmosphere, AtmosphereProperties)
        assert reqs.atmosphere.temperature_k < 288.15  # colder at altitude

    def test_cl_max_override(self) -> None:
        """cl_max_override should take priority over purpose defaults."""
        specs = AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            wing_area_m2=16.0,
            cruise_altitude_m=0.0,
            cruise_velocity_ms=60.0,
            purpose="general_aviation",
            cl_max_override=2.0,
        )
        reqs = compute_wing_requirements(specs)
        assert reqs.cl_max_estimate == 2.0

    def test_stall_speed_override(self) -> None:
        """stall_speed_ms_override should bypass computation."""
        specs = AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            wing_area_m2=16.0,
            cruise_altitude_m=0.0,
            cruise_velocity_ms=60.0,
            purpose="general_aviation",
            stall_speed_ms_override=25.0,
        )
        reqs = compute_wing_requirements(specs)
        assert reqs.stall_speed_ms == 25.0

    def test_area_from_aspect_ratio(self) -> None:
        """If wing_area_m2 is None, derive from span + AR."""
        specs = AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            aspect_ratio=6.25,
            cruise_altitude_m=0.0,
            cruise_velocity_ms=60.0,
            purpose="general_aviation",
        )
        reqs = compute_wing_requirements(specs)
        # S = 10² / 6.25 = 16 → W/S = 10000/16 = 625
        assert reqs.wing_loading_pa == pytest.approx(625.0)

    def test_no_area_no_ar_raises(self) -> None:
        """Must provide either wing_area_m2 or aspect_ratio."""
        specs = AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            cruise_altitude_m=0.0,
            cruise_velocity_ms=60.0,
            purpose="general_aviation",
        )
        with pytest.raises(ValueError, match="wing_area_m2 or aspect_ratio"):
            compute_wing_requirements(specs)

    def test_unknown_purpose_raises(self) -> None:
        """Invalid purpose string should propagate ValueError."""
        specs = AircraftSpecs(
            weight_n=10000.0,
            wing_span_m=10.0,
            wing_area_m2=16.0,
            cruise_altitude_m=0.0,
            cruise_velocity_ms=60.0,
            purpose="stealth_bomber",
        )
        with pytest.raises(ValueError, match="Unknown aircraft purpose"):
            compute_wing_requirements(specs)

    def test_uav_purpose(self) -> None:
        """UAV purpose should use CL_max=1.2."""
        specs = AircraftSpecs(
            weight_n=50.0,
            wing_span_m=2.0,
            wing_area_m2=0.5,
            cruise_altitude_m=500.0,
            cruise_velocity_ms=20.0,
            purpose="uav",
        )
        reqs = compute_wing_requirements(specs)
        assert reqs.cl_max_estimate == 1.2

    def test_glider_high_ar(self) -> None:
        """Glider with high aspect ratio should produce valid results."""
        specs = AircraftSpecs(
            weight_n=4000.0,
            wing_span_m=18.0,
            aspect_ratio=20.0,
            cruise_altitude_m=1500.0,
            cruise_velocity_ms=30.0,
            purpose="glider",
        )
        reqs = compute_wing_requirements(specs)
        assert reqs.aspect_ratio == pytest.approx(20.0)
        assert reqs.wing_loading_pa > 0
        assert reqs.stall_speed_ms < specs.cruise_velocity_ms


# ===================================================================
# Edge cases with inf
# ===================================================================
class TestInfEdgeCases:
    """Ensure inf values are rejected."""

    def test_inf_weight(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compute_wing_loading(float("inf"), 16.0)

    def test_inf_velocity(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compute_reynolds_number(float("inf"), 1.0, _sea_level_atm())
