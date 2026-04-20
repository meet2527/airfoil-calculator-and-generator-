"""
Powertrain & Speed Calculations Engine for RC Aircraft.

Calculates Battery output, Motor RPM, Pitch Speed, and Cruising Speed.
"""

from pydantic import BaseModel

class PowertrainSpecs(BaseModel):
    motor_kv: float
    esc_amps: float
    lipo_cells: int
    lipo_mah: float
    lipo_c_rating: float
    prop_diameter_in: float
    prop_pitch_in: float
    motor_size: str = "Unknown"

class PowertrainResult(BaseModel):
    nominal_voltage_v: float
    max_battery_amps: float
    esc_is_safe: bool
    no_load_rpm: float
    loaded_rpm: float
    v_max_ms: float
    v_cruise_ms: float
    static_thrust_original_kg: float = 0.0
    static_thrust_nominal_grams: float = 0.0
    static_thrust_max_grams: float = 0.0
    thrust_n_nominal: float = 0.0
    thrust_n_max: float = 0.0
    motor_size: str = "Unknown"

def calculate_powertrain(specs: PowertrainSpecs) -> PowertrainResult:
    # Constants from user-provided logic
    AIR_DENSITY = 1.225
    EFFICIENCY_THRUST = 0.85
    CT_DEFAULT = 0.12

    # 1. Battery Output
    nominal_voltage = specs.lipo_cells * 3.7
    max_voltage = specs.lipo_cells * 4.2
    max_battery_amps = (specs.lipo_mah * specs.lipo_c_rating) / 1000.0
    
    esc_is_safe = specs.esc_amps >= 10.0
    
    # 2. Motor RPM (Nominal for V_max/Cruise calculations)
    no_load_rpm = specs.motor_kv * nominal_voltage
    loaded_rpm = no_load_rpm * 0.80 # 80% efficiency for speed calculations
    
    # 3. Speeds
    v_max_ms = (loaded_rpm * specs.prop_pitch_in * 0.0254) / 60.0
    v_cruise_ms = v_max_ms * 0.60
    
    # 4. ORIGINAL Static Thrust Estimate (Empirical)
    static_thrust_original_kg = (specs.prop_diameter_in ** 3) * specs.prop_pitch_in * ((loaded_rpm / 1000.0) ** 2) * 2.83e-5 * 0.453592
    
    # 5. NEW Analytical Thrust Logic (at Nominal and Max voltage)
    def calc_analytical_thrust(volts):
        rpm = specs.motor_kv * volts * EFFICIENCY_THRUST
        n = rpm / 60.0  # rev/sec
        d_meters = specs.prop_diameter_in * 0.0254
        thrust_n = CT_DEFAULT * AIR_DENSITY * (n ** 2) * (d_meters ** 4)
        thrust_g = thrust_n * 102.0  # 1N approx 102g
        return thrust_n, thrust_g

    thrust_n_nom, thrust_g_nom = calc_analytical_thrust(nominal_voltage)
    thrust_n_max, thrust_g_max = calc_analytical_thrust(max_voltage)
    
    return PowertrainResult(
        nominal_voltage_v=nominal_voltage,
        max_battery_amps=max_battery_amps,
        esc_is_safe=esc_is_safe,
        no_load_rpm=no_load_rpm,
        loaded_rpm=loaded_rpm,
        v_max_ms=v_max_ms,
        v_cruise_ms=v_cruise_ms,
        static_thrust_original_kg=static_thrust_original_kg,
        static_thrust_nominal_grams=round(thrust_g_nom, 2),
        static_thrust_max_grams=round(thrust_g_max, 2),
        thrust_n_nominal=round(thrust_n_nom, 3),
        thrust_n_max=round(thrust_n_max, 3),
        motor_size=specs.motor_size
    )
