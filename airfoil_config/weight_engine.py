"""
Weight and Power System Estimation Engine (W15).

Provides heuristic and direct methods for Calculating Total Takeoff Weight (TOW) 
based on airframe (fuselage), power system (motor/ESC), and energy storage (battery).
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class PowerSystemSpecs:
    """Inputs for weight and power estimation."""
    fuselage_mass_g: float
    motor_kv: Optional[float] = None
    esc_max_amps: Optional[float] = None
    lipo_cells: Optional[int] = None
    lipo_mah: Optional[float] = None
    lipo_c_rating: Optional[float] = None
    payload_mass_g: float = 0.0

@dataclass
class WeightBreakdown:
    """Resulting weight estimation summary."""
    fuselage_g: float
    battery_g: float
    motor_g: float
    esc_g: float
    payload_g: float
    total_g: float
    total_n: float


def estimate_lipo_mass_g(cells: int, mah: float) -> float:
    """Estimate LiPo weight based on cells and capacity.
    
    Heuristic: ~21g per 100mAh-S (Standard 150-180 Wh/kg pack).
    """
    if cells <= 0 or mah <= 0:
        return 0.0
    # Approx 21g per 1000 mAh per cell (0.021 g/mAh-S)
    return cells * mah * 0.021


def estimate_motor_mass_g(kv: float, target_mass_kg: float) -> float:
    """Estimate motor weight if not provided.
    
    Heuristic: Trainer motors are roughly 60-120g for a 1.5kg plane.
    """
    if target_mass_kg <= 0:
        return 80.0
    # Power required ~ 200W/kg. 
    # Motor weight ~ 0.25 g per Watt capacity.
    power_w = target_mass_kg * 250.0 
    return power_w * 0.25


def estimate_esc_mass_g(amps: float) -> float:
    """Estimate ESC weight based on max amperage."""
    if amps <= 0:
        return 40.0
    # Heuristic: 1.2g per Amp + 10g base weight
    return (amps * 1.2) + 10.0


def calculate_tow(specs: PowerSystemSpecs) -> WeightBreakdown:
    """Calculate the Total Takeoff Weight (TOW) from components."""
    # 1. Start with fuselage
    f_mass = specs.fuselage_mass_g
    
    # 2. Battery
    if specs.lipo_cells and specs.lipo_mah:
        b_mass = estimate_lipo_mass_g(specs.lipo_cells, specs.lipo_mah)
    else:
        b_mass = 200.0 # Default for ~2200mAh 3S
        
    # 3. Motor (using provided mass or estimating from KV + preliminary TOW)
    # Note: Preliminary TOW for motor estimation uses fuselage + battery
    prelim_tow_kg = (f_mass + b_mass) / 1000.0
    m_mass = estimate_motor_mass_g(specs.motor_kv or 1000.0, prelim_tow_kg)
    
    # 4. ESC
    e_mass = estimate_esc_mass_g(specs.esc_max_amps or 40.0)
    
    # 5. Sum it all up
    total_g = f_mass + b_mass + m_mass + e_mass + specs.payload_mass_g
    
    return WeightBreakdown(
        fuselage_g=f_mass,
        battery_g=b_mass,
        motor_g=m_mass,
        esc_g=e_mass,
        payload_g=specs.payload_mass_g,
        total_g=total_g,
        total_n=total_g * 0.00980665 # g -> N
    )
