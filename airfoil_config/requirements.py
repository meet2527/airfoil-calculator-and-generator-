"""
Wing loading, Reynolds number, and required CL calculator for RC Planes.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel
import math

class RCFlightCategory(str, Enum):
    GLIDER = "glider"
    TRAINER = "trainer"
    SPORT = "sport"
    RACER = "racer"
    CARGO = "cargo"

class RcaSpecs(BaseModel):
    weight_kg: float
    wing_area_dm2: float
    mac_m: float
    v_cruise_ms: float
    intended_category: str = "auto"

class RcaRequirements(BaseModel):
    wing_loading_g_dm2: float
    cubic_wing_loading: float
    category: RCFlightCategory
    reynolds_number: float
    required_cl_cruise: float
    v_stall_ms: float = 0.0

def compute_wing_requirements(specs: RcaSpecs) -> RcaRequirements:
    # 1. Wing Loading (g/dm^2 is common for RC)
    weight_g = specs.weight_kg * 1000.0
    wing_loading_g_dm2 = weight_g / specs.wing_area_dm2
    
    # 2. Cubic Wing Loading (often in oz/ft^3, but prompt asked for simple Weight / Area^1.5)
    # To match standard RC CWL metrics loosely, if we just use the user formula mathematically:
    cwl = specs.weight_kg / (specs.wing_area_dm2 ** 1.5)
    # However standard CWL WCF is WCF = W(oz) / (Area(sq ft))^1.5. 
    # Let's map it based on metric: 
    # 1 kg = 35.27 oz, 1 dm^2 = 0.1076 ft^2
    weight_oz = specs.weight_kg * 35.274
    area_sqft = specs.wing_area_dm2 * 0.107639
    standard_cwl = weight_oz / (area_sqft ** 1.5)
    
    if specs.intended_category != "auto":
        try:
            category = RCFlightCategory(specs.intended_category.lower())
        except ValueError:
            category = RCFlightCategory.TRAINER
    else:
        if standard_cwl < 5.0:
            category = RCFlightCategory.GLIDER
        elif standard_cwl < 8.0:
            category = RCFlightCategory.TRAINER
        elif standard_cwl < 14.0:
            category = RCFlightCategory.SPORT
        else:
            category = RCFlightCategory.RACER

    # 3. Stall Speed Calculation (Assume CL_max = 1.2 for standard RC planes without flaps)
    cl_max_est = 1.2
    wing_area_m2 = specs.wing_area_dm2 / 100.0
    v_stall_ms = math.sqrt((2.0 * specs.weight_kg * 9.81) / (1.225 * wing_area_m2 * cl_max_est))
    
    # 4. Aerodynamic Validation: Cruise must always be > 1.2 * Stall Speed for level flight
    actual_v_cruise = max(specs.v_cruise_ms, v_stall_ms * 1.2)

    # 5. Reynolds Number at Actual Cruise
    re = (actual_v_cruise * specs.mac_m) / 1.46e-5

    # 6. Required CL at Actual Cruise
    cl = (2.0 * specs.weight_kg * 9.81) / (1.225 * (actual_v_cruise ** 2) * wing_area_m2)

    return RcaRequirements(
        wing_loading_g_dm2=wing_loading_g_dm2,
        cubic_wing_loading=standard_cwl,
        category=category,
        reynolds_number=re,
        required_cl_cruise=cl,
        v_stall_ms=v_stall_ms
    )
