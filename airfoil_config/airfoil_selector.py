"""
RC Airfoil Selection Logic.

Evaluates curated RC airfoils against calculated Reynolds Number
and Required Cruise CL.
"""

from typing import List
from pydantic import BaseModel
from airfoil_config.powertrain import PowertrainResult
from airfoil_config.requirements import RcaRequirements, RCFlightCategory

class AirfoilRecommendation(BaseModel):
    name: str
    description: str
    reasons: List[str]
    score: float
    coordinates: List[List[float]] = []

class SelectionResult(BaseModel):
    candidates: List[AirfoilRecommendation]

# Curated RC Airfoil Database with Expert Insights
RC_AIRFOILS = [
    {
        "name": "NACA 6409",
        "description": "Legendary high-lift NACA airfoil for free-flight and RC cargo. 6% camber perfectly targets CL=0.6 to 0.8, providing a 100% fit for slow heavy lifting.",
        "category": [RCFlightCategory.CARGO, RCFlightCategory.TRAINER],
        "optimal_cl_range": (0.50, 1.00),
        "cl_max_est": 1.5,
        "min_re": 80000,
        "max_re": 800000,
        "insight_seeds": [
            "Heavy-lift master: The 6% camber is ideal for payload missions.",
            "High drag at high speed: Best kept at lower throttle settings.",
            "Superior slow-flight handling due to its forgiving leading edge."
        ]
    },
    {
        "name": "NACA 4412",
        "description": "Standard workhorse NACA airfoil. Excellent lifting capacity with predictable stall for stable dropping.",
        "category": [RCFlightCategory.CARGO, RCFlightCategory.SPORT],
        "optimal_cl_range": (0.35, 0.75),
        "cl_max_est": 1.4,
        "min_re": 100000,
        "max_re": 1000000,
        "insight_seeds": [
            "Predictable stall: Provides a clear tactile 'mushy' feel before the wing drops.",
            "Versatile section: Operates efficiently across a wide speed range.",
            "Robust thickness: Allows for strong wing spars and easy building."
        ]
    },
    {
        "name": "Selig S1223",
        "description": "Extreme high-lift, highly cambered airfoil used in SAE Heavy Lift. Incredible low-speed lift perfect for slowing down efficiently.",
        "category": [RCFlightCategory.CARGO],
        "optimal_cl_range": (0.8, 1.6),
        "cl_max_est": 1.9,
        "min_re": 100000,
        "max_re": 800000,
        "insight_seeds": [
            "Maximum Lift: Use this if you are carrying extreme payloads.",
            "High pitching moment: Requires a large horizontal stabilizer for trim.",
            "Draggy profile: Demands consistent power to maintain flight."
        ]
    },
    {
        "name": "Clark Y",
        "description": "Classic flat-bottom airfoil. Exceptional gentle stall characteristics.",
        "category": [RCFlightCategory.TRAINER, RCFlightCategory.SPORT, RCFlightCategory.CARGO],
        "optimal_cl_range": (0.4, 0.7),
        "cl_max_est": 1.3,
        "min_re": 50000,
        "max_re": 1000000,
        "insight_seeds": [
            "Flat bottom: Simplifies construction on flat workbenches.",
            "Gentle stall: Naturally recovers with minimal altitude loss.",
            "Proven heritage: The most widely used RC airfoil for a reason."
        ]
    },
    {
        "name": "NACA 2412",
        "description": "Semi-symmetrical classic. Great all-rounder for sport flying.",
        "category": [RCFlightCategory.SPORT, RCFlightCategory.TRAINER],
        "optimal_cl_range": (0.2, 0.5),
        "cl_max_est": 1.25,
        "min_re": 100000,
        "max_re": 1000000,
        "insight_seeds": [
            "Semi-symmetrical: Offers good inverted flight performance.",
            "Aerobatic ready: Clean response in rolls and loops.",
            "Low drag: Helps maximize flight time at high speed."
        ]
    },
    {
        "name": "NACA 0015",
        "description": "Thick symmetrical airfoil. Perfect for 3D aerobatics.",
        "category": [RCFlightCategory.SPORT],
        "optimal_cl_range": (-0.2, 0.2),
        "cl_max_est": 1.1,
        "min_re": 150000,
        "max_re": 1000000,
        "insight_seeds": [
            "Fully symmetrical: Identical behavior upright or inverted.",
            "Pitch sensitive: Great for tight loops and waterfalls.",
            "Stall behavior: Crisp stall is ideal for snap rolls."
        ]
    },
    {
        "name": "MH32",
        "description": "Thin, highly cambered section. Excellent drag bucket for speed and glide.",
        "category": [RCFlightCategory.GLIDER, RCFlightCategory.RACER],
        "optimal_cl_range": (0.3, 0.6),
        "cl_max_est": 1.2,
        "min_re": 100000,
        "max_re": 800000,
        "insight_seeds": [
            "Low Drag Bucket: Stays extremely efficient at specific speeds.",
            "Glider performance: Excellent energy retention in turns.",
            "Thermal optimized: Climbs well in weak lift conditions."
        ]
    }, 
    {
        "name": "Drela AG35",
        "description": "Specifically engineered for extremely low Re and built-up wings.",
        "category": [RCFlightCategory.GLIDER],
        "optimal_cl_range": (0.4, 0.8),
        "cl_max_est": 1.2,
        "min_re": 30000,
        "max_re": 400000,
        "insight_seeds": [
            "Ultra-low Re: Engineered to prevent separation at low speeds.",
            "Poly-hedral ready: Excellent for built-up balsa wings.",
            "Lightweight specialist: Best for lightweight soaring."
        ]
    },
    {
        "name": "RG15",
        "description": "Fast glider and pylon racer airfoil.",
        "category": [RCFlightCategory.GLIDER, RCFlightCategory.RACER],
        "optimal_cl_range": (0.1, 0.4),
        "cl_max_est": 1.15,
        "min_re": 150000,
        "max_re": 1000000,
        "insight_seeds": [
            "Pylon racing DNA: Built for maximum speed in the turns.",
            "Retains energy: Minimal drag at low angles of attack.",
            "Penetrates wind: Great for flying in gusty conditions."
        ]
    }
]

def select_rc_airfoils(req: RcaRequirements, pt: PowertrainResult, max_candidates: int = 3) -> SelectionResult:
    from airfoil_config.geometry_data import get_airfoil_coordinates
    re = req.reynolds_number
    target_cl = req.required_cl_cruise
    plane_cat = req.category
    
    scored_candidates = []
    
    for af in RC_AIRFOILS:
        score = 0.0
        reasons = []
        
        # 1. Category check
        cat_match = plane_cat in af["category"]
        if cat_match:
            score += 30.0
        
        # 2. Re Check
        if re >= af["min_re"] and re <= af["max_re"]:
            score += 40.0
        elif re < af["min_re"]:
            score -= 50.0
        else:
            score += 10.0
            
        # 3. CL Drag Bucket Check
        cl_min, cl_max = af["optimal_cl_range"]
        cl_efficiency = 0.0
        if cl_min <= target_cl <= cl_max:
            score += 30.0
            cl_efficiency = 1.0 # Optimal
        else:
            dist = min(abs(target_cl - cl_min), abs(target_cl - cl_max))
            penalty = min(dist * 50.0, 30.0)
            score -= penalty
            cl_efficiency = max(0.0, 1.0 - dist * 2.0)
            
        # --- DYNAMIC REASON GENERATION ---
        
        # Reason 1: Aero-Physics (Stall & Re)
        stall_margin = af["cl_max_est"] - target_cl
        re_safety = "excellent" if (af["min_re"] * 1.5 < re < af["max_re"] * 0.8) else "stable"
        reasons.append(f"At Re={int(re/1000)}k, this profile is {re_safety} with a {stall_margin:.2f} CL stall margin.")
        
        # Reason 2: Hardware Synergy (Motor & Thrust)
        performance_match = "Efficiency" if cl_efficiency > 0.8 else "Stability"
        thrust_grams = pt.static_thrust_nominal_grams
        reasons.append(f"Pairs {performance_match} with your {pt.motor_size} motor's {int(thrust_grams)}g thrust.")
        
        # Reason 3: Expert Insight (Unique to airfoil)
        # We pick one expert insight seed based on the situation
        if not cat_match:
            reasons.append(af["insight_seeds"][1]) # Usually the 'downside' or 'specific use' one
        elif cl_efficiency > 0.9:
            reasons.append(af["insight_seeds"][0]) # Strengths
        else:
            reasons.append(af["insight_seeds"][2]) # Handling traits
            
        scored_candidates.append({
            "af": af,
            "score": score,
            "reasons": reasons
        })
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Top candidates
    top = scored_candidates[:max_candidates]
    rec_models = [
        AirfoilRecommendation(
            name=c["af"]["name"],
            description=c["af"]["description"],
            reasons=c["reasons"],
            score=c["score"],
            coordinates=get_airfoil_coordinates(c["af"]["name"])
        )
        for c in top
    ]
    
    return SelectionResult(candidates=rec_models)
