"""
FastAPI Backend for RC Airfoil Configuration and Analysis.
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from typing import Literal, Optional

from airfoil_config.powertrain import PowertrainSpecs, calculate_powertrain
from airfoil_config.requirements import RcaSpecs, compute_wing_requirements
from airfoil_config.airfoil_selector import select_rc_airfoils
from airfoil_config.generator import (
    generate_naca4, extrude_airfoil, 
    to_dat_file, to_csv_file, to_ascii_stl, scale_coordinates
)

app = FastAPI(
    title="RC Airfoil Config API",
    description="Powertrain & Aerodynamics Calculator for RC Modellers",
    version="2.1.0",
)

def resolve_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serve the interactive airfoil designer dashboard."""
    path = resolve_path(os.path.join("templates", "index.html"))
    if not os.path.exists(path):
        return HTMLResponse(f"Templates missing at {path}. Run in project root.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class RCAircraftRequest(BaseModel):
    # Airframe — raw user inputs (geometry computed server-side)
    weight_kg: float = Field(..., gt=0.0, description="Estimated Total Flying Weight in kg")
    wing_span_mm: float = Field(..., gt=0.0, description="Full Wing Span in mm")
    chord_mm: float = Field(..., gt=0.0, description="Root/Mean Chord in mm")
    wing_shape: Literal["Rectangular", "Tapered", "Elliptical"] = Field(
        "Rectangular", description="Wing planform shape"
    )

    # Powertrain
    motor_kv: float = Field(..., gt=0.0, description="Motor KV")
    esc_amps: float = Field(..., gt=0.0, description="ESC Rating (Amps)")
    lipo_cells: int = Field(..., gt=0, description="Battery Cell count (S)")
    lipo_mah: float = Field(..., gt=0.0, description="Battery Capacity (mAh)")
    lipo_c_rating: float = Field(..., gt=0.0, description="Battery C-Rating")
    prop_diameter_in: float = Field(..., gt=0.0, description="Propeller Diameter (inches)")
    prop_pitch_in: float = Field(..., gt=0.0, description="Propeller Pitch (inches)")
    motor_size: str = Field("Unknown", description="Motor size (e.g. 2212)")
    intended_category: str = Field("auto", description="Intended flight category or 'auto'")


def _compute_wing_geometry(req: RCAircraftRequest) -> tuple[float, float]:
    """Compute wing_area_dm2 and mac_m from raw user inputs."""
    import math
    span_m = req.wing_span_mm / 1000.0
    chord_m = req.chord_mm / 1000.0

    if req.wing_shape == "Rectangular":
        wing_area_m2 = span_m * chord_m
        mac_m = chord_m
    elif req.wing_shape == "Tapered":
        taper_ratio = 0.7
        wing_area_m2 = span_m * chord_m * (1.0 + taper_ratio) / 2.0
        mac_m = (2.0 / 3.0) * chord_m * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    else:  # Elliptical
        wing_area_m2 = (math.pi / 4.0) * span_m * chord_m
        mac_m = (math.pi / 4.0) * chord_m

    wing_area_dm2 = wing_area_m2 * 100.0
    return wing_area_dm2, mac_m

@app.post("/design")
def design_rc_plane(req: RCAircraftRequest):
    """Orchestrator: Runs the RC calculations according to the exact expert workflow."""
    try:
        wing_area_dm2, mac_m = _compute_wing_geometry(req)

        pt_specs = PowertrainSpecs(
            motor_kv=req.motor_kv,
            esc_amps=req.esc_amps,
            lipo_cells=req.lipo_cells,
            lipo_mah=req.lipo_mah,
            lipo_c_rating=req.lipo_c_rating,
            prop_diameter_in=req.prop_diameter_in,
            prop_pitch_in=req.prop_pitch_in,
            motor_size=req.motor_size,
        )
        pt_res = calculate_powertrain(pt_specs)

        rc_specs = RcaSpecs(
            weight_kg=req.weight_kg,
            wing_area_dm2=wing_area_dm2,
            mac_m=mac_m,
            v_cruise_ms=pt_res.v_cruise_ms,
            intended_category=req.intended_category,
        )
        reqs_res = compute_wing_requirements(rc_specs)

        af_res = select_rc_airfoils(reqs_res, pt_res, max_candidates=6)

        return {
            "powertrain": pt_res.dict(),
            "aerodynamics": {
                **reqs_res.dict(),
                "v_cruise_ms": pt_res.v_cruise_ms,
                "target_cl": reqs_res.required_cl_cruise,
                "wing_area_dm2": round(wing_area_dm2, 2),
                "mac_m": round(mac_m, 4),
                "wing_span_mm": req.wing_span_mm,
                "chord_mm": req.chord_mm,
                "wing_shape": req.wing_shape,
            },
            "airfoils": af_res.dict()["candidates"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class GeneratorRequest(BaseModel):
    code: str = Field(..., description="NACA 4-digit code")
    chord_mm: float = Field(200.0, gt=0)
    span_mm: float = Field(400.0, gt=0)

@app.post("/generator/calculate")
def generator_calculate(req: GeneratorRequest):
    """Generate 2D airfoil and 3D mesh points for the frontend."""
    data = generate_naca4(req.code)
    if not data:
        raise HTTPException(status_code=400, detail="Invalid NACA 4-digit code")
    
    mesh = extrude_airfoil(data["loop"], req.chord_mm, req.span_mm)
    return {"data": data, "mesh": mesh, "chord": req.chord_mm, "span": req.span_mm}

@app.get("/generator/export")
def generator_export(
    code: str = Query(...), 
    chord: float = Query(...), 
    span: float = Query(...), 
    format: str = Query(...)
):
    """Generate and return files for download (dat, csv, stl)."""
    data = generate_naca4(code)
    if not data:
        raise HTTPException(status_code=400, detail="Invalid NACA 4-digit code")
    
    if format == "dat":
        scaled = scale_coordinates(data["loop"], chord)
        content = to_dat_file(f"NACA {data['params']['code']} (chord={chord}mm)", scaled)
        return Response(content, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=naca{code}_chord{chord}.dat"})
    elif format == "csv":
        scaled = scale_coordinates(data["loop"], chord)
        content = to_csv_file(scaled)
        return Response(content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=naca{code}_chord{chord}.csv"})
    elif format == "stl":
        mesh = extrude_airfoil(data["loop"], chord, span)
        content = to_ascii_stl(f"NACA_{data['params']['code']}", mesh["vertices"], mesh["faces"])
        return Response(content, media_type="model/stl", headers={"Content-Disposition": f"attachment; filename=naca{code}_chord{chord}_span{span}.stl"})
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
