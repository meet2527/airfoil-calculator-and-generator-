"""
Tests for API endpoints (api.py).
"""

from fastapi.testclient import TestClient
import pytest

from api import app

client = TestClient(app)

def test_requirements_endpoint():
    payload = {
        "weight_n": 1.5 * 9.81,
        "wing_span_m": 1.0,
        "cruise_velocity_ms": 12.0,
        "purpose": "trainer",
        "wing_area_m2": 0.22
    }
    response = client.post("/requirements", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "wing_loading_pa" in data
    assert "reynolds_number" in data
    assert data["reynolds_number"] > 0

def test_select_endpoint():
    payload = {
        "wing_loading_pa": 66.8,
        "reynolds_number": 180000,
        "required_cl_cruise": 0.35,
        "max_candidates": 3
    }
    response = client.post("/select", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) <= 3

def test_geometry_endpoint():
    response = client.get("/geometry/NACA 2412")
    assert response.status_code == 200
    data = response.json()
    assert data["designation"] == "NACA 2412"
    assert "x" in data

def test_analyze_endpoint():
    payload = {
        "planform": {
            "wing_span_m": 1.0,
            "wing_area_m2": 0.22,
            "taper_ratio": 1.0,
            "washout_deg": 0.0,
            "n_stations": 10
        },
        "alpha_root_deg": 5.0,
        "designation": "NACA 2412"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cl_wing" in data
    assert data["cl_wing"] > 0

def test_design_orchestrator():
    payload = {
        "weight_n": 15.0,
        "wing_span_m": 1.0,
        "cruise_velocity_ms": 12.0,
        "purpose": "trainer",
        "wing_area_m2": 0.22
    }
    response = client.post("/design", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "selected_airfoil" in data
    assert "requirements" in data
    assert "performance_at_cruise" in data
    assert data["performance_at_cruise"]["cl_wing"] > 0

def test_invalid_input_requirements():
    payload = {
        "weight_n": -10.0,  # Invalid
        "wing_span_m": 1.0,
        "cruise_velocity_ms": 12.0
    }
    response = client.post("/requirements", json=payload)
    assert response.status_code == 422  # Pydantic validation error

def test_report_endpoint():
    # Test PDF generation
    response = client.post("/report?designation=NACA 2412")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
