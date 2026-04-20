"""
Tests for airfoil_config.polar_db — SQLite polar storage.

All tests use in-memory databases (``":memory:"``) so no files are
created on disk.

Covers:
    - Store and retrieve a polar
    - Tolerance-based matching (Re, Mach, N_crit)
    - Cache miss (returns None)
    - Upsert (replace existing)
    - has_polar check
    - list_polars / count
    - delete_polar
    - Multi-polar storage
    - Context manager (with statement)
    - Edge cases (empty polar, zero points)
"""

from __future__ import annotations

import numpy as np
import pytest

from airfoil_config.polar_db import PolarDB, PolarQuery, PolarSummary
from airfoil_config.xfoil_runner import XfoilPolar


# ===================================================================
# Helpers
# ===================================================================
def _make_polar(
    designation: str = "NACA 0012",
    reynolds: float = 1e6,
    mach: float = 0.0,
    n_crit: float = 9.0,
    n_points: int = 5,
) -> XfoilPolar:
    """Create a synthetic XfoilPolar for testing."""
    alpha = np.linspace(-2.0, 8.0, n_points)
    cl = 0.11 * alpha  # rough linear
    cd = 0.005 + 0.001 * alpha ** 2
    cdp = cd * 0.4
    cm = -0.02 * np.ones(n_points)
    top_xtr = 0.5 * np.ones(n_points)
    bot_xtr = 0.6 * np.ones(n_points)

    return XfoilPolar(
        alpha_deg=alpha,
        cl=cl,
        cd=cd,
        cdp=cdp,
        cm=cm,
        top_xtr=top_xtr,
        bot_xtr=bot_xtr,
        reynolds=reynolds,
        mach=mach,
        n_crit=n_crit,
        designation=designation,
        converged_count=n_points,
        total_count=n_points + 2,
    )


# ===================================================================
# Store and retrieve
# ===================================================================
class TestStoreRetrieve:
    """Basic store → get round-trip."""

    def test_store_returns_id(self) -> None:
        with PolarDB() as db:
            pid = db.store_polar(_make_polar())
            assert isinstance(pid, int)
            assert pid >= 1

    def test_retrieve_matches(self) -> None:
        with PolarDB() as db:
            polar = _make_polar()
            db.store_polar(polar)
            cached = db.get_polar("NACA 0012", 1e6)
            assert cached is not None
            assert cached.designation == "NACA 0012"
            assert cached.reynolds == 1e6

    def test_cl_round_trip(self) -> None:
        with PolarDB() as db:
            polar = _make_polar()
            db.store_polar(polar)
            cached = db.get_polar("NACA 0012", 1e6)
            assert cached is not None
            np.testing.assert_allclose(cached.cl, polar.cl, atol=1e-10)

    def test_alpha_sorted(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            cached = db.get_polar("NACA 0012", 1e6)
            assert cached is not None
            assert np.all(np.diff(cached.alpha_deg) >= 0)

    def test_metadata_preserved(self) -> None:
        with PolarDB() as db:
            polar = _make_polar(n_crit=11.0, mach=0.3)
            db.store_polar(polar)
            cached = db.get_polar("NACA 0012", 1e6, mach=0.3, n_crit=11.0)
            assert cached is not None
            assert cached.mach == 0.3
            assert cached.n_crit == 11.0
            assert cached.converged_count == polar.converged_count
            assert cached.total_count == polar.total_count


# ===================================================================
# Tolerance-based matching
# ===================================================================
class TestToleranceMatching:
    """Re/Mach/N_crit tolerance matching."""

    def test_re_within_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(reynolds=1e6))
            # 0.5 % off → should match within 1 % tolerance
            cached = db.get_polar("NACA 0012", 1.005e6)
            assert cached is not None

    def test_re_outside_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(reynolds=1e6))
            # 5 % off → should NOT match
            cached = db.get_polar("NACA 0012", 1.05e6)
            assert cached is None

    def test_mach_within_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(mach=0.3))
            cached = db.get_polar("NACA 0012", 1e6, mach=0.303)
            assert cached is not None

    def test_mach_outside_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(mach=0.3))
            cached = db.get_polar("NACA 0012", 1e6, mach=0.35)
            assert cached is None

    def test_ncrit_within_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(n_crit=9.0))
            cached = db.get_polar("NACA 0012", 1e6, n_crit=9.3)
            assert cached is not None

    def test_ncrit_outside_tolerance(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(n_crit=9.0))
            cached = db.get_polar("NACA 0012", 1e6, n_crit=11.0)
            assert cached is None


# ===================================================================
# Cache miss
# ===================================================================
class TestCacheMiss:
    """Queries that should return None."""

    def test_empty_db(self) -> None:
        with PolarDB() as db:
            assert db.get_polar("NACA 0012", 1e6) is None

    def test_wrong_designation(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(designation="NACA 0012"))
            assert db.get_polar("NACA 2412", 1e6) is None

    def test_wrong_re(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(reynolds=1e6))
            assert db.get_polar("NACA 0012", 5e6) is None


# ===================================================================
# Upsert (replace existing)
# ===================================================================
class TestUpsert:
    """Storing the same (designation, Re, Mach, N_crit) replaces old data."""

    def test_replaces_old_polar(self) -> None:
        with PolarDB() as db:
            p1 = _make_polar(n_points=5)
            p2 = _make_polar(n_points=10)
            db.store_polar(p1)
            db.store_polar(p2)

            cached = db.get_polar("NACA 0012", 1e6)
            assert cached is not None
            assert cached.converged_count == 10

    def test_count_stays_at_one(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            db.store_polar(_make_polar())
            assert db.count() == 1


# ===================================================================
# has_polar
# ===================================================================
class TestHasPolar:
    """Cache-hit check without loading data."""

    def test_true_when_exists(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            assert db.has_polar("NACA 0012", 1e6) is True

    def test_false_when_missing(self) -> None:
        with PolarDB() as db:
            assert db.has_polar("NACA 0012", 1e6) is False


# ===================================================================
# PolarQuery
# ===================================================================
class TestPolarQuery:
    """Query-object API."""

    def test_get_by_query(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            q = PolarQuery("NACA 0012", 1e6)
            cached = db.get_polar_by_query(q)
            assert cached is not None


# ===================================================================
# Listing and counting
# ===================================================================
class TestListAndCount:
    """list_polars / count."""

    def test_count_empty(self) -> None:
        with PolarDB() as db:
            assert db.count() == 0

    def test_count_after_store(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            assert db.count() == 1

    def test_list_returns_summaries(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            summaries = db.list_polars()
            assert len(summaries) == 1
            assert isinstance(summaries[0], PolarSummary)

    def test_list_filter_by_designation(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(designation="NACA 0012"))
            db.store_polar(_make_polar(designation="NACA 2412"))
            all_polars = db.list_polars()
            filtered = db.list_polars("NACA 0012")
            assert len(all_polars) == 2
            assert len(filtered) == 1
            assert filtered[0].designation == "NACA 0012"

    def test_summary_fields(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(n_points=7))
            s = db.list_polars()[0]
            assert s.n_points == 7
            assert s.reynolds == 1e6
            assert s.alpha_min < s.alpha_max


# ===================================================================
# Delete
# ===================================================================
class TestDelete:
    """Polar deletion."""

    def test_delete_existing(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            assert db.delete_polar("NACA 0012", 1e6) is True
            assert db.count() == 0

    def test_delete_nonexistent(self) -> None:
        with PolarDB() as db:
            assert db.delete_polar("NACA 0012", 1e6) is False

    def test_delete_removes_points(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            db.delete_polar("NACA 0012", 1e6)
            # Verify no orphan points
            row = db._conn.execute(
                "SELECT COUNT(*) FROM polar_points"
            ).fetchone()
            assert row[0] == 0


# ===================================================================
# Multi-polar
# ===================================================================
class TestMultiPolar:
    """Multiple different polars."""

    def test_different_designations(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(designation="NACA 0012"))
            db.store_polar(_make_polar(designation="NACA 2412"))
            assert db.count() == 2

    def test_same_designation_different_re(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(reynolds=1e6))
            db.store_polar(_make_polar(reynolds=3e6))
            assert db.count() == 2
            assert db.has_polar("NACA 0012", 1e6)
            assert db.has_polar("NACA 0012", 3e6)

    def test_correct_polar_returned(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar(reynolds=1e6, n_points=5))
            db.store_polar(_make_polar(reynolds=3e6, n_points=10))
            c1 = db.get_polar("NACA 0012", 1e6)
            c3 = db.get_polar("NACA 0012", 3e6)
            assert c1 is not None and c1.converged_count == 5
            assert c3 is not None and c3.converged_count == 10


# ===================================================================
# Edge cases
# ===================================================================
class TestEdgeCases:
    """Boundary conditions."""

    def test_zero_points_raises(self) -> None:
        polar = XfoilPolar(
            alpha_deg=np.array([]),
            cl=np.array([]),
            cd=np.array([]),
            cdp=np.array([]),
            cm=np.array([]),
            top_xtr=np.array([]),
            bot_xtr=np.array([]),
            reynolds=1e6, mach=0.0, n_crit=9.0,
            designation="empty", converged_count=0, total_count=10,
        )
        with PolarDB() as db:
            with pytest.raises(ValueError, match="zero"):
                db.store_polar(polar)

    def test_single_point_polar(self) -> None:
        polar = _make_polar(n_points=1)
        with PolarDB() as db:
            db.store_polar(polar)
            cached = db.get_polar("NACA 0012", 1e6)
            assert cached is not None
            assert cached.converged_count == 1


# ===================================================================
# Context manager
# ===================================================================
class TestContextManager:
    """Verify with-statement works."""

    def test_enter_exit(self) -> None:
        with PolarDB() as db:
            db.store_polar(_make_polar())
            assert db.count() == 1
        # db.close() was called — no error
