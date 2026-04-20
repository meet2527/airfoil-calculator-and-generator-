"""
SQLite polar storage and retrieval.

Provides a lightweight cache for XFOIL polar data keyed by airfoil
designation, Reynolds number, Mach number, and N_crit.  Downstream
modules (``scoring.py``, ``airfoil_selector.py``) query stored polars
instead of re-running XFOIL.

The database uses two tables:

* **polars** — one row per unique (designation, Re, Mach, N_crit) run
* **polar_points** — one row per converged α station, FK → polars

Typical usage:
    >>> db = PolarDB("polars.db")
    >>> db.store_polar(xfoil_polar)
    >>> cached = db.get_polar("NACA 2412", reynolds=1e6)
    >>> cached.cl   # np.ndarray
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from airfoil_config.xfoil_runner import XfoilPolar


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarQuery:
    """Query parameters for polar lookup.

    Attributes:
        designation: Airfoil name (exact match).
        reynolds: Reynolds number (matched within *re_tolerance*).
        mach: Mach number (matched within *mach_tolerance*).
        n_crit: e^n transition criterion (matched within *ncrit_tolerance*).
    """

    designation: str
    reynolds: float
    mach: float = 0.0
    n_crit: float = 9.0


@dataclass(frozen=True)
class PolarSummary:
    """Lightweight summary of a stored polar (no point data).

    Attributes:
        polar_id: Database primary key.
        designation: Airfoil name.
        reynolds: Reynolds number.
        mach: Mach number.
        n_crit: N_crit value.
        n_points: Number of converged α stations.
        alpha_min: Minimum α [deg].
        alpha_max: Maximum α [deg].
    """

    polar_id: int
    designation: str
    reynolds: float
    mach: float
    n_crit: float
    n_points: int
    alpha_min: float
    alpha_max: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class PolarDB:
    """SQLite-backed polar cache.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"``
            for an in-memory database (useful for tests).
    """

    # Tolerance for floating-point matching
    RE_TOL_FRAC = 0.01    # 1 % relative tolerance on Re
    MACH_TOL_ABS = 0.005  # absolute tolerance on Mach
    NCRIT_TOL_ABS = 0.5   # absolute tolerance on N_crit

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # --- Store -----------------------------------------------------------
    def store_polar(self, polar: XfoilPolar) -> int:
        """Store an XFOIL polar in the database.

        If an entry with the same (designation, Re, Mach, N_crit) already
        exists (within tolerance), it is replaced.

        Args:
            polar: XFOIL polar result.

        Returns:
            Database row ID of the stored polar.

        Raises:
            ValueError: If the polar has no data points.
        """
        if polar.converged_count == 0:
            raise ValueError("Cannot store a polar with zero converged points")

        # Remove existing match if any
        existing = self._find_polar_id(
            polar.designation, polar.reynolds, polar.mach, polar.n_crit,
        )
        if existing is not None:
            self._delete_polar(existing)

        with self._transaction():
            cur = self._conn.execute(
                """INSERT INTO polars
                   (designation, reynolds, mach, n_crit,
                    converged_count, total_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (polar.designation, polar.reynolds, polar.mach,
                 polar.n_crit, polar.converged_count, polar.total_count),
            )
            polar_id = cur.lastrowid
            assert polar_id is not None

            rows = [
                (polar_id, float(a), float(cl), float(cd),
                 float(cdp), float(cm), float(tx), float(bx))
                for a, cl, cd, cdp, cm, tx, bx in zip(
                    polar.alpha_deg, polar.cl, polar.cd,
                    polar.cdp, polar.cm, polar.top_xtr, polar.bot_xtr,
                )
            ]
            self._conn.executemany(
                """INSERT INTO polar_points
                   (polar_id, alpha_deg, cl, cd, cdp, cm,
                    top_xtr, bot_xtr)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        return polar_id

    # --- Retrieve --------------------------------------------------------
    def get_polar(
        self,
        designation: str,
        reynolds: float,
        mach: float = 0.0,
        n_crit: float = 9.0,
    ) -> Optional[XfoilPolar]:
        """Retrieve a polar from the cache.

        Matches within the class tolerance attributes (``RE_TOL_FRAC``,
        ``MACH_TOL_ABS``, ``NCRIT_TOL_ABS``).

        Args:
            designation: Airfoil name.
            reynolds: Reynolds number.
            mach: Mach number.
            n_crit: N_crit value.

        Returns:
            :class:`XfoilPolar` or ``None`` if no match.
        """
        polar_id = self._find_polar_id(designation, reynolds, mach, n_crit)
        if polar_id is None:
            return None
        return self._load_polar(polar_id)

    def get_polar_by_query(self, query: PolarQuery) -> Optional[XfoilPolar]:
        """Retrieve a polar using a :class:`PolarQuery`.

        Args:
            query: Query parameters.

        Returns:
            :class:`XfoilPolar` or ``None``.
        """
        return self.get_polar(
            query.designation, query.reynolds, query.mach, query.n_crit,
        )

    def has_polar(
        self,
        designation: str,
        reynolds: float,
        mach: float = 0.0,
        n_crit: float = 9.0,
    ) -> bool:
        """Check if a polar exists in the cache.

        Args:
            designation: Airfoil name.
            reynolds: Reynolds number.
            mach: Mach number.
            n_crit: N_crit value.

        Returns:
            True if a matching polar is stored.
        """
        return self._find_polar_id(designation, reynolds, mach, n_crit) is not None

    # --- Listing / admin -------------------------------------------------
    def list_polars(
        self, designation: Optional[str] = None,
    ) -> list[PolarSummary]:
        """List stored polars, optionally filtered by designation.

        Args:
            designation: If given, only return polars for this airfoil.

        Returns:
            List of :class:`PolarSummary` objects.
        """
        if designation is not None:
            rows = self._conn.execute(
                """SELECT p.id, p.designation, p.reynolds, p.mach, p.n_crit,
                          p.converged_count,
                          MIN(pp.alpha_deg), MAX(pp.alpha_deg)
                   FROM polars p
                   LEFT JOIN polar_points pp ON pp.polar_id = p.id
                   WHERE p.designation = ?
                   GROUP BY p.id
                   ORDER BY p.reynolds""",
                (designation,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT p.id, p.designation, p.reynolds, p.mach, p.n_crit,
                          p.converged_count,
                          MIN(pp.alpha_deg), MAX(pp.alpha_deg)
                   FROM polars p
                   LEFT JOIN polar_points pp ON pp.polar_id = p.id
                   GROUP BY p.id
                   ORDER BY p.designation, p.reynolds""",
            ).fetchall()

        return [
            PolarSummary(
                polar_id=r[0], designation=r[1], reynolds=r[2],
                mach=r[3], n_crit=r[4], n_points=r[5],
                alpha_min=r[6] if r[6] is not None else 0.0,
                alpha_max=r[7] if r[7] is not None else 0.0,
            )
            for r in rows
        ]

    def delete_polar(self, designation: str, reynolds: float,
                     mach: float = 0.0, n_crit: float = 9.0) -> bool:
        """Delete a polar from the cache.

        Args:
            designation: Airfoil name.
            reynolds: Reynolds number.
            mach: Mach number.
            n_crit: N_crit value.

        Returns:
            True if a polar was deleted, False if not found.
        """
        pid = self._find_polar_id(designation, reynolds, mach, n_crit)
        if pid is None:
            return False
        self._delete_polar(pid)
        return True

    def count(self) -> int:
        """Return the total number of stored polars.

        Returns:
            Integer count.
        """
        row = self._conn.execute("SELECT COUNT(*) FROM polars").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> PolarDB:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- Private ---------------------------------------------------------
    def _create_tables(self) -> None:
        """Create tables if they do not exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS polars (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                designation     TEXT    NOT NULL,
                reynolds        REAL    NOT NULL,
                mach            REAL    NOT NULL DEFAULT 0.0,
                n_crit          REAL    NOT NULL DEFAULT 9.0,
                converged_count INTEGER NOT NULL,
                total_count     INTEGER NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS polar_points (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                polar_id  INTEGER NOT NULL REFERENCES polars(id) ON DELETE CASCADE,
                alpha_deg REAL NOT NULL,
                cl        REAL NOT NULL,
                cd        REAL NOT NULL,
                cdp       REAL NOT NULL,
                cm        REAL NOT NULL,
                top_xtr   REAL NOT NULL,
                bot_xtr   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_polars_lookup
                ON polars (designation, reynolds, mach, n_crit);

            CREATE INDEX IF NOT EXISTS idx_polar_points_fk
                ON polar_points (polar_id);
        """)

    def _find_polar_id(
        self, designation: str, reynolds: float, mach: float, n_crit: float,
    ) -> Optional[int]:
        """Find a polar matching within tolerance.

        Args:
            designation: Airfoil name.
            reynolds: Re.
            mach: Mach.
            n_crit: N_crit.

        Returns:
            Polar row ID or None.
        """
        re_lo = reynolds * (1.0 - self.RE_TOL_FRAC)
        re_hi = reynolds * (1.0 + self.RE_TOL_FRAC)
        m_lo = mach - self.MACH_TOL_ABS
        m_hi = mach + self.MACH_TOL_ABS
        nc_lo = n_crit - self.NCRIT_TOL_ABS
        nc_hi = n_crit + self.NCRIT_TOL_ABS

        row = self._conn.execute(
            """SELECT id FROM polars
               WHERE designation = ?
                 AND reynolds BETWEEN ? AND ?
                 AND mach BETWEEN ? AND ?
                 AND n_crit BETWEEN ? AND ?
               ORDER BY ABS(reynolds - ?) ASC
               LIMIT 1""",
            (designation, re_lo, re_hi, m_lo, m_hi, nc_lo, nc_hi, reynolds),
        ).fetchone()

        return row[0] if row else None

    def _load_polar(self, polar_id: int) -> XfoilPolar:
        """Load full polar data from the database.

        Args:
            polar_id: Row ID in the polars table.

        Returns:
            :class:`XfoilPolar`.
        """
        meta = self._conn.execute(
            """SELECT designation, reynolds, mach, n_crit,
                      converged_count, total_count
               FROM polars WHERE id = ?""",
            (polar_id,),
        ).fetchone()
        assert meta is not None

        points = self._conn.execute(
            """SELECT alpha_deg, cl, cd, cdp, cm, top_xtr, bot_xtr
               FROM polar_points
               WHERE polar_id = ?
               ORDER BY alpha_deg""",
            (polar_id,),
        ).fetchall()

        data = np.array(points)

        return XfoilPolar(
            alpha_deg=data[:, 0],
            cl=data[:, 1],
            cd=data[:, 2],
            cdp=data[:, 3],
            cm=data[:, 4],
            top_xtr=data[:, 5],
            bot_xtr=data[:, 6],
            reynolds=meta[1],
            mach=meta[2],
            n_crit=meta[3],
            designation=meta[0],
            converged_count=meta[4],
            total_count=meta[5],
        )

    def _delete_polar(self, polar_id: int) -> None:
        """Delete a polar and its points.

        Args:
            polar_id: Row ID.
        """
        with self._transaction():
            self._conn.execute(
                "DELETE FROM polar_points WHERE polar_id = ?", (polar_id,),
            )
            self._conn.execute(
                "DELETE FROM polars WHERE id = ?", (polar_id,),
            )

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Context manager for an atomic transaction."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
