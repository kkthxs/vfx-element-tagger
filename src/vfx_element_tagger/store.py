from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Element, ProcessingEvent


JSON_SCHEMA_VERSION = "0.2.0-json"
SQLITE_SCHEMA_VERSION = "0.3.0-sqlite"
SCHEMA_VERSION = JSON_SCHEMA_VERSION
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_RATER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")


class LibraryStore:
    """Catalog persistence with JSON compatibility and transactional SQLite support.

    JSON remains useful for portable exports and backups. SQLite is the operational
    store for a central server: callers can update one element or rating without
    rewriting the whole library, and concurrent requests are serialized by SQLite.
    Network clients should use the HTTP API rather than mounting this file directly.
    """

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.is_sqlite = self.path.suffix.lower() in SQLITE_SUFFIXES
        self.log_path = (
            self.path if self.is_sqlite else self.path.with_suffix(".processing-log.jsonl")
        )
        if self.is_sqlite:
            self._initialize_sqlite()

    def load(self) -> list[Element]:
        if not self.path.exists():
            return []
        if self.is_sqlite:
            with closing(self._connect()) as connection, connection:
                rows = connection.execute(
                    "SELECT payload_json FROM elements ORDER BY sort_order, element_id"
                ).fetchall()
            return [Element.from_dict(json.loads(row["payload_json"])) for row in rows]
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [Element.from_dict(item) for item in data.get("elements", [])]

    @classmethod
    def load_readonly(cls, path: Path) -> list[Element]:
        """Read existing data without schema initialization, journal changes or creation."""
        path = path.expanduser().resolve()
        if path.suffix.lower() in SQLITE_SUFFIXES:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                rows = connection.execute("SELECT payload_json FROM elements ORDER BY sort_order, element_id").fetchall()
            return [Element.from_dict(json.loads(row[0])) for row in rows]
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Element.from_dict(item) for item in data.get("elements", [])]

    def load_one(self, element_id: str) -> Element | None:
        if self.is_sqlite:
            with closing(self._connect()) as connection, connection:
                row = connection.execute(
                    "SELECT payload_json FROM elements WHERE element_id = ?",
                    (element_id,),
                ).fetchone()
            return Element.from_dict(json.loads(row["payload_json"])) if row else None
        return next((item for item in self.load() if item.element_id == element_id), None)

    def save(self, elements: list[Element]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.is_sqlite:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS desired_element_ids "
                    "(element_id TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                connection.execute("DELETE FROM desired_element_ids")
                connection.executemany(
                    "INSERT INTO desired_element_ids (element_id) VALUES (?)",
                    ((element.element_id,) for element in elements),
                )
                for sort_order, element in enumerate(elements):
                    self._upsert_element(connection, element, sort_order)
                connection.execute(
                    "DELETE FROM elements WHERE element_id NOT IN "
                    "(SELECT element_id FROM desired_element_ids)"
                )
                try:
                    connection.execute(
                        "DELETE FROM element_search WHERE element_id NOT IN "
                        "(SELECT element_id FROM desired_element_ids)"
                    )
                except sqlite3.OperationalError:
                    pass
                self._bump_revision(connection)
            return
        payload: dict[str, Any] = {
            "schema_version": JSON_SCHEMA_VERSION,
            "elements": [element.to_dict() for element in elements],
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save_element(self, element: Element) -> None:
        """Persist one changed element without an O(library-size) rewrite."""

        if not self.is_sqlite:
            elements = self.load()
            by_id = {item.element_id: index for index, item in enumerate(elements)}
            if element.element_id in by_id:
                elements[by_id[element.element_id]] = element
            else:
                elements.append(element)
            self.save(elements)
            return
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT sort_order FROM elements WHERE element_id = ?",
                (element.element_id,),
            ).fetchone()
            if row:
                sort_order = int(row["sort_order"])
            else:
                sort_order = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sort_order), -1) + 1 AS value FROM elements"
                    ).fetchone()["value"]
                )
            self._upsert_element(connection, element, sort_order)
            self._bump_revision(connection)

    def append_events(self, events: list[ProcessingEvent]) -> None:
        if not events:
            return
        if self.is_sqlite:
            now = _now()
            with closing(self._connect()) as connection, connection:
                connection.executemany(
                    """
                    INSERT INTO processing_events (
                        recorded_at, element_id, stage, status, error_kind,
                        error_message, model_version, wall_clock_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            now,
                            event.element_id,
                            event.stage,
                            event.status,
                            event.error_kind,
                            event.error_message,
                            event.model_version,
                            event.wall_clock_ms,
                        )
                        for event in events
                    ],
                )
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True))
                handle.write("\n")

    def backup(self, destination: Path) -> Path:
        """Write a consistent catalog backup, including while SQLite is in WAL mode."""

        destination = destination.expanduser().resolve()
        if destination == self.path:
            raise ValueError("Backup destination must differ from the live catalog")
        if destination.exists():
            raise FileExistsError(f"Backup already exists: {destination}")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            if self.is_sqlite:
                with closing(self._connect()) as source, closing(sqlite3.connect(temporary)) as target:
                    source.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("Backup integrity check failed")
            else:
                shutil.copy2(self.path, temporary)
            # Link publishes atomically and refuses to overwrite a racing backup.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def revision(self) -> str:
        if self.is_sqlite:
            with closing(self._connect()) as connection, connection:
                row = connection.execute(
                    "SELECT value FROM catalog_meta WHERE key = 'revision'"
                ).fetchone()
            return str(row["value"] if row else "0")
        try:
            return str(self.path.stat().st_mtime_ns)
        except FileNotFoundError:
            return "0"

    def rate_element(
        self,
        element_id: str,
        rater_id: str,
        rating: int,
        rater_name: str = "",
    ) -> dict[str, Any]:
        rater_id = str(rater_id or "").strip()
        if not _RATER_ID.fullmatch(rater_id):
            raise ValueError("rater_id must be 1–128 safe identifier characters")
        try:
            rating = int(rating)
        except (TypeError, ValueError) as exc:
            raise ValueError("rating must be an integer from 1 to 5") from exc
        if rating < 1 or rating > 5:
            raise ValueError("rating must be an integer from 1 to 5")
        rater_name = " ".join(str(rater_name or "").split())[:100]
        database_path = self._ratings_database_path()
        with closing(self._ratings_connect(database_path)) as connection, connection:
            if self.is_sqlite:
                exists = connection.execute(
                    "SELECT 1 FROM elements WHERE element_id = ?", (element_id,)
                ).fetchone()
            else:
                exists = self.load_one(element_id) is not None
            if not exists:
                raise KeyError(element_id)
            now = _now()
            connection.execute(
                """
                INSERT INTO ratings (
                    element_id, rater_id, rater_name, rating, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(element_id, rater_id) DO UPDATE SET
                    rater_name = excluded.rater_name,
                    rating = excluded.rating,
                    updated_at = excluded.updated_at
                """,
                (element_id, rater_id, rater_name, rating, now, now),
            )
        return self.rating_summary(element_id, rater_id)

    def rating_summary(self, element_id: str, rater_id: str | None = None) -> dict[str, Any]:
        return self.rating_summaries(rater_id=rater_id).get(
            element_id,
            _empty_rating_summary(rater_id),
        )

    def rating_summaries(self, rater_id: str | None = None) -> dict[str, dict[str, Any]]:
        database_path = self._ratings_database_path()
        if not database_path.exists():
            return {}
        with closing(self._ratings_connect(database_path)) as connection, connection:
            rows = connection.execute(
                """
                SELECT element_id, COUNT(*) AS rating_count, AVG(rating) AS rating_average,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS star_1,
                    SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) AS star_2,
                    SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) AS star_3,
                    SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) AS star_4,
                    SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS star_5
                FROM ratings GROUP BY element_id
                """
            ).fetchall()
            own: dict[str, int] = {}
            if rater_id:
                own = {
                    str(row["element_id"]): int(row["rating"])
                    for row in connection.execute(
                        "SELECT element_id, rating FROM ratings WHERE rater_id = ?",
                        (rater_id,),
                    ).fetchall()
                }
        return {
            str(row["element_id"]): {
                "rating_average": round(float(row["rating_average"] or 0.0), 2),
                "rating_count": int(row["rating_count"] or 0),
                "rating_quality": _rating_quality(
                    float(row["rating_average"] or 0.0),
                    int(row["rating_count"] or 0),
                ),
                "rating_distribution": {
                    str(star): int(row[f"star_{star}"] or 0) for star in range(1, 6)
                },
                "user_rating": own.get(str(row["element_id"])),
            }
            for row in rows
        }

    def _initialize_sqlite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect(initialize=False)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY,
                    sort_order INTEGER NOT NULL,
                    primary_family TEXT NOT NULL DEFAULT 'unknown',
                    effect_type TEXT NOT NULL DEFAULT 'unclassified',
                    analysis_status TEXT NOT NULL DEFAULT 'unanalysed',
                    duration_seconds REAL,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    payload_json TEXT NOT NULL,
                    search_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_elements_family
                    ON elements(primary_family, effect_type);
                CREATE INDEX IF NOT EXISTS idx_elements_status
                    ON elements(analysis_status);
                CREATE INDEX IF NOT EXISTS idx_elements_sort
                    ON elements(sort_order);
                CREATE TABLE IF NOT EXISTS ratings (
                    element_id TEXT NOT NULL REFERENCES elements(element_id) ON DELETE CASCADE,
                    rater_id TEXT NOT NULL,
                    rater_name TEXT NOT NULL DEFAULT '',
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(element_id, rater_id)
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS idx_ratings_rater ON ratings(rater_id);
                CREATE INDEX IF NOT EXISTS idx_ratings_element_rating
                    ON ratings(element_id, rating);
                CREATE TABLE IF NOT EXISTS processing_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    element_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_kind TEXT,
                    error_message TEXT,
                    model_version TEXT,
                    wall_clock_ms INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_processing_events_element
                    ON processing_events(element_id, recorded_at);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO catalog_meta (key, value) VALUES ('schema_version', ?)",
                (SQLITE_SCHEMA_VERSION,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO catalog_meta (key, value) VALUES ('revision', '0')"
            )
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS element_search USING fts5("
                    "element_id UNINDEXED, search_text)"
                )
            except sqlite3.OperationalError:
                # Python builds without FTS5 still retain the indexed structured columns.
                pass

    def _connect(self, initialize: bool = True) -> sqlite3.Connection:
        if initialize and not self.path.exists():
            self._initialize_sqlite()
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
        except BaseException:
            connection.close()
            raise
        return connection

    def _ratings_database_path(self) -> Path:
        if self.is_sqlite:
            return self.path
        return self.path.with_suffix(".ratings.sqlite3")

    def _ratings_connect(self, path: Path) -> sqlite3.Connection:
        if self.is_sqlite:
            return self._connect()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    element_id TEXT NOT NULL,
                    rater_id TEXT NOT NULL,
                    rater_name TEXT NOT NULL DEFAULT '',
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(element_id, rater_id)
                ) WITHOUT ROWID
                """
            )
        except BaseException:
            connection.close()
            raise
        return connection

    def _upsert_element(
        self,
        connection: sqlite3.Connection,
        element: Element,
        sort_order: int,
    ) -> None:
        payload = element.to_dict()
        semantic = element.semantic_analysis or {}
        effect_type = str(semantic.get("effect_type") or "unclassified")
        searchable = _searchable_text(payload)
        connection.execute(
            """
            INSERT INTO elements (
                element_id, sort_order, primary_family, effect_type, analysis_status,
                duration_seconds, width, height, fps, payload_json, search_text, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(element_id) DO UPDATE SET
                sort_order = excluded.sort_order,
                primary_family = excluded.primary_family,
                effect_type = excluded.effect_type,
                analysis_status = excluded.analysis_status,
                duration_seconds = excluded.duration_seconds,
                width = excluded.width,
                height = excluded.height,
                fps = excluded.fps,
                payload_json = excluded.payload_json,
                search_text = excluded.search_text,
                updated_at = excluded.updated_at
            """,
            (
                element.element_id,
                sort_order,
                element.primary_family or element.category or "unknown",
                effect_type,
                element.analysis_status,
                element.duration_seconds,
                element.width,
                element.height,
                element.fps,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                searchable,
                _now(),
            ),
        )
        try:
            connection.execute("DELETE FROM element_search WHERE element_id = ?", (element.element_id,))
            connection.execute(
                "INSERT INTO element_search (element_id, search_text) VALUES (?, ?)",
                (element.element_id, searchable),
            )
        except sqlite3.OperationalError:
            pass

    @staticmethod
    def _bump_revision(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO catalog_meta (key, value) VALUES ('revision', '1')
            ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
            """
        )


def _searchable_text(value: Any) -> str:
    terms: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float)):
            terms.append(str(item).replace("_", " "))

    visit(value)
    return " ".join(terms)


def _empty_rating_summary(rater_id: str | None = None) -> dict[str, Any]:
    return {
        "rating_average": 0.0,
        "rating_count": 0,
        "rating_quality": 0.0,
        "rating_distribution": {str(star): 0 for star in range(1, 6)},
        "user_rating": None if rater_id else None,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rating_quality(average: float, count: int) -> float:
    """Bayesian 1–5 score so one five-star vote cannot dominate the library."""

    if count <= 0:
        return 0.0
    prior_mean = 3.0
    prior_weight = 5
    return round((average * count + prior_mean * prior_weight) / (count + prior_weight), 3)
