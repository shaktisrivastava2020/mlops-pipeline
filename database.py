"""
Cloud SQL connection management.

Pattern: a single pooled connector + sessionmaker, instantiated lazily.
Keeps connection cost off the import path, lets tests substitute their
own connection without monkey-patching globals.

Auth: IAM-based, via Cloud SQL Python Connector. The pg8000 driver is
pure-Python (no native dependencies) and plays nicely with the connector.

Mirrors the pattern used by Day 1 (~/quickshop-ai) and Day 2
(~/churn-predictor) so anyone reading any of the three projects sees the
same connection abstraction.
"""
from __future__ import annotations
from typing import Iterator

import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings


# Module-level singletons, lazy-initialized.
_connector: Connector | None = None
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_connector() -> Connector:
    """Lazy connector — created on first use, reused thereafter."""
    global _connector
    if _connector is None:
        _connector = Connector()
    return _connector


def _make_pg_connection():
    """
    Factory the SQLAlchemy engine calls every time it needs a fresh connection.
    Wraps the Cloud SQL connector with our project's instance + credentials.
    """
    return _get_connector().connect(
        settings.CLOUD_SQL_INSTANCE,
        "pg8000",
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=settings.DB_NAME,
    )


def get_engine() -> Engine:
    """Lazy engine. Connection pool size is intentionally small for Cloud Run."""
    global _engine
    if _engine is None:
        _engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=_make_pg_connection,
            pool_size=2,
            max_overflow=2,
            pool_pre_ping=True,  # avoids stale-connection errors after Cloud SQL idle restarts
        )
    return _engine


def get_sessionmaker() -> sessionmaker:
    """Lazy sessionmaker. Used by the rest of the codebase via get_session()."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_session() -> Iterator[Session]:
    """
    Generator-style session for FastAPI dependency injection or `with` blocks.
    Always closes, regardless of exceptions.
    """
    sess = get_sessionmaker()()
    try:
        yield sess
    finally:
        sess.close()


def ping() -> str:
    """Cheap health probe. Returns 'ok' or raises."""
    with get_engine().connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT 1")).scalar()
    if result != 1:
        raise RuntimeError(f"Database ping returned unexpected value: {result}")
    return "ok"
