from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str | None:
    return os.environ.get("DATABASE_URL") or None


def get_engine(url: str | None = None) -> Engine:
    db_url = url or get_database_url()
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True)


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), autoflush=False, autocommit=False)
