"""Async database access (SQLAlchemy Core on psycopg 3). One engine per process, created at start-up."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("tutor_ai.db")


def create_engine(url: str, *, pool_size: int = 10, max_overflow: int = 15) -> AsyncEngine:
    return create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Neon suspends idle computes; a dead pooled connection is replaced, not used
        pool_recycle=300,
    )


async def ping(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:  # any failure means "database not reachable"; reported by /health
        logger.warning("database ping failed", exc_info=True)
        return False
