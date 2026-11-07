"""The lesson index: replace, delete and list the indexed lessons (design §5).

Guarantees:
- **Atomic replacement.** A lesson's old chunks are deleted and the new ones inserted in one transaction, so
  retrieval never sees a half-indexed lesson or two versions at once.
- **Never older over newer.** Every chunk carries Django's version number. A request for an older version than the
  one indexed is refused (409 `stale_version`), so a delayed retry can't bring back replaced content. Concurrent
  requests for the same lesson are serialised with a transaction-scoped advisory lock and the check is repeated
  under the lock.
- **Idempotent.** Each call carries an Idempotency-Key. Replaying a finished request returns its original result
  without doing the work again; reusing a key for a different request is refused (409 `idempotency_conflict`).
  Indexing the version that is already indexed with the current embedding model is a cheap no-op ("unchanged").
- **No work inside a transaction.** Embeddings (the slow part) are computed before the transaction starts.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Row, delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .chunking import chunk_sections
from .errors import AppError
from .providers.base import ModelProvider, ProviderError
from .schema import EMBEDDING_DIMENSIONS, index_event, lesson_chunk

logger = logging.getLogger("tutor_ai.indexing")

MAX_CHUNKS_PER_LESSON = 400  # ~120k words; a lesson this long is a content mistake, not something to embed

Action = Literal["index", "delete"]


@dataclass(frozen=True)
class LessonDocument:
    lesson_id: uuid.UUID
    content_version_id: uuid.UUID
    version_no: int
    course_id: uuid.UUID
    subject: str
    class_number: int | None
    title: str
    sections: list[dict[str, Any]]


@dataclass(frozen=True)
class IndexResult:
    lesson_id: uuid.UUID
    content_version_id: uuid.UUID
    version_no: int
    chunks: int
    embedding_model: str
    status: Literal["indexed", "unchanged"]


@dataclass(frozen=True)
class IndexedLesson:
    lesson_id: uuid.UUID
    content_version_id: uuid.UUID
    version_no: int
    embedding_model: str
    chunks: int
    indexed_at: datetime


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------


async def _lock_lesson(conn: AsyncConnection, lesson_id: uuid.UUID) -> None:
    """Serialise writers of one lesson until the transaction ends (readers are never blocked)."""
    await conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": f"ai.lesson_chunk:{lesson_id}"}
    )


def _indexed_query() -> Any:
    return select(
        lesson_chunk.c.lesson_id,
        lesson_chunk.c.content_version_id,
        lesson_chunk.c.version_no,
        lesson_chunk.c.embedding_model,
        func.count().label("chunks"),
        func.max(lesson_chunk.c.created_at).label("indexed_at"),
    ).group_by(
        lesson_chunk.c.lesson_id,
        lesson_chunk.c.content_version_id,
        lesson_chunk.c.version_no,
        lesson_chunk.c.embedding_model,
    )


def _to_indexed(row: Row[Any]) -> IndexedLesson:
    return IndexedLesson(
        lesson_id=row.lesson_id,
        content_version_id=row.content_version_id,
        version_no=row.version_no,
        embedding_model=row.embedding_model,
        chunks=row.chunks,
        indexed_at=row.indexed_at,
    )


async def _current(conn: AsyncConnection, lesson_id: uuid.UUID) -> IndexedLesson | None:
    query = _indexed_query().where(lesson_chunk.c.lesson_id == lesson_id).order_by(lesson_chunk.c.version_no.desc())
    row = (await conn.execute(query.limit(1))).first()
    return _to_indexed(row) if row else None


async def _event(conn: AsyncConnection, key: str) -> Row[Any] | None:
    return (await conn.execute(select(index_event).where(index_event.c.idempotency_key == key))).first()


def _check_same_request(
    event: Row[Any], lesson_id: uuid.UUID, action: Action, content_version_id: uuid.UUID | None
) -> None:
    if (event.lesson_id, event.action, event.content_version_id) != (lesson_id, action, content_version_id):
        raise AppError(409, "idempotency_conflict", "This Idempotency-Key was already used for a different request.")


async def _record(
    conn: AsyncConnection,
    *,
    key: str,
    lesson_id: uuid.UUID,
    action: Action,
    status: Literal["done", "failed"],
    content_version_id: uuid.UUID | None = None,
    version_no: int | None = None,
    chunk_count: int | None = None,
    embedding_model: str | None = None,
    error: str | None = None,
) -> None:
    values = {
        "lesson_id": lesson_id,
        "content_version_id": content_version_id,
        "action": action,
        "idempotency_key": key,
        "status": status,
        "error": error[:2000] if error else None,
        "version_no": version_no,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
    }
    stmt = pg_insert(index_event).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_index_event_idempotency_key",
        set_={
            "status": status,
            "error": values["error"],
            "version_no": version_no,
            "chunk_count": chunk_count,
            "embedding_model": embedding_model,
            "updated_at": func.now(),
        },
    )
    await conn.execute(stmt)


def _supersedes(indexed: IndexedLesson, doc: LessonDocument) -> bool:
    """True if what is indexed must not be replaced by `doc` (it is newer, or a different version with the same
    number, which Django never produces)."""
    return indexed.version_no > doc.version_no or (
        indexed.version_no == doc.version_no and indexed.content_version_id != doc.content_version_id
    )


def _stale(indexed: IndexedLesson) -> AppError:
    return AppError(
        409,
        "stale_version",
        f"A newer version (v{indexed.version_no}) of this lesson is already indexed.",
        {"indexed_version_no": str(indexed.version_no)},
    )


def _check_vectors(vectors: list[list[float]], expected: int, provider: ModelProvider) -> None:
    if len(vectors) != expected:
        raise ProviderError(f"expected {expected} embeddings, got {len(vectors)}", provider=provider.name)
    wrong = next((len(v) for v in vectors if len(v) != EMBEDDING_DIMENSIONS), None)
    if wrong is not None:
        raise ProviderError(
            f"embedding has {wrong} dimensions; the index stores {EMBEDDING_DIMENSIONS}", provider=provider.name
        )


# ---------------------------------------------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------------------------------------------


async def index_lesson(engine: AsyncEngine, provider: ModelProvider, doc: LessonDocument, key: str) -> IndexResult:
    def result(chunks: int, model: str, status: Literal["indexed", "unchanged"]) -> IndexResult:
        return IndexResult(doc.lesson_id, doc.content_version_id, doc.version_no, chunks, model, status)

    async with engine.connect() as conn:
        event = await _event(conn, key)
        current = await _current(conn, doc.lesson_id)

    if event is not None:
        _check_same_request(event, doc.lesson_id, "index", doc.content_version_id)
        if event.status == "done":
            return result(event.chunk_count, event.embedding_model, "unchanged")

    if current is not None and _supersedes(current, doc):
        raise _stale(current)
    if (
        current is not None
        and current.content_version_id == doc.content_version_id
        and current.embedding_model == provider.embedding_model
    ):
        async with engine.begin() as conn:
            await _record(
                conn,
                key=key,
                lesson_id=doc.lesson_id,
                action="index",
                status="done",
                content_version_id=doc.content_version_id,
                version_no=doc.version_no,
                chunk_count=current.chunks,
                embedding_model=current.embedding_model,
            )
        return result(current.chunks, current.embedding_model, "unchanged")

    chunks = chunk_sections(doc.title, doc.sections)
    if not chunks:
        raise AppError(400, "empty_lesson", "The lesson has no text to index.")
    if len(chunks) > MAX_CHUNKS_PER_LESSON:
        raise AppError(
            400, "lesson_too_large", f"The lesson makes {len(chunks)} passages; the limit is {MAX_CHUNKS_PER_LESSON}."
        )

    try:
        # Optimized batch embedding: compute all passage vectors in parallel before DB transaction
        vectors = await provider.embed([c.embed_text for c in chunks], kind="document")
        _check_vectors(vectors, len(chunks), provider)
    except ProviderError as exc:
        await _record_failure(engine, key, doc, str(exc))
        raise

    rows = [
        {
            "lesson_id": doc.lesson_id,
            "content_version_id": doc.content_version_id,
            "version_no": doc.version_no,
            "course_id": doc.course_id,
            "subject": doc.subject,
            "class_number": doc.class_number,
            "position": chunk.position,
            "heading": chunk.heading[:200],
            "text": chunk.text,
            "token_count": chunk.token_count,
            "embedding": vector,
            "embedding_model": provider.embedding_model,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    async with engine.begin() as conn:
        await _lock_lesson(conn, doc.lesson_id)
        latest = await _current(conn, doc.lesson_id)  # another request may have finished while we were embedding
        if latest is not None and _supersedes(latest, doc):
            raise _stale(latest)
        await conn.execute(delete(lesson_chunk).where(lesson_chunk.c.lesson_id == doc.lesson_id))
        await conn.execute(insert(lesson_chunk), rows)
        await _record(
            conn,
            key=key,
            lesson_id=doc.lesson_id,
            action="index",
            status="done",
            content_version_id=doc.content_version_id,
            version_no=doc.version_no,
            chunk_count=len(rows),
            embedding_model=provider.embedding_model,
        )
    logger.info(
        "lesson indexed",
        extra={
            "lesson_id": str(doc.lesson_id),
            "version_no": doc.version_no,
            "chunks": len(rows),
            "embedding_model": provider.embedding_model,
        },
    )
    return result(len(rows), provider.embedding_model, "indexed")


async def _record_failure(engine: AsyncEngine, key: str, doc: LessonDocument, error: str) -> None:
    """Best effort: a failure to record the failure must not hide the original error."""
    try:
        async with engine.begin() as conn:
            await _record(
                conn,
                key=key,
                lesson_id=doc.lesson_id,
                action="index",
                status="failed",
                content_version_id=doc.content_version_id,
                version_no=doc.version_no,
                error=error,
            )
    except Exception:  # the provider error is what the caller needs to see
        logger.warning("could not record a failed index event", exc_info=True)


async def delete_lesson(engine: AsyncEngine, lesson_id: uuid.UUID, key: str, up_to_version: int | None = None) -> int:
    """Remove a lesson's chunks; returns how many were removed (0 when it wasn't indexed).

    `up_to_version` (Django sends the lesson's highest version number) makes a delayed delete harmless: if a newer
    version has been indexed since, nothing is removed and 409 `stale_version` is returned.
    """
    async with engine.begin() as conn:
        event = await _event(conn, key)
        if event is not None:
            _check_same_request(event, lesson_id, "delete", None)
            if event.status == "done":
                return int(event.chunk_count or 0)
        await _lock_lesson(conn, lesson_id)
        current = await _current(conn, lesson_id)
        if current is not None and up_to_version is not None and current.version_no > up_to_version:
            raise _stale(current)
        removed = (await conn.execute(delete(lesson_chunk).where(lesson_chunk.c.lesson_id == lesson_id))).rowcount
        await _record(conn, key=key, lesson_id=lesson_id, action="delete", status="done", chunk_count=removed)
    logger.info("lesson removed from the index", extra={"lesson_id": str(lesson_id), "chunks": removed})
    return int(removed)


async def indexed_lessons(engine: AsyncEngine) -> list[IndexedLesson]:
    async with engine.connect() as conn:
        rows = (await conn.execute(_indexed_query().order_by(lesson_chunk.c.lesson_id))).all()
    return [_to_indexed(row) for row in rows]
