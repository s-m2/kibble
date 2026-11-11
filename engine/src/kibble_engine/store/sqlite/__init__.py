"""SQLite backend (default): sqlite-vec vectors, FTS5 keywords, plain-table graph."""

from kibble_engine.store.sqlite.store import (
    SCHEMA_VERSION,
    EmbedderMismatchError,
    SchemaVersionError,
    SqliteStore,
    StoreError,
    VectorBackendError,
)

__all__ = [
    "SCHEMA_VERSION",
    "EmbedderMismatchError",
    "SchemaVersionError",
    "SqliteStore",
    "StoreError",
    "VectorBackendError",
]
