"""Storage for chunks and their embeddings."""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from digsite.models import Chunk, Passage, indexed_text

_MODEL_SETTING = "vector_index.model"
_FLOAT32 = np.dtype("<f4")


class ChunkStore:
    """Reads and writes chunks and embeddings on an open corpus database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def replace_all(self, documents: Iterable[tuple[int, Sequence[Passage]]]) -> int:
        """Replace every chunk with the passages of the given documents.

        Args:
            documents: For each document, its page id and its passages in order.

        Returns:
            The number of chunks stored.
        """
        rows = [
            (page_id, ordinal, passage.context, passage.text, _hash(passage.context, passage.text))
            for page_id, passages in documents
            for ordinal, passage in enumerate(passages)
        ]
        with self._db:
            self._db.execute("DELETE FROM chunks")
            self._db.executemany(
                "INSERT INTO chunks (page_id, ordinal, context, text, text_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def chunks(self) -> list[Chunk]:
        """Every chunk, in document and then passage order."""
        rows = self._db.execute(
            "SELECT id, page_id, ordinal, context, text FROM chunks ORDER BY id"
        )
        return [Chunk(*row) for row in rows]

    def chunk(self, chunk_id: int) -> Chunk | None:
        row = self._db.execute(
            "SELECT id, page_id, ordinal, context, text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return Chunk(*row) if row else None

    def page_ids(self) -> dict[int, int]:
        """The document each chunk belongs to: page id by chunk id."""
        return dict(self._db.execute("SELECT id, page_id FROM chunks ORDER BY id"))

    def count(self) -> int:
        value: int = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return value

    def document_count(self) -> int:
        """Number of documents that have at least one chunk."""
        value: int = self._db.execute("SELECT COUNT(DISTINCT page_id) FROM chunks").fetchone()[0]
        return value

    def distinct_text_count(self) -> int:
        """Number of different texts among the chunks; identical chunks share a vector."""
        value: int = self._db.execute("SELECT COUNT(DISTINCT text_hash) FROM chunks").fetchone()[0]
        return value

    def texts_without_embedding(self, model: str) -> list[tuple[str, str]]:
        """Distinct (hash, indexed text) of the chunks the model has no vector for."""
        rows = self._db.execute(
            """
            SELECT c.text_hash, c.context, c.text
            FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.model = ? AND e.text_hash = c.text_hash
            )
            GROUP BY c.text_hash
            ORDER BY MIN(c.id)
            """,
            (model,),
        )
        return [(row[0], indexed_text(row[1], row[2])) for row in rows]

    def save_embeddings(
        self, model: str, text_hashes: Sequence[str], vectors: NDArray[np.float32]
    ) -> None:
        """Store one vector per text hash for a model."""
        if len(text_hashes) != len(vectors):
            raise ValueError("Expected one vector per text hash")
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vector) VALUES (?, ?, ?)",
                (
                    (model, text_hash, vector.astype(_FLOAT32).tobytes())
                    for text_hash, vector in zip(text_hashes, vectors, strict=True)
                ),
            )

    def vectors(self, model: str) -> tuple[list[int], NDArray[np.float32]]:
        """Return the chunk ids that have a vector for the model, and those vectors by row."""
        rows = self._db.execute(
            """
            SELECT c.id, e.vector
            FROM chunks c JOIN embeddings e ON e.text_hash = c.text_hash
            WHERE e.model = ?
            ORDER BY c.id
            """,
            (model,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([np.frombuffer(row[1], dtype=_FLOAT32) for row in rows])
        return [row[0] for row in rows], matrix.astype(np.float32)

    def prune_embeddings(self) -> int:
        """Delete the vectors of texts that no chunk has any more. Returns how many."""
        with self._db:
            cursor = self._db.execute(
                "DELETE FROM embeddings WHERE text_hash NOT IN (SELECT text_hash FROM chunks)"
            )
        return cursor.rowcount

    def set_embedding_model(self, model: str | None) -> None:
        """Record which model the corpus is embedded with; None if with none."""
        with self._db:
            if model is None:
                self._db.execute("DELETE FROM settings WHERE name = ?", (_MODEL_SETTING,))
            else:
                self._db.execute(
                    "INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)",
                    (_MODEL_SETTING, json.dumps(model)),
                )

    def embedding_model(self) -> str | None:
        row = self._db.execute(
            "SELECT value FROM settings WHERE name = ?", (_MODEL_SETTING,)
        ).fetchone()
        return json.loads(row[0]) if row else None


def _hash(context: str, text: str) -> str:
    return hashlib.sha256(indexed_text(context, text).encode()).hexdigest()
