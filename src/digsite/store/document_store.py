"""Storage for the documents extracted from pages and their duplicate marks."""

import sqlite3
from collections.abc import Mapping

from digsite.models import (
    Document,
    DocumentStats,
    Duplicate,
    DuplicateDocument,
    DuplicateKind,
    Fingerprint,
)


class DocumentStore:
    """Reads and writes documents on an open corpus database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def save(
        self,
        page_id: int,
        title: str,
        text: str,
        content_hash: str,
        simhash: int | None,
    ) -> None:
        """Store the document of a page, replacing any previous one.

        Args:
            page_id: The page the document was extracted from.
            title: Document title.
            text: Main content as Markdown; empty if nothing could be extracted.
            content_hash: Hash used to detect exact duplicates.
            simhash: Fingerprint used to detect near-duplicates; None for empty text.
        """
        # SQLite integers are signed 64-bit, too narrow for an unsigned fingerprint.
        blob = _to_bytes(simhash) if simhash is not None else None
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO documents (page_id, title, text, content_hash, simhash) "
                "VALUES (?, ?, ?, ?, ?)",
                (page_id, title, text, content_hash, blob),
            )

    def page_ids(self) -> set[int]:
        """Ids of the pages that already have a document, empty or not."""
        return {row[0] for row in self._db.execute("SELECT page_id FROM documents")}

    def fingerprints(self) -> list[Fingerprint]:
        """Fingerprints of every document with content, in page order."""
        rows = self._db.execute(
            "SELECT page_id, content_hash, simhash FROM documents WHERE text <> '' ORDER BY page_id"
        )
        return [Fingerprint(row[0], row[1], int.from_bytes(row[2], "big")) for row in rows]

    def set_duplicates(self, duplicates: Mapping[int, Duplicate]) -> None:
        """Replace every duplicate mark with the given ones, keyed by page id."""
        with self._db:
            self._db.execute("UPDATE documents SET duplicate_of = NULL, duplicate_kind = NULL")
            self._db.executemany(
                "UPDATE documents SET duplicate_of = ?, duplicate_kind = ? WHERE page_id = ?",
                (
                    (duplicate.canonical_id, duplicate.kind.value, page_id)
                    for page_id, duplicate in duplicates.items()
                ),
            )

    def clear(self) -> None:
        """Delete every document, so that the next ingest starts from scratch."""
        with self._db:
            self._db.execute("DELETE FROM documents")

    def documents(self) -> list[Document]:
        """Documents with content that do not repeat another one, in page order."""
        rows = self._db.execute(
            """
            SELECT d.page_id, p.url, d.title, d.text
            FROM documents d JOIN pages p ON p.id = d.page_id
            WHERE d.text <> '' AND d.duplicate_of IS NULL
            ORDER BY d.page_id
            """
        )
        return [Document(row[0], row[1], row[2], row[3]) for row in rows]

    def document(self, page_id: int) -> Document | None:
        """Return the document of a page, or None if the page has none."""
        row = self._db.execute(
            """
            SELECT d.page_id, p.url, d.title, d.text
            FROM documents d JOIN pages p ON p.id = d.page_id
            WHERE d.page_id = ?
            """,
            (page_id,),
        ).fetchone()
        return Document(row[0], row[1], row[2], row[3]) if row else None

    def duplicates(self) -> list[DuplicateDocument]:
        """Every duplicate with the document that stands for it, in page order."""
        rows = self._db.execute(
            """
            SELECT p.url, canonical.url, d.duplicate_kind
            FROM documents d
            JOIN pages p ON p.id = d.page_id
            JOIN pages canonical ON canonical.id = d.duplicate_of
            ORDER BY d.page_id
            """
        )
        return [DuplicateDocument(row[0], row[1], DuplicateKind(row[2])) for row in rows]

    def stats(self) -> DocumentStats:
        row = self._db.execute(
            """
            SELECT
                COALESCE(SUM(text <> '' AND duplicate_of IS NULL), 0),
                COALESCE(SUM(duplicate_kind = 'exact'), 0),
                COALESCE(SUM(duplicate_kind = 'near'), 0),
                COALESCE(SUM(text = ''), 0)
            FROM documents
            """
        ).fetchone()
        return DocumentStats(
            unique=row[0], exact_duplicates=row[1], near_duplicates=row[2], empty=row[3]
        )


def _to_bytes(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
