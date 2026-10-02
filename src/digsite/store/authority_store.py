"""Storage for the importance of each document according to the link graph."""

import sqlite3
from collections.abc import Mapping


class AuthorityStore:
    """Reads and writes PageRank scores on an open corpus database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def replace_all(self, scores: Mapping[int, float]) -> None:
        """Replace every score with the given ones, keyed by page id."""
        with self._db:
            self._db.execute("DELETE FROM authority")
            self._db.executemany(
                "INSERT INTO authority (page_id, pagerank) VALUES (?, ?)", scores.items()
            )

    def scores(self) -> dict[int, float]:
        """The score of each document, by page id."""
        return dict(self._db.execute("SELECT page_id, pagerank FROM authority ORDER BY page_id"))
