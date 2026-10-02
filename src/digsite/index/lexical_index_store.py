"""Storage for the lexical index and the analyzer it was built with."""

import json
import sqlite3
from dataclasses import asdict

import numpy as np

from digsite.index.analyzer import AnalyzerSettings, Language
from digsite.index.inverted_index import InvertedIndex, Postings

_ANALYZER_SETTING = "lexical_index.analyzer"
_UINT32 = np.dtype("<u4")


class LexicalIndexStore:
    """Saves and loads the inverted index on an open corpus database.

    The index is written and read whole. Postings are stored as one packed
    array per term, so loading does not create a Python object per posting.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def save(self, index: InvertedIndex[int], settings: AnalyzerSettings) -> None:
        """Replace the stored index.

        The analyzer settings are stored with it: a query must be analyzed
        exactly as the documents were.
        """
        with self._db:
            self._db.execute("DELETE FROM lexical_items")
            self._db.execute("DELETE FROM lexical_terms")
            self._db.executemany(
                "INSERT INTO lexical_items (position, item_id, length) VALUES (?, ?, ?)",
                (
                    (position, key, int(length))
                    for position, (key, length) in enumerate(
                        zip(index.keys, index.lengths, strict=True)
                    )
                ),
            )
            self._db.executemany(
                "INSERT INTO lexical_terms (term, positions, frequencies) VALUES (?, ?, ?)",
                (
                    (
                        term,
                        postings.positions.astype(_UINT32).tobytes(),
                        postings.frequencies.astype(_UINT32).tobytes(),
                    )
                    for term, postings in index.terms()
                ),
            )
            self._db.execute(
                "INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)",
                (_ANALYZER_SETTING, json.dumps(asdict(settings))),
            )

    def load(self) -> tuple[InvertedIndex[int], AnalyzerSettings] | None:
        """Return the stored index and its analyzer settings, or None if there is none."""
        row = self._db.execute(
            "SELECT value FROM settings WHERE name = ?", (_ANALYZER_SETTING,)
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])
        settings = AnalyzerSettings(
            language=Language(raw["language"]) if raw["language"] else None,
            remove_stopwords=raw["remove_stopwords"],
            stem=raw["stem"],
        )

        items = self._db.execute(
            "SELECT item_id, length FROM lexical_items ORDER BY position"
        ).fetchall()
        keys = [item[0] for item in items]
        lengths = np.array([item[1] for item in items], dtype=np.uint32)
        postings = {
            term: Postings(
                np.frombuffer(positions, dtype=_UINT32).astype(np.uint32),
                np.frombuffer(frequencies, dtype=_UINT32).astype(np.uint32),
            )
            for term, positions, frequencies in self._db.execute(
                "SELECT term, positions, frequencies FROM lexical_terms"
            )
        }
        return InvertedIndex(keys, lengths, postings), settings
