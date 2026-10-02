"""Storage for what a collection is: its title and where it comes from."""

import json
import sqlite3

from digsite.models import CollectionInfo

_INFO_SETTING = "collection.info"


class CollectionStore:
    """Reads and writes the description of the collection an open database holds."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def info(self) -> CollectionInfo | None:
        row = self._db.execute(
            "SELECT value FROM settings WHERE name = ?", (_INFO_SETTING,)
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])
        return CollectionInfo(title=raw["title"], source=raw["source"])

    def save(self, info: CollectionInfo) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)",
                (_INFO_SETTING, json.dumps({"title": info.title, "source": info.source})),
            )
