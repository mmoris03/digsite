"""What every way of searching has in common."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Hit[K]:
    """One search result: the key of an indexed item and its score."""

    key: K
    score: float


class Retriever[K](Protocol):
    """Anything that answers a text query with a ranked list of items.

    Lexical, semantic and hybrid search all have this shape, which is what lets
    them be compared in evaluation and combined with each other.
    """

    def search(self, query: str, limit: int = 10) -> list[Hit[K]]:
        """Return the items that best match the query, best first."""
        ...
