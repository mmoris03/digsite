"""The inverted index: for every term, which items contain it and how often."""

from array import array
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Postings:
    """Where a term occurs.

    Attributes:
        positions: Internal positions of the items that contain the term,
            in increasing order.
        frequencies: How many times the term occurs in each of those items.
    """

    positions: NDArray[np.uint32]
    frequencies: NDArray[np.uint32]


class InvertedIndex[K]:
    """An immutable inverted index over a collection of items.

    Items are identified by a caller-chosen key and, internally, by their
    position: a dense number from 0 that makes postings compact and lets scores
    be accumulated in an array.

    Build one with `IndexBuilder`.
    """

    def __init__(
        self, keys: list[K], lengths: NDArray[np.uint32], postings: dict[str, Postings]
    ) -> None:
        self._keys = keys
        self._lengths = lengths
        self._postings = postings

    def __len__(self) -> int:
        """Number of indexed items."""
        return len(self._keys)

    @property
    def keys(self) -> list[K]:
        """Item keys, by position."""
        return self._keys

    @property
    def lengths(self) -> NDArray[np.uint32]:
        """Number of terms of each item, by position."""
        return self._lengths

    @property
    def average_length(self) -> float:
        return float(self._lengths.mean()) if len(self._keys) else 0.0

    @property
    def total_length(self) -> int:
        """Number of term occurrences in the whole collection."""
        return int(self._lengths.sum(dtype=np.uint64))

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    @property
    def postings_count(self) -> int:
        """Total number of (term, item) pairs."""
        return sum(len(postings.positions) for postings in self._postings.values())

    def postings(self, term: str) -> Postings | None:
        """Return the postings of a term, or None if no item contains it."""
        return self._postings.get(term)

    def collection_frequency(self, term: str) -> int:
        """Number of times a term occurs in the whole collection."""
        postings = self._postings.get(term)
        return int(postings.frequencies.sum(dtype=np.uint64)) if postings else 0

    def terms(self) -> Iterator[tuple[str, Postings]]:
        """Iterate over every term and its postings."""
        return iter(self._postings.items())


class IndexBuilder[K]:
    """Accumulates items and produces an `InvertedIndex`."""

    def __init__(self) -> None:
        self._keys: list[K] = []
        self._lengths = array("I")
        # Compact typed arrays: a large collection has tens of millions of postings.
        self._positions: dict[str, array[int]] = {}
        self._frequencies: dict[str, array[int]] = {}

    def add(self, key: K, terms: Iterable[str]) -> None:
        """Index an item under a key, given its terms with repetitions."""
        position = len(self._keys)
        counts = Counter(terms)
        self._keys.append(key)
        self._lengths.append(counts.total())
        for term, frequency in counts.items():
            positions = self._positions.get(term)
            if positions is None:
                positions = self._positions[term] = array("I")
                self._frequencies[term] = array("I")
            positions.append(position)
            self._frequencies[term].append(frequency)

    def build(self) -> InvertedIndex[K]:
        postings = {
            term: Postings(
                np.array(positions, dtype=np.uint32),
                np.array(self._frequencies[term], dtype=np.uint32),
            )
            for term, positions in self._positions.items()
        }
        return InvertedIndex(list(self._keys), np.array(self._lengths, dtype=np.uint32), postings)
