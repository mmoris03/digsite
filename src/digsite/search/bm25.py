"""BM25 ranking over an inverted index.

BM25 scores an item for a query as the sum, over the query terms it contains,
of two factors: how rare the term is in the collection (IDF), and how often it
occurs in the item, with diminishing returns and a correction for item length.

The variants differ in the exact shape of those two factors. They are the ones
compared by Kamphuis et al. (2020), "Which BM25 do you mean?".
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from digsite.index.inverted_index import InvertedIndex

type Floats = NDArray[np.float64]


class Bm25Variant(StrEnum):
    LUCENE = "lucene"
    ROBERTSON = "robertson"
    ATIRE = "atire"
    BM25L = "bm25l"
    BM25_PLUS = "bm25+"


@dataclass(frozen=True, slots=True)
class Bm25Params:
    """BM25 settings.

    Attributes:
        variant: Which formulation to use.
        k1: Term-frequency saturation. Higher values let repeated occurrences
            keep adding to the score for longer.
        b: Length normalisation, from 0 (ignore item length) to 1 (full).
        delta: Lower bound added by BM25L and BM25+ so that long items are not
            penalised too much. None means the variant's usual value.
    """

    variant: Bm25Variant = Bm25Variant.LUCENE
    k1: float = 1.2
    b: float = 0.75
    delta: float | None = None


class Bm25Ranker[K]:
    """Ranks the items of an inverted index against a bag of query terms."""

    def __init__(self, index: InvertedIndex[K], params: Bm25Params | None = None) -> None:
        self._index = index
        self._params = params or Bm25Params()
        average = index.average_length or 1.0
        b = self._params.b
        # Length normalisation of every item: 1 for an item of average length.
        self._norms: Floats = 1.0 - b + b * index.lengths.astype(np.float64) / average

    def rank(
        self, terms: Iterable[str] | Mapping[str, float], limit: int = 10
    ) -> list[tuple[K, float]]:
        """Return the best-scoring items that contain at least one query term.

        Args:
            terms: Query terms, already analyzed. Repetitions are ignored. A
                mapping gives each term a weight that multiplies its contribution.
            limit: Maximum number of results.

        Returns:
            (key, score) pairs by decreasing score; ties keep index order.
        """
        if limit <= 0:
            return []
        query = terms if isinstance(terms, Mapping) else dict.fromkeys(terms, 1.0)
        count = len(self._index)
        scores = np.zeros(count, dtype=np.float64)
        matched = np.zeros(count, dtype=np.bool_)
        for term, weight in query.items():
            postings = self._index.postings(term)
            if postings is None:
                continue
            positions = postings.positions
            term_weights = self._term_weights(
                postings.frequencies.astype(np.float64), self._norms[positions]
            )
            scores[positions] += weight * self._idf(len(positions)) * term_weights
            matched[positions] = True

        candidates = np.flatnonzero(matched)
        if len(candidates) > limit:
            # Keep the `limit` best before sorting: most candidates are irrelevant.
            best = np.argpartition(-scores[candidates], limit - 1)[:limit]
            threshold = scores[candidates[best]].min()
            candidates = candidates[scores[candidates] >= threshold]
        order = np.lexsort((candidates, -scores[candidates]))[:limit]
        keys = self._index.keys
        return [(keys[position], float(scores[position])) for position in candidates[order]]

    def _idf(self, document_frequency: int) -> float:
        count = len(self._index)
        match self._params.variant:
            case Bm25Variant.LUCENE:
                return math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
            case Bm25Variant.ROBERTSON:
                return math.log((count - document_frequency + 0.5) / (document_frequency + 0.5))
            case Bm25Variant.ATIRE:
                return math.log(count / document_frequency)
            case Bm25Variant.BM25L:
                return math.log((count + 1.0) / (document_frequency + 0.5))
            case Bm25Variant.BM25_PLUS:
                return math.log((count + 1.0) / document_frequency)

    def _term_weights(self, frequencies: Floats, norms: Floats) -> Floats:
        k1 = self._params.k1
        match self._params.variant:
            case Bm25Variant.LUCENE | Bm25Variant.ROBERTSON:
                return frequencies / (frequencies + k1 * norms)
            case Bm25Variant.ATIRE:
                return (k1 + 1.0) * frequencies / (frequencies + k1 * norms)
            case Bm25Variant.BM25L:
                delta = 0.5 if self._params.delta is None else self._params.delta
                adjusted = frequencies / norms + delta
                return (k1 + 1.0) * adjusted / (k1 + adjusted)
            case Bm25Variant.BM25_PLUS:
                delta = 1.0 if self._params.delta is None else self._params.delta
                return (k1 + 1.0) * frequencies / (frequencies + k1 * norms) + delta
