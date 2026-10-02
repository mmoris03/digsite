"""Lexical search: free-text queries against an inverted index."""

from collections import Counter
from collections.abc import Callable

from digsite.index.analyzer import Analyzer
from digsite.index.inverted_index import InvertedIndex
from digsite.search.bm25 import Bm25Params, Bm25Ranker
from digsite.search.expansion import ExpansionSettings, expansion_terms
from digsite.search.retriever import Hit


class LexicalSearcher[K]:
    """Answers text queries with BM25, optionally expanding them first.

    Args:
        index: The index to search.
        analyzer: The analyzer the index was built with.
        params: BM25 settings.
        expansion: How to expand queries by pseudo-relevance feedback. None
            searches for the query as typed.
        text_of: Returns the text an item was indexed with, given its key.
            Required for expansion, which reads the top results.
    """

    def __init__(
        self,
        index: InvertedIndex[K],
        analyzer: Analyzer,
        params: Bm25Params | None = None,
        *,
        expansion: ExpansionSettings | None = None,
        text_of: Callable[[K], str] | None = None,
    ) -> None:
        if expansion is not None and text_of is None:
            raise ValueError("Query expansion needs `text_of` to read the feedback documents")
        self._index = index
        self._analyzer = analyzer
        self._ranker = Bm25Ranker(index, params)
        self._expansion = expansion
        self._text_of = text_of

    @property
    def analyzer(self) -> Analyzer:
        """The analyzer queries go through."""
        return self._analyzer

    def search(self, query: str, limit: int = 10) -> list[Hit[K]]:
        """Return the items that best match the query, best first."""
        terms = self.query_terms(query)
        return [Hit(key, score) for key, score in self._ranker.rank(terms, limit)]

    def query_terms(self, query: str) -> dict[str, float]:
        """Return the weighted terms a query is searched with.

        The terms of the query itself weigh 1. With expansion enabled, the terms
        added by feedback follow them, with the configured weight.
        """
        terms = dict.fromkeys(self._analyzer.analyze(query), 1.0)
        if self._expansion is None or self._text_of is None or not terms:
            return terms

        feedback = [
            Counter(self._analyzer.analyze(self._text_of(key)))
            for key, _ in self._ranker.rank(terms, self._expansion.feedback_documents)
        ]
        for term, _ in expansion_terms(self._index, feedback, terms, self._expansion):
            terms[term] = self._expansion.weight
        return terms
