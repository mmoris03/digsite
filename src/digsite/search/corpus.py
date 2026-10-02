"""Searching an indexed corpus: the retrievers built from what the index stage stored."""

import sqlite3
from collections.abc import Callable
from enum import StrEnum
from functools import cached_property

from digsite.embedding import Embedder, create_embedder
from digsite.index.analyzer import Analyzer
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.vector_index import VectorIndex
from digsite.search.bm25 import Bm25Params
from digsite.search.dense import DenseSearcher
from digsite.search.expansion import ExpansionSettings
from digsite.search.fusion import DEFAULT_RANK_CONSTANT
from digsite.search.grouping import search_best_per_group
from digsite.search.hybrid import HybridSearcher
from digsite.search.lexical import LexicalSearcher
from digsite.search.results import DocumentResult, section_of
from digsite.search.retriever import Retriever
from digsite.store import AuthorityStore, ChunkStore, DocumentStore


class SearchMode(StrEnum):
    """How queries are matched against the corpus."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MissingIndexError(RuntimeError):
    """The corpus lacks something a search mode needs. The message says how to build it."""


class CorpusSearch:
    """The retrievers of an indexed corpus, and the documents they find.

    The retrievers rank chunks and return their ids. Each one is built the
    first time it is asked for, so that a lexical search does not pay for
    loading the embedding model. Once built, they are kept: one instance can
    serve every search of a long-running process.

    Args:
        connection: An open corpus database.
        bm25: BM25 settings of lexical search.
        expansion: How lexical search expands queries. None searches for the
            query as typed.
        rank_constant: See `reciprocal_rank_fusion`.
        authority_weight: How much the importance of a document according to
            the links it receives counts in hybrid search, a retriever counting
            1. 0 ignores it.
        embedder_factory: Builds the embedder with a given name.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        bm25: Bm25Params | None = None,
        expansion: ExpansionSettings | None = None,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        authority_weight: float = 0.0,
        embedder_factory: Callable[[str], Embedder] = create_embedder,
    ) -> None:
        self._chunks = ChunkStore(connection)
        self._documents = DocumentStore(connection)
        self._index_store = LexicalIndexStore(connection)
        self._authority = AuthorityStore(connection)
        self._bm25 = bm25
        self._expansion = expansion
        self._rank_constant = rank_constant
        self._authority_weight = authority_weight
        self._embedder_factory = embedder_factory

    @property
    def chunks(self) -> ChunkStore:
        """Where the chunks of the corpus are read from."""
        return self._chunks

    @property
    def documents(self) -> DocumentStore:
        """Where the documents of the corpus are read from."""
        return self._documents

    @property
    def has_embeddings(self) -> bool:
        """Tell whether the corpus was indexed with an embedding model."""
        return self._chunks.embedding_model() is not None

    @property
    def default_mode(self) -> SearchMode:
        """Hybrid search if the corpus allows it; lexical otherwise."""
        return SearchMode.HYBRID if self.has_embeddings else SearchMode.LEXICAL

    def retriever(self, mode: SearchMode) -> Retriever[int]:
        """Return the retriever for a search mode.

        Raises:
            MissingIndexError: If the corpus was not indexed, or not with what
                the mode needs.
        """
        match mode:
            case SearchMode.LEXICAL:
                return self.lexical
            case SearchMode.SEMANTIC:
                return self.semantic
            case SearchMode.HYBRID:
                return self.hybrid

    @cached_property
    def lexical(self) -> LexicalSearcher[int]:
        loaded = self._index_store.load()
        if loaded is None:
            raise MissingIndexError("the corpus has no index; run 'digsite index' first")
        index, settings = loaded
        return LexicalSearcher(
            index, Analyzer(settings), self._bm25, expansion=self._expansion, text_of=self._text_of
        )

    @cached_property
    def semantic(self) -> DenseSearcher[int]:
        model = self._chunks.embedding_model()
        if model is None:
            raise MissingIndexError(
                "the corpus has no embeddings; run 'digsite index' without --no-embeddings"
            )
        keys, vectors = self._chunks.vectors(model)
        return DenseSearcher(VectorIndex(keys, vectors), self._embedder_factory(model))

    @cached_property
    def hybrid(self) -> HybridSearcher[int]:
        return self.hybrid_with_authority(self._authority_weight)

    def hybrid_with_authority(self, weight: float) -> HybridSearcher[int]:
        """Return hybrid search with the given weight for link authority.

        It shares the lexical index and the embedding model with the other
        retrievers, so comparing weights does not load them again.
        """
        retrievers: list[Retriever[int]] = [self.lexical, self.semantic]
        if weight <= 0:
            return HybridSearcher(retrievers, rank_constant=self._rank_constant)
        return HybridSearcher(
            retrievers,
            rank_constant=self._rank_constant,
            prior=self._authority_positions.__getitem__,
            prior_weight=weight,
        )

    def find_documents(self, query: str, mode: SearchMode, limit: int = 10) -> list[DocumentResult]:
        """Search the corpus and return documents, each shown by its best passage.

        Raises:
            MissingIndexError: If the corpus was not indexed for the mode.
        """
        best = search_best_per_group(self.retriever(mode), query, self.page_ids.__getitem__, limit)
        results = []
        for hit in best:
            chunk = self._chunks.chunk(hit.key)
            document = self._documents.document(chunk.page_id) if chunk else None
            # Both exist unless the corpus changed after it was indexed.
            if chunk is None or document is None:
                continue
            results.append(
                DocumentResult(
                    url=document.url,
                    title=document.title,
                    section=section_of(chunk.context, document.title),
                    passage=chunk.text,
                    score=hit.score,
                    chunk_id=chunk.id,
                )
            )
        return results

    @cached_property
    def page_ids(self) -> dict[int, int]:
        """The document each chunk belongs to: page id by chunk id."""
        return self._chunks.page_ids()

    @cached_property
    def _authority_positions(self) -> dict[int, int]:
        """Position of each chunk's document, from 1, by decreasing link score."""
        scores = self._authority.scores()
        if not scores:
            raise MissingIndexError("the corpus has no link scores; run 'digsite index' first")
        ranked = sorted(scores, key=lambda page_id: (-scores[page_id], page_id))
        position = {page_id: rank for rank, page_id in enumerate(ranked, start=1)}
        # A document indexed after the links were scored counts as the least important.
        last = len(ranked) + 1
        return {chunk: position.get(page, last) for chunk, page in self.page_ids.items()}

    def _text_of(self, chunk_id: int) -> str:
        chunk = self._chunks.chunk(chunk_id)
        return chunk.indexed_text if chunk else ""
