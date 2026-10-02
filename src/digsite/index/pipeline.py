"""The index stage: make the documents searchable.

Documents are split into chunks, the chunks go into the lexical and vector
indexes, and every document is scored by the links it receives.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from digsite.embedding import Embedder
from digsite.index.analyzer import Analyzer, AnalyzerSettings
from digsite.index.chunking import DEFAULT_MAX_CHARS, split_into_passages
from digsite.index.inverted_index import IndexBuilder, InvertedIndex
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.link_graph import document_links
from digsite.index.pagerank import DEFAULT_DAMPING, pagerank
from digsite.models import Document, Passage
from digsite.store import (
    AuthorityStore,
    ChunkStore,
    CrawlStore,
    DocumentStore,
)

logger = logging.getLogger(__name__)

_EMBEDDING_BATCH = 64


@dataclass(frozen=True, slots=True)
class IndexStats:
    """What one run of the index stage produced.

    Attributes:
        documents: Documents that were split.
        chunks: Chunks they were split into.
        terms: Distinct terms in the lexical index.
        postings: (term, chunk) pairs in the lexical index.
        embedded: Vectors computed by this run.
        reused: Vectors kept from earlier runs, for text that did not change.
    """

    documents: int
    chunks: int
    terms: int
    postings: int
    embedded: int
    reused: int


@dataclass(frozen=True, slots=True)
class AuthorityStats:
    """What ranking the documents by their links produced.

    Attributes:
        documents: Documents that were scored.
        links: Distinct links between them.
    """

    documents: int
    links: int


def index_texts[K](items: Iterable[tuple[K, str]], analyzer: Analyzer) -> InvertedIndex[K]:
    """Build an inverted index from (key, text) pairs."""
    builder: IndexBuilder[K] = IndexBuilder()
    for key, text in items:
        builder.add(key, analyzer.analyze(text))
    return builder.build()


def document_passages(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Passage]:
    """Split a document into passages that carry the document's title as context."""
    passages = split_into_passages(document.text, max_chars)
    if not document.title:
        return passages
    # Text outside any heading still belongs to a document that has a title.
    return [
        passage if passage.headings else Passage((document.title,), passage.text)
        for passage in passages
    ]


def build_index(
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
    settings: AnalyzerSettings,
    *,
    embedder: Embedder | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IndexStats:
    """Chunk every unique document and build the lexical and vector indexes.

    Chunks and the lexical index are rebuilt from scratch. Embeddings are only
    computed for text that has none yet, which is what makes re-indexing after
    a small change to the corpus cheap.

    Args:
        document_store: Source of documents.
        chunk_store: Destination of chunks and embeddings.
        index_store: Destination of the lexical index.
        settings: How text is analyzed for the lexical index.
        embedder: Model to embed the chunks with. None skips the vector index.
        max_chars: Upper bound on the size of a chunk.
    """
    documents = document_store.documents()
    chunk_store.replace_all(
        (document.page_id, document_passages(document, max_chars)) for document in documents
    )
    chunks = chunk_store.chunks()

    index = index_texts(((chunk.id, chunk.indexed_text) for chunk in chunks), Analyzer(settings))
    index_store.save(index, settings)

    embedded = reused = 0
    if embedder is not None:
        pending = chunk_store.texts_without_embedding(embedder.name)
        for start in range(0, len(pending), _EMBEDDING_BATCH):
            batch = pending[start : start + _EMBEDDING_BATCH]
            vectors = embedder.embed_passages([text for _, text in batch])
            chunk_store.save_embeddings(embedder.name, [key for key, _ in batch], vectors)
            logger.info(
                "embedded %d/%d chunks", min(start + len(batch), len(pending)), len(pending)
            )
        embedded = len(pending)
        reused = chunk_store.distinct_text_count() - embedded
    chunk_store.set_embedding_model(embedder.name if embedder is not None else None)
    chunk_store.prune_embeddings()

    return IndexStats(
        documents=len(documents),
        chunks=len(chunks),
        terms=index.vocabulary_size,
        postings=index.postings_count,
        embedded=embedded,
        reused=reused,
    )


def build_authority(
    crawl_store: CrawlStore,
    document_store: DocumentStore,
    authority_store: AuthorityStore,
    *,
    damping: float = DEFAULT_DAMPING,
) -> AuthorityStats:
    """Score every unique document by the links it receives from the others.

    Args:
        crawl_store: Source of the links and redirects recorded by the crawl.
        document_store: Source of the documents and their duplicates.
        authority_store: Destination of the scores.
        damping: PageRank's probability of following a link.
    """
    page_of = {document.url: document.page_id for document in document_store.documents()}
    duplicates = {
        duplicate.url: duplicate.canonical_url for duplicate in document_store.duplicates()
    }
    links = document_links(
        crawl_store.edges(),
        page_of.keys(),
        redirects=crawl_store.redirects(),
        duplicates=duplicates,
    )
    scores = pagerank(list(page_of), links, damping=damping)
    authority_store.replace_all({page_of[url]: score for url, score in scores.items()})
    return AuthorityStats(documents=len(page_of), links=len(links))
