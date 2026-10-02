import sqlite3

import numpy as np
import pytest

from digsite.models import Chunk, Passage
from digsite.store import ChunkStore, CrawlStore, DocumentStore

SITE = "https://example.com"
MODEL = "test-model"


@pytest.fixture
def chunk_store(connection: sqlite3.Connection) -> ChunkStore:
    return ChunkStore(connection)


@pytest.fixture
def page_ids(store: CrawlStore, document_store: DocumentStore) -> list[int]:
    ids = []
    for name in ("a", "b"):
        page_id = store.save_page(f"{SITE}/{name}", 0, b"<p>x</p>")
        document_store.save(page_id, name.upper(), f"text of {name}", f"hash-{name}", page_id)
        ids.append(page_id)
    return ids


PASSAGES_A = [Passage(("A", "Intro"), "first passage"), Passage(("A", "More"), "second passage")]
PASSAGES_B = [Passage((), "a passage without headings")]


def test_chunks_round_trip_in_document_and_passage_order(
    chunk_store: ChunkStore, page_ids: list[int]
) -> None:
    stored = chunk_store.replace_all([(page_ids[0], PASSAGES_A), (page_ids[1], PASSAGES_B)])

    assert stored == 3
    assert chunk_store.chunks() == [
        Chunk(1, page_ids[0], 0, "A > Intro", "first passage"),
        Chunk(2, page_ids[0], 1, "A > More", "second passage"),
        Chunk(3, page_ids[1], 0, "", "a passage without headings"),
    ]
    assert chunk_store.count() == 3
    assert chunk_store.document_count() == 2
    assert chunk_store.chunk(2) == Chunk(2, page_ids[0], 1, "A > More", "second passage")
    assert chunk_store.chunk(99) is None
    assert chunk_store.page_ids() == {1: page_ids[0], 2: page_ids[0], 3: page_ids[1]}


def test_the_indexed_text_puts_the_context_first(
    chunk_store: ChunkStore, page_ids: list[int]
) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A), (page_ids[1], PASSAGES_B)])
    with_context, _, without_context = chunk_store.chunks()

    assert with_context.indexed_text == "A > Intro\nfirst passage"
    assert without_context.indexed_text == "a passage without headings"


def test_replace_all_discards_the_previous_chunks(
    chunk_store: ChunkStore, page_ids: list[int]
) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A)])

    chunk_store.replace_all([(page_ids[1], PASSAGES_B)])

    assert [chunk.text for chunk in chunk_store.chunks()] == ["a passage without headings"]
    assert chunk_store.document_count() == 1


def test_a_chunk_needs_an_existing_document(chunk_store: ChunkStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        chunk_store.replace_all([(999, PASSAGES_B)])


def vectors(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_embeddings_are_missing_until_saved(chunk_store: ChunkStore, page_ids: list[int]) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A)])

    pending = chunk_store.texts_without_embedding(MODEL)

    assert [text for _, text in pending] == ["A > Intro\nfirst passage", "A > More\nsecond passage"]
    assert chunk_store.vectors(MODEL)[0] == []

    chunk_store.save_embeddings(MODEL, [key for key, _ in pending], vectors([1, 0], [0, 1]))

    assert chunk_store.texts_without_embedding(MODEL) == []
    keys, matrix = chunk_store.vectors(MODEL)
    assert keys == [1, 2]
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert matrix.dtype == np.float32


def test_embeddings_are_kept_per_model(chunk_store: ChunkStore, page_ids: list[int]) -> None:
    chunk_store.replace_all([(page_ids[1], PASSAGES_B)])
    pending = chunk_store.texts_without_embedding(MODEL)
    chunk_store.save_embeddings(MODEL, [pending[0][0]], vectors([1, 0]))

    assert len(chunk_store.texts_without_embedding("another-model")) == 1
    assert chunk_store.vectors("another-model")[0] == []


def test_embeddings_survive_re_chunking_when_the_text_is_unchanged(
    chunk_store: ChunkStore, page_ids: list[int]
) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A)])
    pending = chunk_store.texts_without_embedding(MODEL)
    chunk_store.save_embeddings(MODEL, [key for key, _ in pending], vectors([1, 0], [0, 1]))

    # The first passage is unchanged, the second is new; chunk ids are all new.
    changed = [PASSAGES_A[0], Passage(("A", "More"), "a rewritten passage")]
    chunk_store.replace_all([(page_ids[0], changed)])

    assert [text for _, text in chunk_store.texts_without_embedding(MODEL)] == [
        "A > More\na rewritten passage"
    ]
    keys, matrix = chunk_store.vectors(MODEL)
    assert len(keys) == 1
    assert matrix.tolist() == [[1.0, 0.0]]


def test_identical_chunks_share_one_embedding(chunk_store: ChunkStore, page_ids: list[int]) -> None:
    same = Passage(("Notice",), "the same legal notice")
    chunk_store.replace_all([(page_ids[0], [same]), (page_ids[1], [same])])

    pending = chunk_store.texts_without_embedding(MODEL)
    chunk_store.save_embeddings(MODEL, [pending[0][0]], vectors([0.6, 0.8]))

    assert len(pending) == 1
    assert chunk_store.distinct_text_count() == 1
    keys, matrix = chunk_store.vectors(MODEL)
    assert keys == [1, 2]
    assert np.allclose(matrix, [[0.6, 0.8], [0.6, 0.8]])


def test_prune_removes_embeddings_of_text_that_is_gone(
    chunk_store: ChunkStore, page_ids: list[int]
) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A)])
    pending = chunk_store.texts_without_embedding(MODEL)
    chunk_store.save_embeddings(MODEL, [key for key, _ in pending], vectors([1, 0], [0, 1]))
    chunk_store.replace_all([(page_ids[0], PASSAGES_A[:1])])

    assert chunk_store.prune_embeddings() == 1
    assert chunk_store.prune_embeddings() == 0
    assert len(chunk_store.vectors(MODEL)[0]) == 1


def test_save_embeddings_needs_one_vector_per_text(chunk_store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="one vector per text hash"):
        chunk_store.save_embeddings(MODEL, ["a", "b"], vectors([1, 0]))


def test_the_embedding_model_is_remembered(chunk_store: ChunkStore) -> None:
    assert chunk_store.embedding_model() is None

    chunk_store.set_embedding_model("intfloat/multilingual-e5-small")
    assert chunk_store.embedding_model() == "intfloat/multilingual-e5-small"

    chunk_store.set_embedding_model(None)
    assert chunk_store.embedding_model() is None


def test_deleting_a_document_deletes_its_chunks(
    chunk_store: ChunkStore, document_store: DocumentStore, page_ids: list[int]
) -> None:
    chunk_store.replace_all([(page_ids[0], PASSAGES_A)])

    document_store.clear()

    assert chunk_store.count() == 0
