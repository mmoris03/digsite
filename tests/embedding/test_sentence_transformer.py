"""Tests that load the real embedding model.

They download it on first use and take several seconds, so they are left out of
the default run: `pytest -m model` runs them.
"""

import numpy as np
import pytest

from digsite.embedding import DEFAULT_MODEL, Embedder, create_embedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return create_embedder(DEFAULT_MODEL)


def test_reports_its_name_and_dimension(embedder: Embedder) -> None:
    assert embedder.name == DEFAULT_MODEL
    assert embedder.dimension == 384


def test_vectors_have_unit_length(embedder: Embedder) -> None:
    vectors = embedder.embed_passages(["El gato duerme en el sofá.", "Logs rotate daily."])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)
    assert np.linalg.norm(embedder.embed_query("gato")) == pytest.approx(1.0, abs=1e-4)


def test_no_texts_give_an_empty_matrix(embedder: Embedder) -> None:
    assert embedder.embed_passages([]).shape == (0, 384)


ANSWER = "Use pip to add third-party libraries to your environment."


def test_meaning_matters_more_than_a_shared_word(embedder: Embedder) -> None:
    query = embedder.embed_query("How do I install a package?")
    answer, shared_word = embedder.embed_passages(
        [ANSWER, "The package arrived by mail at the wrong office."]
    )

    # The answer shares no content word with the question; the other passage
    # repeats "package" in a different sense.
    assert float(query @ answer) > float(query @ shared_word)


def test_a_question_finds_its_answer_in_another_language(embedder: Embedder) -> None:
    query = embedder.embed_query("¿Cómo instalo un paquete?")
    answer, parcel, unrelated = embedder.embed_passages(
        [
            ANSWER,
            "The parcel was delivered to the wrong office.",
            "Tuples are immutable sequences.",
        ]
    )

    # The candidates are all in English on purpose: the model scores a passage in
    # the language of the query a little higher whatever it says, so across
    # languages only passages in the same language compare fairly.
    assert float(query @ answer) > float(query @ parcel) > float(query @ unrelated)


def test_queries_and_passages_are_marked_differently(embedder: Embedder) -> None:
    text = "pseudo-relevance feedback"

    as_query = embedder.embed_query(text)
    as_passage = embedder.embed_passages([text])[0]

    assert not np.allclose(as_query, as_passage)
    assert float(as_query @ as_passage) > 0.8
