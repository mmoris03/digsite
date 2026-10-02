import numpy as np
import pytest

from digsite.embedding import HashingEmbedder, create_embedder


def test_vectors_have_the_requested_dimension_and_unit_length() -> None:
    embedder = HashingEmbedder(64)

    vectors = embedder.embed_passages(["the cache stores results", "logs rotate daily"])

    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert embedder.dimension == 64
    assert embedder.name == "hashing-64"


def test_the_same_text_always_gets_the_same_vector() -> None:
    first = HashingEmbedder().embed_query("cache eviction policy")
    second = HashingEmbedder().embed_query("cache eviction policy")

    assert np.array_equal(first, second)


def test_queries_and_passages_share_one_space() -> None:
    embedder = HashingEmbedder()

    assert np.array_equal(embedder.embed_query("cache"), embedder.embed_passages(["cache"])[0])


def test_texts_that_share_words_are_closer_than_texts_that_do_not() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("how does the cache evict entries")
    related, unrelated = embedder.embed_passages(
        ["the cache evicts old entries first", "install the package with pip"]
    )

    assert float(query @ related) > float(query @ unrelated)


def test_case_and_punctuation_do_not_matter() -> None:
    embedder = HashingEmbedder()

    assert np.array_equal(
        embedder.embed_query("Cache, EVICTION!"), embedder.embed_query("cache eviction")
    )


def test_text_without_words_is_the_zero_vector() -> None:
    vector = HashingEmbedder(16).embed_query("... !!")

    assert vector.shape == (16,)
    assert not vector.any()


def test_no_texts_give_an_empty_matrix() -> None:
    assert HashingEmbedder(16).embed_passages([]).shape == (0, 16)


def test_the_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        HashingEmbedder(0)


@pytest.mark.parametrize(("name", "dimension"), [("hashing", 256), ("hashing-32", 32)])
def test_create_embedder_builds_the_model_free_embedder_by_name(name: str, dimension: int) -> None:
    embedder = create_embedder(name)

    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dimension == dimension
