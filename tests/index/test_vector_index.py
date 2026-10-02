import numpy as np
import pytest

from digsite.index.vector_index import VectorIndex


def unit(*components: float) -> np.ndarray:
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


VECTORS = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0), unit(0, 0, 1)])
INDEX = VectorIndex(["x", "y", "xy", "z"], VECTORS)


def test_returns_the_most_similar_items_first() -> None:
    results = INDEX.search(unit(1, 0.2, 0), limit=3)

    assert [key for key, _ in results] == ["x", "xy", "y"]
    assert results[0][1] == pytest.approx(0.980581, abs=1e-5)


def test_scores_are_cosine_similarities() -> None:
    scores = dict(INDEX.search(unit(1, 0, 0), limit=4))

    assert scores["x"] == pytest.approx(1.0)
    assert scores["xy"] == pytest.approx(2**-0.5)
    assert scores["y"] == pytest.approx(0.0)
    assert scores["z"] == pytest.approx(0.0)


def test_limit_keeps_the_best() -> None:
    assert [key for key, _ in INDEX.search(unit(0, 0, 1), limit=1)] == ["z"]
    assert INDEX.search(unit(0, 0, 1), limit=0) == []
    assert len(INDEX.search(unit(0, 0, 1), limit=100)) == 4


def test_ties_are_broken_by_index_order() -> None:
    index = VectorIndex(["c", "a", "b"], np.stack([unit(1, 0), unit(1, 0), unit(1, 0)]))

    assert [key for key, _ in index.search(unit(1, 0), limit=2)] == ["c", "a"]


def test_agrees_with_a_full_sort() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(500, 24)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = VectorIndex(list(range(500)), vectors)
    query = vectors[123]

    results = index.search(query, limit=20)

    expected = np.argsort(-(vectors @ query), kind="stable")[:20]
    assert [key for key, _ in results] == expected.tolist()
    assert results[0] == (123, pytest.approx(1.0, abs=1e-5))


def test_reports_size_and_dimension() -> None:
    assert len(INDEX) == 4
    assert INDEX.dimension == 3


def test_an_empty_index_returns_nothing() -> None:
    index: VectorIndex[str] = VectorIndex([], np.zeros((0, 0), dtype=np.float32))

    assert len(index) == 0
    assert index.search(unit(1, 0), limit=5) == []


def test_rejects_mismatched_keys_and_vectors() -> None:
    with pytest.raises(ValueError, match="one vector per key"):
        VectorIndex(["a", "b"], VECTORS)


def test_rejects_a_query_of_the_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="Query has shape"):
        INDEX.search(unit(1, 0), limit=1)
