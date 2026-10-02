import numpy as np

from digsite.index.inverted_index import IndexBuilder, InvertedIndex


def build(items: dict[str, str]) -> InvertedIndex[str]:
    builder: IndexBuilder[str] = IndexBuilder()
    for key, text in items.items():
        builder.add(key, text.split())
    return builder.build()


def test_postings_list_the_items_that_contain_a_term_and_how_often() -> None:
    index = build({"d1": "a a b", "d2": "a c", "d3": "c c c c"})

    a = index.postings("a")
    c = index.postings("c")

    assert a is not None
    assert c is not None
    assert a.positions.tolist() == [0, 1]
    assert a.frequencies.tolist() == [2, 1]
    assert c.positions.tolist() == [1, 2]
    assert c.frequencies.tolist() == [1, 4]
    assert index.postings("missing") is None


def test_records_keys_and_lengths_by_position() -> None:
    index = build({"d1": "a a b", "d2": "a c", "d3": "c c c c"})

    assert len(index) == 3
    assert index.keys == ["d1", "d2", "d3"]
    assert index.lengths.tolist() == [3, 2, 4]
    assert index.average_length == 3.0


def test_counts_terms_and_postings() -> None:
    index = build({"d1": "a a b", "d2": "a c", "d3": "c c c c"})

    assert index.vocabulary_size == 3
    assert index.postings_count == 5
    assert {term for term, _ in index.terms()} == {"a", "b", "c"}


def test_counts_term_occurrences_in_the_whole_collection() -> None:
    index = build({"d1": "a a b", "d2": "a c", "d3": "c c c c"})

    assert index.total_length == 9
    assert index.collection_frequency("a") == 3
    assert index.collection_frequency("c") == 5
    assert index.collection_frequency("missing") == 0


def test_an_item_without_terms_is_still_an_item() -> None:
    index = build({"d1": "a", "empty": "", "d3": "a"})

    postings = index.postings("a")

    assert len(index) == 3
    assert index.lengths.tolist() == [1, 0, 1]
    assert postings is not None
    assert postings.positions.tolist() == [0, 2]


def test_an_empty_index_is_valid() -> None:
    index = build({})

    assert len(index) == 0
    assert index.average_length == 0.0
    assert index.vocabulary_size == 0


def test_arrays_are_compact_unsigned_integers() -> None:
    index = build({"d1": "a b"})
    postings = index.postings("a")

    assert postings is not None
    assert postings.positions.dtype == np.uint32
    assert postings.frequencies.dtype == np.uint32
    assert index.lengths.dtype == np.uint32
