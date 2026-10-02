import pytest

from digsite.index.inverted_index import IndexBuilder, InvertedIndex
from digsite.search.bm25 import Bm25Params, Bm25Ranker, Bm25Variant


def build(items: dict[str, str]) -> InvertedIndex[str]:
    builder: IndexBuilder[str] = IndexBuilder()
    for key, text in items.items():
        builder.add(key, text.split())
    return builder.build()


# Three items, average length 3. The term "a" is in d1 (twice) and d2 (once).
# Length normalisation with b=0.75: d1 -> 1.0, d2 -> 0.75, d3 -> 1.25.
INDEX = build({"d1": "a a b", "d2": "a c", "d3": "c c c c"})


def rank(terms: list[str], limit: int = 10, **params: object) -> list[tuple[str, float]]:
    return Bm25Ranker(INDEX, Bm25Params(**params)).rank(terms, limit)  # type: ignore[arg-type]


def test_scores_match_the_formula_worked_out_by_hand() -> None:
    # idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln 1.6
    # d1: 2 / (2 + 1.2 * 1.0) = 0.625     d2: 1 / (1 + 1.2 * 0.75) = 0.526316
    results = rank(["a"])

    assert [key for key, _ in results] == ["d1", "d2"]
    assert results[0][1] == pytest.approx(0.293752, abs=1e-6)
    assert results[1][1] == pytest.approx(0.247370, abs=1e-6)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (Bm25Variant.ATIRE, [("d1", 0.557514), ("d2", 0.469485)]),
        (Bm25Variant.BM25L, [("d1", 0.698654), ("d2", 0.624950)]),
        (Bm25Variant.BM25_PLUS, [("d1", 1.646225), ("d2", 1.495739)]),
        # "a" is in most items, so Robertson's IDF is negative and the order flips.
        (Bm25Variant.ROBERTSON, [("d2", -0.268856), ("d1", -0.319266)]),
    ],
)
def test_variants_match_their_formulas(
    variant: Bm25Variant, expected: list[tuple[str, float]]
) -> None:
    results = rank(["a"], variant=variant)

    assert [key for key, _ in results] == [key for key, _ in expected]
    assert [score for _, score in results] == pytest.approx(
        [score for _, score in expected], abs=1e-6
    )


def test_scores_of_several_terms_add_up() -> None:
    both = dict(rank(["a", "b"]))

    assert both["d1"] == pytest.approx(dict(rank(["a"]))["d1"] + dict(rank(["b"]))["d1"])
    assert both["d2"] == pytest.approx(dict(rank(["a"]))["d2"])


def test_a_rare_term_outweighs_a_common_one() -> None:
    # "b" is only in d1; "c" is in d2 and d3.
    results = dict(rank(["b", "c"]))

    assert results["d1"] > results["d2"]


def test_repeating_a_query_term_changes_nothing() -> None:
    assert rank(["a", "a", "a"]) == rank(["a"])


def test_a_term_weight_multiplies_its_contribution() -> None:
    ranker = Bm25Ranker(INDEX)
    plain = dict(ranker.rank(["a", "b"]))
    only_a = dict(ranker.rank(["a"]))
    only_b = dict(ranker.rank(["b"]))

    weighted = dict(ranker.rank({"a": 1.0, "b": 0.25}))

    assert weighted["d1"] == pytest.approx(only_a["d1"] + 0.25 * only_b["d1"])
    assert weighted["d2"] == pytest.approx(only_a["d2"])
    assert dict(ranker.rank({"a": 1.0, "b": 1.0})) == pytest.approx(plain)


def test_weights_can_change_the_order() -> None:
    ranker = Bm25Ranker(INDEX)

    # "b" is only in d1 and "c" is in d2 and d3.
    assert ranker.rank({"b": 1.0, "c": 0.1})[0][0] == "d1"
    assert ranker.rank({"b": 0.1, "c": 1.0})[0][0] == "d3"


def test_only_items_that_contain_a_query_term_are_returned() -> None:
    assert [key for key, _ in rank(["b"])] == ["d1"]
    assert rank(["missing"]) == []
    assert rank([]) == []


def test_limit_keeps_the_best_results() -> None:
    assert [key for key, _ in rank(["a", "c"], limit=1)] == [rank(["a", "c"])[0][0]]
    assert len(rank(["a", "c"], limit=2)) == 2
    assert rank(["a"], limit=0) == []


def test_ties_are_broken_by_index_order() -> None:
    index = build({"z": "x y", "m": "x y", "a": "x y", "other": "q"})
    ranker = Bm25Ranker(index)

    assert [key for key, _ in ranker.rank(["x"])] == ["z", "m", "a"]
    assert [key for key, _ in ranker.rank(["x"], limit=2)] == ["z", "m"]


def test_longer_items_score_lower_for_the_same_frequency() -> None:
    index = build({"short": "x y", "long": "x y y y y y y y"})
    scores = dict(Bm25Ranker(index).rank(["x"]))

    assert scores["short"] > scores["long"]


def test_b_zero_ignores_length() -> None:
    index = build({"short": "x y", "long": "x y y y y y y y"})
    scores = dict(Bm25Ranker(index, Bm25Params(b=0.0)).rank(["x"]))

    assert scores["short"] == pytest.approx(scores["long"])


def test_more_occurrences_help_less_and_less() -> None:
    index = build({"one": "x a a a", "two": "x x a a", "four": "x x x x"})
    scores = dict(Bm25Ranker(index).rank(["x"]))

    assert scores["one"] < scores["two"] < scores["four"]
    assert scores["two"] - scores["one"] > scores["four"] - scores["two"]


def test_an_empty_index_returns_nothing() -> None:
    assert Bm25Ranker(build({})).rank(["a"]) == []
