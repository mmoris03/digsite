from collections import Counter

import pytest

from digsite.index.inverted_index import IndexBuilder, InvertedIndex
from digsite.search.expansion import (
    ExpansionSettings,
    expansion_terms,
    log_likelihood_ratio,
    signed_root_llr,
)


def build(items: dict[str, str]) -> InvertedIndex[str]:
    builder: IndexBuilder[str] = IndexBuilder()
    for key, text in items.items():
        builder.add(key, text.split())
    return builder.build()


def test_llr_matches_a_reference_value() -> None:
    # Produced by an independent port of Apache Mahout's implementation.
    assert log_likelihood_ratio(100, 5, 900, 4995) == pytest.approx(328.47961215037503)


def test_llr_is_zero_when_the_event_is_equally_likely_in_both_groups() -> None:
    assert log_likelihood_ratio(500, 2500, 500, 2500) == 0.0
    assert log_likelihood_ratio(1, 10, 99, 990) == pytest.approx(0.0, abs=1e-9)


def test_llr_handles_empty_cells() -> None:
    assert log_likelihood_ratio(0, 0, 0, 0) == 0.0
    assert log_likelihood_ratio(5, 0, 0, 5) == pytest.approx(2 * 10 * 0.6931471805599453)


def test_signed_root_is_positive_when_the_event_favours_the_first_group() -> None:
    assert signed_root_llr(100, 5, 900, 4995) == pytest.approx(328.47961215037503**0.5)


def test_signed_root_is_negative_when_the_event_favours_the_second_group() -> None:
    assert signed_root_llr(5, 100, 4995, 900) == pytest.approx(-(328.47961215037503**0.5))


def test_signed_root_is_zero_without_association() -> None:
    assert signed_root_llr(500, 2500, 500, 2500) == 0.0


# "cache" and "lru" are concentrated in the first two documents; "the" is everywhere.
INDEX = build(
    {
        "d1": "the cache evicts with lru the cache",
        "d2": "the lru cache is the fast cache",
        "d3": "the log rotates the log daily",
        "d4": "the package installs the service",
        "d5": "the queue retries the message",
        "d6": "the daily report lists the message",
    }
)
FEEDBACK = [
    Counter(["the", "cache", "evicts", "with", "lru", "the", "cache"]),
    Counter(["the", "lru", "cache", "is", "the", "fast", "cache"]),
]


def terms(exclude: frozenset[str] = frozenset(), **settings: int) -> list[str]:
    found = expansion_terms(INDEX, FEEDBACK, exclude, ExpansionSettings(**settings))
    return [term for term, _ in found]


def test_picks_the_terms_that_set_the_feedback_documents_apart() -> None:
    assert terms(terms=2) == ["cache", "lru"]


def test_a_word_that_is_everywhere_is_not_characteristic() -> None:
    assert "the" not in terms(terms=20)


def test_terms_of_the_query_are_never_returned() -> None:
    assert terms(frozenset({"cache"}), terms=1) == ["lru"]


def test_results_come_with_a_positive_score_strongest_first() -> None:
    found = expansion_terms(INDEX, FEEDBACK, set(), ExpansionSettings(terms=20))
    scores = [score for _, score in found]

    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_equally_strong_terms_come_in_alphabetical_order() -> None:
    # Each occurs once, in one feedback document, and nowhere else.
    found = dict(expansion_terms(INDEX, FEEDBACK, set(), ExpansionSettings(terms=20)))

    assert found["evicts"] == found["fast"] == found["is"] == found["with"]
    assert terms(terms=20)[2:] == ["evicts", "fast", "is", "with"]


def test_a_term_can_be_required_in_several_feedback_documents() -> None:
    assert terms(terms=20, min_feedback_documents=2) == ["cache", "lru"]


def test_no_feedback_gives_no_terms() -> None:
    assert expansion_terms(INDEX, [], set(), ExpansionSettings()) == []
