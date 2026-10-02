import pytest

from digsite.search.fusion import reciprocal_rank_fusion
from digsite.search.retriever import Hit


def test_an_item_both_rankings_like_beats_the_favourite_of_one() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])

    assert fused == [
        Hit("b", pytest.approx(1 / 62 + 1 / 61)),
        Hit("a", pytest.approx(1 / 61 + 1 / 63)),
        Hit("d", pytest.approx(1 / 62)),
        Hit("c", pytest.approx(1 / 63)),
    ]


def test_a_single_ranking_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"]])

    assert [hit.key for hit in fused] == ["x", "y", "z"]


def test_ties_keep_the_order_in_which_items_were_first_seen() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])

    assert [hit.key for hit in fused] == ["a", "c", "b", "d"]


def test_a_heavier_ranking_has_more_say() -> None:
    rankings = [["a", "b"], ["b", "a"]]

    assert [hit.key for hit in reciprocal_rank_fusion(rankings, weights=[2, 1])] == ["a", "b"]
    assert [hit.key for hit in reciprocal_rank_fusion(rankings, weights=[1, 2])] == ["b", "a"]


def test_a_ranking_with_no_weight_is_ignored_but_its_items_are_listed() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]], weights=[1, 0])

    assert fused == [
        Hit("a", pytest.approx(1 / 61)),
        Hit("b", pytest.approx(1 / 62)),
        Hit("c", 0.0),
    ]


def test_a_smaller_constant_rewards_first_places_more() -> None:
    # `a` is first once and fourth once; `b` is second in both rankings.
    rankings = [["a", "b", "c", "d"], ["e", "b", "f", "a"]]

    assert reciprocal_rank_fusion(rankings, rank_constant=60)[0].key == "b"
    assert reciprocal_rank_fusion(rankings, rank_constant=0)[0].key == "a"


def test_an_item_repeated_in_a_ranking_counts_at_its_first_position() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]])

    assert fused == [Hit("a", pytest.approx(1 / 61)), Hit("b", pytest.approx(1 / 62))]


def test_nothing_to_fuse_gives_nothing() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_must_match_the_rankings() -> None:
    with pytest.raises(ValueError, match="one weight per ranking"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_the_rank_constant_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="rank constant"):
        reciprocal_rank_fusion([["a"]], rank_constant=-1)
