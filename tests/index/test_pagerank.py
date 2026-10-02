import random

import pytest

from digsite.index.pagerank import pagerank


def test_a_cycle_gives_every_node_the_same_score() -> None:
    scores = pagerank(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])

    assert scores == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_the_node_everyone_links_to_scores_highest() -> None:
    edges = [("a", "hub"), ("b", "hub"), ("c", "hub"), ("hub", "a")]

    scores = pagerank(["a", "b", "c", "hub"], edges)

    assert max(scores, key=scores.__getitem__) == "hub"
    # `a` is the only page the hub links to, so it inherits the hub's importance.
    assert scores["a"] > scores["b"] == pytest.approx(scores["c"])


def test_a_link_from_an_important_page_is_worth_more() -> None:
    # `x` and `y` both receive one link: `x` from the hub, `y` from a page nobody links to.
    edges = [("a", "hub"), ("b", "hub"), ("hub", "x"), ("lonely", "y")]

    scores = pagerank(["a", "b", "hub", "x", "lonely", "y"], edges)

    assert scores["x"] > scores["y"]


def test_a_page_without_links_shares_its_score_with_everyone() -> None:
    # Solved by hand for damping 0.85: a = 0.075 + 0.425 b, and a + b = 1.
    scores = pagerank(["a", "b"], [("a", "b")])

    assert scores == pytest.approx({"a": 20 / 57, "b": 37 / 57})


def test_scores_add_up_to_one_with_dangling_and_isolated_nodes() -> None:
    scores = pagerank(["a", "b", "c", "isolated"], [("a", "b"), ("a", "c")])

    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(score > 0 for score in scores.values())
    assert scores["b"] == pytest.approx(scores["c"])


def test_without_edges_every_node_scores_the_same() -> None:
    assert pagerank(["a", "b"], []) == pytest.approx({"a": 0.5, "b": 0.5})


def test_with_no_damping_links_do_not_matter() -> None:
    scores = pagerank(["a", "b", "c"], [("a", "b"), ("c", "b")], damping=0.0)

    assert scores == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_scores_come_in_the_order_of_the_nodes() -> None:
    assert list(pagerank([3, 1, 2], [(1, 2)])) == [3, 1, 2]


def test_an_empty_graph_has_no_scores() -> None:
    assert pagerank([], []) == {}


def test_a_link_given_twice_counts_twice() -> None:
    once = pagerank(["a", "b", "c"], [("a", "b"), ("a", "c")])
    twice = pagerank(["a", "b", "c"], [("a", "b"), ("a", "b"), ("a", "c")])

    assert once["b"] == pytest.approx(once["c"])
    assert twice["b"] > twice["c"]


def test_an_edge_to_an_unknown_node_is_an_error() -> None:
    with pytest.raises(ValueError, match="'z', which is not a node"):
        pagerank(["a"], [("a", "z")])


def test_repeated_nodes_are_an_error() -> None:
    with pytest.raises(ValueError, match="must not be repeated"):
        pagerank(["a", "a"], [])


@pytest.mark.parametrize("damping", [-0.1, 1.0])
def test_damping_out_of_range_is_an_error(damping: float) -> None:
    with pytest.raises(ValueError, match="Damping"):
        pagerank(["a"], [], damping=damping)


def test_stopping_early_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    pagerank(["a", "b", "c"], [("a", "b"), ("b", "c")], max_iterations=1)

    assert "did not converge in 1 iterations" in caplog.text


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_networkx_on_random_graphs(seed: int) -> None:
    networkx = pytest.importorskip("networkx")
    rng = random.Random(seed)
    nodes = list(range(60))
    # Sparse enough that some nodes have no outgoing links and some no incoming ones.
    edges = list({(rng.choice(nodes), rng.choice(nodes)) for _ in range(rng.randint(40, 400))})
    graph = networkx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    expected = networkx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)

    assert pagerank(nodes, edges) == pytest.approx(expected, abs=1e-9)
