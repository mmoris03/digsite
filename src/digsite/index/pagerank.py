"""PageRank: how important a node of a graph is, judged by who links to it.

A node is important if important nodes link to it. The score of a node is the
share of time a reader would spend on it by following links at random and, now
and then, jumping to any node at all.
"""

import logging
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DAMPING = 0.85


def pagerank[K: Hashable](
    nodes: Sequence[K],
    edges: Iterable[tuple[K, K]],
    *,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> dict[K, float]:
    """Compute the PageRank of every node by power iteration.

    A node without outgoing links spreads its score evenly over all nodes, as
    if it linked to every one of them. Scores are positive and add up to 1.

    Args:
        nodes: The nodes of the graph, without repetitions.
        edges: (source, target) links between those nodes. A link given twice
            counts twice.
        damping: Probability of following a link instead of jumping to a random
            node. Must be at least 0 and less than 1.
        tolerance: Stop when the scores change by less than this in total
            from one iteration to the next.
        max_iterations: Stop after this many iterations even if still changing.

    Returns:
        The score of each node, in the order of `nodes`.
    """
    if not 0 <= damping < 1:
        raise ValueError("Damping must be at least 0 and less than 1")
    position = {node: number for number, node in enumerate(nodes)}
    if len(position) != len(nodes):
        raise ValueError("Nodes must not be repeated")
    count = len(nodes)
    if count == 0:
        return {}

    try:
        pairs = [(position[source], position[target]) for source, target in edges]
    except KeyError as error:
        raise ValueError(f"An edge mentions {error.args[0]!r}, which is not a node") from None
    sources = np.fromiter((source for source, _ in pairs), dtype=np.intp, count=len(pairs))
    targets = np.fromiter((target for _, target in pairs), dtype=np.intp, count=len(pairs))

    out_degree = np.bincount(sources, minlength=count)
    dangling = out_degree == 0
    # What each link carries is its source's score divided among the source's links.
    share = 1.0 / out_degree[sources]

    rank = np.full(count, 1.0 / count)
    for iteration in range(1, max_iterations + 1):
        received = np.bincount(targets, weights=rank[sources] * share, minlength=count)
        updated = (1 - damping) / count + damping * (received + rank[dangling].sum() / count)
        change = float(np.abs(updated - rank).sum())
        rank = updated
        if change < tolerance:
            logger.debug("PageRank converged in %d iterations", iteration)
            break
    else:
        logger.warning("PageRank did not converge in %d iterations", max_iterations)
    return dict(zip(nodes, rank.tolist(), strict=True))
