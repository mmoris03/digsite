"""Telling a real difference between two systems from the luck of the queries."""

from collections.abc import Sequence

import numpy as np


def paired_randomization_test(
    first: Sequence[float], second: Sequence[float], *, trials: int = 10_000, seed: int = 0
) -> float:
    """Probability of a difference in means this large if the two systems were the same.

    Both systems were scored on the same queries. If neither were better, which
    of the two scores of a query belongs to which system would be arbitrary.
    The test swaps them at random many times and counts how often the mean
    difference is at least as large, in either direction, as the one observed.
    It assumes nothing about how the scores are distributed.

    Args:
        first: One score per query for the first system.
        second: The scores of the second system for the same queries, in the same order.
        trials: Random reassignments to try.
        seed: Seed of the random generator, so that the result can be reproduced.

    Returns:
        The two-sided p-value. It is never exactly 0: the observed assignment
        counts as one of the possible ones.
    """
    if len(first) != len(second):
        raise ValueError("Both systems must be scored on the same queries")
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if differences.size == 0 or not differences.any():
        return 1.0
    observed = abs(differences.sum())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(trials, differences.size))
    # The tolerance keeps rounding from hiding assignments that tie with the observed one.
    extreme = int((np.abs(signs @ differences) >= observed - 1e-12).sum())
    return (extreme + 1) / (trials + 1)
