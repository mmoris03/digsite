import random

import pytest

from digsite.ingest.simhash import hamming_distance
from digsite.ingest.simhash_index import SimHashIndex


def flip_bits(fingerprint: int, positions: list[int]) -> int:
    for position in positions:
        fingerprint ^= 1 << position
    return fingerprint


def test_finds_fingerprints_within_the_maximum_distance() -> None:
    index: SimHashIndex[str] = SimHashIndex(bits=64, max_distance=3)
    base = 0x0123456789ABCDEF
    index.add("same", base)
    index.add("three-bits-off", flip_bits(base, [0, 31, 63]))
    index.add("four-bits-off", flip_bits(base, [0, 20, 40, 60]))

    assert index.near(base) == [("same", 0), ("three-bits-off", 3)]


def test_results_are_sorted_by_distance_then_insertion_order() -> None:
    index: SimHashIndex[str] = SimHashIndex(bits=64, max_distance=3)
    base = 0xFFFF0000FFFF0000
    index.add("two", flip_bits(base, [1, 2]))
    index.add("zero", base)
    index.add("also-two", flip_bits(base, [50, 51]))
    index.add("one", flip_bits(base, [7]))

    assert index.near(base) == [("zero", 0), ("one", 1), ("two", 2), ("also-two", 2)]


def test_distance_zero_only_matches_identical_fingerprints() -> None:
    index: SimHashIndex[int] = SimHashIndex(bits=64, max_distance=0)
    index.add(1, 42)
    index.add(2, 43)

    assert index.near(42) == [(1, 0)]


def test_an_empty_index_matches_nothing() -> None:
    assert SimHashIndex[str](bits=64, max_distance=3).near(123) == []


@pytest.mark.parametrize(
    ("bits", "max_distance"),
    [(64, 0), (64, 3), (64, 6), (64, 10), (128, 6), (16, 5), (8, 7)],
)
def test_agrees_with_brute_force(bits: int, max_distance: int) -> None:
    # The block tables are only an optimisation: results must equal a full scan.
    rng = random.Random(bits * 1000 + max_distance)
    stored = {}
    for key in range(300):
        if key % 3 == 0 or not stored:
            fingerprint = rng.getrandbits(bits)
        else:
            # Derive most entries from earlier ones so that near matches exist.
            source = stored[rng.randrange(key)]
            flips = rng.sample(range(bits), rng.randint(0, min(bits, max_distance + 3)))
            fingerprint = flip_bits(source, flips)
        stored[key] = fingerprint

    index: SimHashIndex[int] = SimHashIndex(bits, max_distance)
    for key, fingerprint in stored.items():
        index.add(key, fingerprint)

    for query in list(stored.values())[:60]:
        expected = {
            key: hamming_distance(query, fingerprint)
            for key, fingerprint in stored.items()
            if hamming_distance(query, fingerprint) <= max_distance
        }
        assert dict(index.near(query)) == expected


def test_rejects_distances_that_make_no_sense() -> None:
    with pytest.raises(ValueError, match="negative"):
        SimHashIndex[str](bits=64, max_distance=-1)
    with pytest.raises(ValueError, match="smaller than the fingerprint length"):
        SimHashIndex[str](bits=8, max_distance=8)
