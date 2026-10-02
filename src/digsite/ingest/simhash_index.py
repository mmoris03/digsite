"""Lookup of fingerprints within a small Hamming distance (Manku et al., 2007)."""

from collections import defaultdict

from digsite.ingest.simhash import hamming_distance


class SimHashIndex[K]:
    """Finds stored fingerprints that differ from a query in at most a few bits.

    Comparing a query with every stored fingerprint costs O(n). This index
    avoids it with the pigeonhole principle: split the fingerprint into
    `max_distance + 1` blocks; two fingerprints that differ in at most
    `max_distance` bits must agree exactly on at least one block. So it keeps
    one hash table per block and only compares the query with the fingerprints
    that share a block with it.

    Args:
        bits: Fingerprint length.
        max_distance: Largest Hamming distance that still counts as a match.
    """

    def __init__(self, bits: int = 64, max_distance: int = 3) -> None:
        if max_distance < 0:
            raise ValueError("Maximum distance cannot be negative")
        if max_distance >= bits:
            raise ValueError("Maximum distance must be smaller than the fingerprint length")
        self._max_distance = max_distance
        self._blocks = _split(bits, max_distance + 1)
        # Each table maps a block value to the positions of the entries that have it.
        self._tables: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in self._blocks]
        self._entries: list[tuple[K, int]] = []

    def add(self, key: K, fingerprint: int) -> None:
        """Store a fingerprint under a key."""
        position = len(self._entries)
        self._entries.append((key, fingerprint))
        for table, block in zip(self._tables, self._block_values(fingerprint), strict=True):
            table[block].append(position)

    def near(self, fingerprint: int) -> list[tuple[K, int]]:
        """Return the stored (key, distance) pairs within the maximum distance.

        The result is sorted by distance; entries at the same distance keep the
        order in which they were added.
        """
        candidates: set[int] = set()
        for table, block in zip(self._tables, self._block_values(fingerprint), strict=True):
            candidates.update(table.get(block, ()))

        matches = []
        for position in candidates:
            key, candidate = self._entries[position]
            distance = hamming_distance(fingerprint, candidate)
            if distance <= self._max_distance:
                matches.append((distance, position, key))
        matches.sort(key=lambda match: match[:2])
        return [(key, distance) for distance, _, key in matches]

    def _block_values(self, fingerprint: int) -> list[int]:
        return [(fingerprint >> shift) & mask for shift, mask in self._blocks]


def _split(bits: int, count: int) -> list[tuple[int, int]]:
    """Divide `bits` positions into `count` contiguous blocks, as (shift, mask) pairs."""
    blocks = []
    shift = 0
    for index in range(count):
        # Spread the remainder over the first blocks so widths differ by at most one.
        width = bits // count + (1 if index < bits % count else 0)
        blocks.append((shift, (1 << width) - 1))
        shift += width
    return blocks
