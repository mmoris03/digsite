"""SimHash fingerprints for near-duplicate detection (Charikar, 2002).

A fingerprint is a fixed number of bits computed so that similar texts get
fingerprints that differ in few bits. Comparing two documents then costs one
XOR instead of a comparison of their full texts.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence


def shingles(tokens: Sequence[str], size: int) -> Iterator[str]:
    """Yield every run of `size` consecutive tokens, joined by a space.

    A text shorter than `size` tokens yields itself as a single shingle, so
    that short texts still have a feature to hash.
    """
    if size < 1:
        raise ValueError("Shingle size must be at least 1")
    if not tokens:
        return
    if len(tokens) <= size:
        yield " ".join(tokens)
        return
    for start in range(len(tokens) - size + 1):
        yield " ".join(tokens[start : start + size])


def simhash(features: Iterable[str], bits: int = 64) -> int:
    """Compute the SimHash fingerprint of a bag of features.

    Every feature is hashed to `bits` bits and votes on each bit position: +w
    where its hash has a 1 and -w where it has a 0, with w the number of times
    the feature occurs. The fingerprint has a 1 wherever the votes add up to a
    positive number.

    Args:
        features: The features of the text, typically its word shingles.
        bits: Fingerprint length; a multiple of 8.

    Returns:
        The fingerprint as a non-negative integer below 2**bits. A text with no
        features has fingerprint 0.
    """
    if bits <= 0 or bits % 8:
        raise ValueError("Fingerprint length must be a positive multiple of 8")

    votes = [0] * bits
    for feature, weight in Counter(features).items():
        hashed = _hash(feature, bits)
        for position in range(bits):
            votes[position] += weight if (hashed >> position) & 1 else -weight
    return sum(1 << position for position, vote in enumerate(votes) if vote > 0)


def hamming_distance(a: int, b: int) -> int:
    """Count the bit positions in which two fingerprints differ."""
    return (a ^ b).bit_count()


def _hash(feature: str, bits: int) -> int:
    digest = hashlib.blake2b(feature.encode(), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "big")
