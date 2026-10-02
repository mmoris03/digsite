"""Duplicate detection: exact copies by hash, near-duplicates by SimHash."""

import hashlib
from collections.abc import Iterable

from digsite.ingest.simhash import shingles, simhash
from digsite.ingest.simhash_index import SimHashIndex
from digsite.models import Duplicate, DuplicateKind, Fingerprint
from digsite.text import tokenize

# Chosen by evaluation on labelled near-duplicate collections; see docs/decisions.md.
SIMHASH_BITS = 64
SHINGLE_SIZE = 2
DEFAULT_MAX_DISTANCE = 6


def content_hash(text: str) -> str:
    """Hash a text so that copies differing only in whitespace hash the same."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def fingerprint(text: str, *, bits: int = SIMHASH_BITS, shingle_size: int = SHINGLE_SIZE) -> int:
    """Compute the SimHash fingerprint of a text from its distinct word shingles.

    Each distinct shingle counts once, however often it occurs. Weighting by
    frequency makes list-like pages look alike: a phrase repeated on every line
    ("in module", "class in") outvotes the entries that tell the pages apart.
    """
    return simhash(set(shingles(tokenize(text), shingle_size)), bits)


def find_duplicates(
    fingerprints: Iterable[Fingerprint], *, max_distance: int = DEFAULT_MAX_DISTANCE
) -> dict[int, Duplicate]:
    """Decide which documents repeat an earlier one.

    Documents are taken in the given order. The first document with some
    content becomes canonical; a later one is a duplicate if its hash equals a
    canonical document's (exact) or its fingerprint is within `max_distance`
    bits of one (near). A duplicate always points at a canonical document,
    never at another duplicate.

    Args:
        fingerprints: One entry per document, in order of preference.
        max_distance: Largest Hamming distance between near-duplicates.

    Returns:
        The duplicates found, keyed by page id.
    """
    canonical_by_hash: dict[str, int] = {}
    index: SimHashIndex[int] = SimHashIndex(SIMHASH_BITS, max_distance)
    duplicates: dict[int, Duplicate] = {}

    for item in fingerprints:
        canonical_id = canonical_by_hash.get(item.content_hash)
        if canonical_id is not None:
            duplicates[item.page_id] = Duplicate(canonical_id, DuplicateKind.EXACT)
            continue

        matches = index.near(item.simhash)
        if matches:
            canonical_id = matches[0][0]
            duplicates[item.page_id] = Duplicate(canonical_id, DuplicateKind.NEAR)
            # An exact copy of this near-duplicate must point at the canonical one too.
            canonical_by_hash[item.content_hash] = canonical_id
            continue

        canonical_by_hash[item.content_hash] = item.page_id
        index.add(item.page_id, item.simhash)
    return duplicates
