"""Evaluation of near-duplicate detection against labelled duplicate pairs."""

from collections.abc import Iterable, Mapping
from pathlib import Path

from digsite.evaluation.metrics import SetMetrics, compare_sets
from digsite.ingest.dedup import fingerprint
from digsite.ingest.simhash_index import SimHashIndex

type Pair = tuple[str, str]


def load_texts(path: Path) -> dict[str, str]:
    """Read a collection with one `id<separator>text` entry per line.

    The separator is a tab if the line has one, otherwise the first run of
    whitespace. Lines without both an id and a text are ignored.
    """
    texts = {}
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.split("\t", 1) if "\t" in line else line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                texts[parts[0].strip()] = parts[1].strip()
    return texts


def load_pairs(path: Path) -> set[Pair]:
    """Read labelled duplicates, one whitespace-separated pair of ids per line."""
    pairs = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            ids = line.split()
            if len(ids) >= 2:
                pairs.add(_pair(ids[0], ids[1]))
    return pairs


def fingerprint_texts(texts: Mapping[str, str], *, bits: int, shingle_size: int) -> dict[str, int]:
    """Fingerprint every text exactly as the ingest stage does, with the given settings."""
    return {
        text_id: fingerprint(text, bits=bits, shingle_size=shingle_size)
        for text_id, text in texts.items()
    }


def predict_pairs(fingerprints: Mapping[str, int], *, bits: int, max_distance: int) -> set[Pair]:
    """Return every pair of ids whose fingerprints are within `max_distance` bits."""
    index: SimHashIndex[str] = SimHashIndex(bits, max_distance)
    pairs = set()
    for text_id, value in fingerprints.items():
        for other_id, _ in index.near(value):
            pairs.add(_pair(text_id, other_id))
        index.add(text_id, value)
    return pairs


def evaluate(
    texts: Mapping[str, str],
    truth: set[Pair],
    *,
    bits: int,
    shingle_size: int,
    distances: Iterable[int],
) -> dict[int, SetMetrics]:
    """Score near-duplicate detection at several distance thresholds.

    Returns:
        The metrics for each maximum Hamming distance in `distances`.
    """
    fingerprints = fingerprint_texts(texts, bits=bits, shingle_size=shingle_size)
    return {
        distance: compare_sets(truth, predict_pairs(fingerprints, bits=bits, max_distance=distance))
        for distance in distances
    }


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)
