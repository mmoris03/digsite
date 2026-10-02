import pytest

from digsite.ingest.simhash import hamming_distance, shingles, simhash
from digsite.text import tokenize

ARTICLE = (
    "The city council approved the new budget on Tuesday after a long debate. "
    "The plan raises spending on public transport and cuts funding for road building. "
    "Opposition members said the vote was rushed and promised to challenge it in court. "
    "The mayor defended the decision and said the city could not afford to wait."
)


def fingerprint(text: str, size: int = 2) -> int:
    return simhash(shingles(tokenize(text), size))


def test_shingles_are_runs_of_consecutive_tokens() -> None:
    tokens = ["a", "b", "c", "d"]

    assert list(shingles(tokens, 1)) == ["a", "b", "c", "d"]
    assert list(shingles(tokens, 2)) == ["a b", "b c", "c d"]
    assert list(shingles(tokens, 4)) == ["a b c d"]


def test_a_text_shorter_than_the_shingle_size_is_one_shingle() -> None:
    assert list(shingles(["a", "b"], 3)) == ["a b"]
    assert list(shingles([], 3)) == []


def test_shingle_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(shingles(["a"], 0))


def test_the_same_text_always_gets_the_same_fingerprint() -> None:
    # Pinned value: fingerprints are stored, so the hash must not change between releases.
    assert simhash(["alpha", "beta", "gamma"]) == simhash(["gamma", "alpha", "beta"])
    assert simhash(["alpha", "beta", "gamma"]) == 0x53465888AE1B08BE


def test_fingerprint_fits_in_the_requested_number_of_bits() -> None:
    features = list(shingles(tokenize(ARTICLE), 2))

    assert 0 < simhash(features, bits=64) < 2**64
    assert 2**64 < simhash(features, bits=128) < 2**128
    assert simhash(features, bits=8) < 2**8


def test_no_features_gives_fingerprint_zero() -> None:
    assert simhash([]) == 0


@pytest.mark.parametrize("bits", [0, -8, 12, 65])
def test_fingerprint_length_must_be_a_positive_multiple_of_eight(bits: int) -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        simhash(["a"], bits=bits)


def test_similar_texts_get_close_fingerprints() -> None:
    edited = ARTICLE.replace("on Tuesday", "on Wednesday").replace("long", "lengthy")
    unrelated = (
        "Researchers described a new species of deep-sea fish found near volcanic vents. "
        "The animal survives crushing pressure and total darkness by feeding on bacteria. "
        "Its discovery suggests that life in the abyss is more diverse than expected. "
        "The team plans another expedition to map the area in more detail next year."
    )

    near = hamming_distance(fingerprint(ARTICLE), fingerprint(edited))
    far = hamming_distance(fingerprint(ARTICLE), fingerprint(unrelated))

    assert near <= 6
    assert far > 20


def test_frequent_features_weigh_more() -> None:
    once = simhash(["common", "rare"])
    dominated = simhash(["common"] * 50 + ["rare"])

    assert dominated == simhash(["common"])
    assert once != dominated


def test_hamming_distance_counts_differing_bits() -> None:
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0b1010, 0b0110) == 2
    assert hamming_distance(0, 2**64 - 1) == 64
