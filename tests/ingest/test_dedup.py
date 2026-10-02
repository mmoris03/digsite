from digsite.ingest.dedup import content_hash, find_duplicates, fingerprint
from digsite.ingest.simhash import hamming_distance, shingles, simhash
from digsite.models import Duplicate, DuplicateKind, Fingerprint
from digsite.text import tokenize

ARTICLE = (
    "The city council approved the new budget on Tuesday after a long debate. "
    "The plan raises spending on public transport and cuts funding for road building. "
    "Opposition members said the vote was rushed and promised to challenge it in court. "
    "The mayor defended the decision and said the city could not afford to wait."
)
EDITED = ARTICLE.replace("on Tuesday", "on Wednesday")
UNRELATED = (
    "Researchers described a new species of deep-sea fish found near volcanic vents. "
    "The animal survives crushing pressure and total darkness by feeding on bacteria. "
    "Its discovery suggests that life in the abyss is more diverse than expected."
)


def entry(page_id: int, text: str) -> Fingerprint:
    return Fingerprint(page_id, content_hash(text), fingerprint(text))


def test_content_hash_ignores_whitespace_differences_only() -> None:
    assert content_hash("a  b\n\nc") == content_hash(" a b c ")
    assert content_hash("a b c") != content_hash("A b c")
    assert content_hash("a b c") != content_hash("a b, c")


def test_the_example_texts_behave_as_the_tests_assume() -> None:
    assert hamming_distance(fingerprint(ARTICLE), fingerprint(EDITED)) <= 6
    assert hamming_distance(fingerprint(ARTICLE), fingerprint(UNRELATED)) > 6


def test_distinct_documents_are_not_duplicates() -> None:
    assert find_duplicates([entry(1, ARTICLE), entry(2, UNRELATED)]) == {}


def test_list_pages_that_share_a_repeated_phrase_are_not_duplicates() -> None:
    # Found on a real crawl: alphabetical index pages list different entries, but
    # every line repeats the same few words.
    def index_page(letter: str) -> str:
        return "\n".join(
            f"{letter}_name_{number}() (function in the module)" for number in range(200)
        )

    page_a, page_c = index_page("a"), index_page("c")

    # Counting each occurrence lets the repeated phrase decide the fingerprint...
    by_frequency_a = simhash(shingles(tokenize(page_a), 2))
    by_frequency_c = simhash(shingles(tokenize(page_c), 2))
    assert hamming_distance(by_frequency_a, by_frequency_c) <= 6
    # ...which is why the fingerprint counts each distinct shingle once.
    assert hamming_distance(fingerprint(page_a), fingerprint(page_c)) > 15
    assert find_duplicates([entry(1, page_a), entry(2, page_c)]) == {}


def test_an_identical_copy_is_an_exact_duplicate_of_the_first_one() -> None:
    duplicates = find_duplicates([entry(1, ARTICLE), entry(2, UNRELATED), entry(3, ARTICLE)])

    assert duplicates == {3: Duplicate(1, DuplicateKind.EXACT)}


def test_a_slightly_edited_copy_is_a_near_duplicate() -> None:
    duplicates = find_duplicates([entry(1, ARTICLE), entry(2, EDITED)])

    assert duplicates == {2: Duplicate(1, DuplicateKind.NEAR)}


def test_the_earlier_document_is_the_canonical_one() -> None:
    duplicates = find_duplicates([entry(7, EDITED), entry(9, ARTICLE)])

    assert duplicates == {9: Duplicate(7, DuplicateKind.NEAR)}


def test_duplicates_never_point_at_another_duplicate() -> None:
    # 3 is an exact copy of 2, which is itself a near-duplicate of 1.
    duplicates = find_duplicates([entry(1, ARTICLE), entry(2, EDITED), entry(3, EDITED)])

    assert duplicates == {
        2: Duplicate(1, DuplicateKind.NEAR),
        3: Duplicate(1, DuplicateKind.EXACT),
    }


def test_max_distance_zero_disables_near_duplicate_detection() -> None:
    duplicates = find_duplicates([entry(1, ARTICLE), entry(2, EDITED)], max_distance=0)

    assert duplicates == {}


def test_among_several_matches_the_closest_wins() -> None:
    # 1 and 2 are 10 bits apart, so both are canonical. 3 is 6 bits from 1 and 4 from 2.
    first = Fingerprint(1, "hash-1", 0b00_0000_0000)
    second = Fingerprint(2, "hash-2", 0b11_1111_1111)
    candidate = Fingerprint(3, "hash-3", 0b00_0011_1111)

    duplicates = find_duplicates([first, second, candidate], max_distance=6)

    assert duplicates == {3: Duplicate(2, DuplicateKind.NEAR)}
