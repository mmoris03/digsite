import sqlite3

import pytest

from digsite.models import (
    Document,
    DocumentStats,
    Duplicate,
    DuplicateDocument,
    DuplicateKind,
    Fingerprint,
)
from digsite.store import CrawlStore, DocumentStore

SITE = "https://example.com"


@pytest.fixture
def page_ids(store: CrawlStore) -> list[int]:
    return [store.save_page(f"{SITE}/{name}", 0, b"<p>x</p>") for name in ("a", "b", "c", "d")]


def test_documents_round_trip(document_store: DocumentStore, page_ids: list[int]) -> None:
    document_store.save(page_ids[0], "Title A", "# A\n\nText of a.", "hash-a", 0xABCDEF)
    document_store.save(page_ids[1], "Title B", "Text of b.", "hash-b", 2**64 - 1)

    assert document_store.documents() == [
        Document(page_ids[0], f"{SITE}/a", "Title A", "# A\n\nText of a."),
        Document(page_ids[1], f"{SITE}/b", "Title B", "Text of b."),
    ]
    assert document_store.page_ids() == {page_ids[0], page_ids[1]}


def test_fingerprints_survive_storage_at_any_size(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    values = [0, 2**64 - 1, 2**127 + 5]
    for page_id, value in zip(page_ids, values, strict=False):
        document_store.save(page_id, "", "text", f"hash-{page_id}", value)

    assert document_store.fingerprints() == [
        Fingerprint(page_id, f"hash-{page_id}", value)
        for page_id, value in zip(page_ids, values, strict=False)
    ]


def test_empty_documents_are_recorded_but_not_listed(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    document_store.save(page_ids[0], "Empty", "", "hash-empty", None)

    assert document_store.page_ids() == {page_ids[0]}
    assert document_store.documents() == []
    assert document_store.fingerprints() == []
    assert document_store.stats() == DocumentStats(
        unique=0, exact_duplicates=0, near_duplicates=0, empty=1
    )


def test_a_document_with_text_must_have_a_fingerprint(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        document_store.save(page_ids[0], "", "text", "hash", None)
    with pytest.raises(sqlite3.IntegrityError):
        document_store.save(page_ids[0], "", "", "hash", 123)


def test_a_document_needs_an_existing_page(document_store: DocumentStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        document_store.save(999, "", "text", "hash", 1)


def test_saving_again_replaces_the_document(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    document_store.save(page_ids[0], "Old", "old text", "hash-old", 1)

    document_store.save(page_ids[0], "New", "new text", "hash-new", 2)

    assert document_store.documents() == [Document(page_ids[0], f"{SITE}/a", "New", "new text")]


def test_duplicates_are_hidden_from_documents_and_counted(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    for page_id in page_ids:
        document_store.save(page_id, "", f"text {page_id}", f"hash-{page_id}", page_id)

    document_store.set_duplicates(
        {
            page_ids[1]: Duplicate(page_ids[0], DuplicateKind.EXACT),
            page_ids[3]: Duplicate(page_ids[0], DuplicateKind.NEAR),
        }
    )

    assert [doc.url for doc in document_store.documents()] == [f"{SITE}/a", f"{SITE}/c"]
    assert document_store.duplicates() == [
        DuplicateDocument(f"{SITE}/b", f"{SITE}/a", DuplicateKind.EXACT),
        DuplicateDocument(f"{SITE}/d", f"{SITE}/a", DuplicateKind.NEAR),
    ]
    assert document_store.stats() == DocumentStats(
        unique=2, exact_duplicates=1, near_duplicates=1, empty=0
    )


def test_setting_duplicates_replaces_the_previous_marks(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    for page_id in page_ids:
        document_store.save(page_id, "", f"text {page_id}", f"hash-{page_id}", page_id)
    document_store.set_duplicates({page_ids[1]: Duplicate(page_ids[0], DuplicateKind.EXACT)})

    document_store.set_duplicates({page_ids[2]: Duplicate(page_ids[0], DuplicateKind.NEAR)})

    assert document_store.duplicates() == [
        DuplicateDocument(f"{SITE}/c", f"{SITE}/a", DuplicateKind.NEAR)
    ]


def test_clear_removes_every_document_even_with_duplicate_marks(
    document_store: DocumentStore, page_ids: list[int]
) -> None:
    for page_id in page_ids:
        document_store.save(page_id, "", f"text {page_id}", f"hash-{page_id}", page_id)
    document_store.set_duplicates({page_ids[1]: Duplicate(page_ids[0], DuplicateKind.EXACT)})

    document_store.clear()

    assert document_store.page_ids() == set()
    assert document_store.stats() == DocumentStats(
        unique=0, exact_duplicates=0, near_duplicates=0, empty=0
    )
