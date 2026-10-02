import sqlite3

import pytest

from digsite.store import AuthorityStore, CrawlStore, DocumentStore


@pytest.fixture
def authority_store(connection: sqlite3.Connection) -> AuthorityStore:
    return AuthorityStore(connection)


def add_document(store: CrawlStore, documents: DocumentStore, path: str) -> int:
    page_id = store.save_page(f"https://example.com{path}", 0, b"<p>x</p>")
    documents.save(page_id, path, f"text of {path}", f"hash-{page_id}", page_id)
    return page_id


def test_a_new_corpus_has_no_scores(authority_store: AuthorityStore) -> None:
    assert authority_store.scores() == {}


def test_scores_are_read_back_by_page(
    store: CrawlStore, document_store: DocumentStore, authority_store: AuthorityStore
) -> None:
    first = add_document(store, document_store, "/a")
    second = add_document(store, document_store, "/b")

    authority_store.replace_all({second: 0.75, first: 0.25})

    assert authority_store.scores() == {first: 0.25, second: 0.75}


def test_saving_again_replaces_every_score(
    store: CrawlStore, document_store: DocumentStore, authority_store: AuthorityStore
) -> None:
    first = add_document(store, document_store, "/a")
    second = add_document(store, document_store, "/b")
    authority_store.replace_all({first: 0.5, second: 0.5})

    authority_store.replace_all({second: 1.0})

    assert authority_store.scores() == {second: 1.0}


def test_scores_go_away_with_their_documents(
    store: CrawlStore, document_store: DocumentStore, authority_store: AuthorityStore
) -> None:
    authority_store.replace_all({add_document(store, document_store, "/a"): 1.0})

    document_store.clear()

    assert authority_store.scores() == {}


def test_only_documents_can_be_scored(authority_store: AuthorityStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        authority_store.replace_all({999: 1.0})
