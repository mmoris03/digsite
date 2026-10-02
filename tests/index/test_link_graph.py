from digsite.index.link_graph import document_links
from digsite.models import Edge

DOCUMENTS = {"/a", "/b", "/c"}


def edge(source: str, target: str, *, nofollow: bool = False) -> Edge:
    return Edge(source, target, nofollow)


def test_links_between_documents_are_kept() -> None:
    links = document_links([edge("/b", "/c"), edge("/a", "/b")], DOCUMENTS)

    assert links == [("/a", "/b"), ("/b", "/c")]


def test_links_that_leave_the_corpus_are_dropped() -> None:
    edges = [edge("/a", "https://elsewhere.example/"), edge("/not-indexed", "/a")]

    assert document_links(edges, DOCUMENTS) == []


def test_a_document_linking_to_itself_does_not_count() -> None:
    assert document_links([edge("/a", "/a")], DOCUMENTS) == []


def test_nofollow_links_do_not_count() -> None:
    assert document_links([edge("/a", "/b", nofollow=True)], DOCUMENTS) == []


def test_a_link_to_a_redirect_counts_for_where_it_ends() -> None:
    redirects = {"/old": "/older", "/older": "/b"}

    links = document_links([edge("/a", "/old")], DOCUMENTS, redirects=redirects)

    assert links == [("/a", "/b")]


def test_a_redirect_loop_leads_nowhere() -> None:
    redirects = {"/x": "/y", "/y": "/x"}

    assert document_links([edge("/a", "/x")], DOCUMENTS, redirects=redirects) == []


def test_links_of_a_duplicate_count_for_its_canonical_document() -> None:
    duplicates = {"/a-copy": "/a", "/b-copy": "/b"}
    edges = [edge("/a-copy", "/c"), edge("/c", "/b-copy")]

    links = document_links(edges, DOCUMENTS, duplicates=duplicates)

    assert links == [("/a", "/c"), ("/c", "/b")]


def test_a_link_between_a_document_and_its_duplicate_is_a_self_link() -> None:
    assert document_links([edge("/a", "/a-copy")], DOCUMENTS, duplicates={"/a-copy": "/a"}) == []


def test_a_redirect_may_end_at_a_duplicate() -> None:
    links = document_links(
        [edge("/a", "/old")],
        DOCUMENTS,
        redirects={"/old": "/b-copy"},
        duplicates={"/b-copy": "/b"},
    )

    assert links == [("/a", "/b")]


def test_the_same_link_reached_in_two_ways_counts_once() -> None:
    edges = [edge("/a", "/b"), edge("/a", "/b-copy"), edge("/a-copy", "/b")]
    duplicates = {"/a-copy": "/a", "/b-copy": "/b"}

    assert document_links(edges, DOCUMENTS, duplicates=duplicates) == [("/a", "/b")]
