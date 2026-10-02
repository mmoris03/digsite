from digsite.ingest.content import Content
from digsite.ingest.pipeline import ingest
from digsite.models import DocumentStats, DuplicateDocument, DuplicateKind
from digsite.store import CrawlStore, DocumentStore

SITE = "https://example.com"

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


class FakeExtractor:
    """Treats the stored body as the already-extracted text, and counts calls."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, html: str, url: str) -> Content | None:
        self.urls.append(url)
        return Content(title="", text=html) if html.strip() else None


def add_page(
    store: CrawlStore, path: str, text: str, *, title: str = "", noindex: bool = False
) -> None:
    store.save_page(f"{SITE}{path}", 0, text.encode(), title=title, noindex=noindex)


def test_every_stored_page_becomes_a_document(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/a", ARTICLE)
    add_page(store, "/b", UNRELATED)

    stats = ingest(store, document_store, extract=FakeExtractor())

    assert stats.extracted == 2
    assert stats.documents == DocumentStats(
        unique=2, exact_duplicates=0, near_duplicates=0, empty=0
    )
    assert [(doc.url, doc.text) for doc in document_store.documents()] == [
        (f"{SITE}/a", ARTICLE),
        (f"{SITE}/b", UNRELATED),
    ]


def test_marks_exact_and_near_duplicates_of_the_earliest_page(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/", ARTICLE)
    add_page(store, "/other", UNRELATED)
    add_page(store, "/index.html", ARTICLE)
    add_page(store, "/print", EDITED)

    stats = ingest(store, document_store, extract=FakeExtractor())

    assert stats.documents == DocumentStats(
        unique=2, exact_duplicates=1, near_duplicates=1, empty=0
    )
    assert document_store.duplicates() == [
        DuplicateDocument(f"{SITE}/index.html", f"{SITE}/", DuplicateKind.EXACT),
        DuplicateDocument(f"{SITE}/print", f"{SITE}/", DuplicateKind.NEAR),
    ]
    assert [doc.url for doc in document_store.documents()] == [f"{SITE}/", f"{SITE}/other"]


def test_pages_without_content_are_recorded_as_empty(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/blank", "   ")
    add_page(store, "/symbols", "--- ¶ ---")
    add_page(store, "/a", ARTICLE)

    stats = ingest(store, document_store, extract=FakeExtractor())

    # Two empty pages are not duplicates of each other.
    assert stats.documents == DocumentStats(
        unique=1, exact_duplicates=0, near_duplicates=0, empty=2
    )
    assert [doc.url for doc in document_store.documents()] == [f"{SITE}/a"]


def test_noindex_pages_are_left_out(store: CrawlStore, document_store: DocumentStore) -> None:
    add_page(store, "/search", ARTICLE, noindex=True)
    add_page(store, "/a", UNRELATED)
    extractor = FakeExtractor()

    stats = ingest(store, document_store, extract=extractor)

    assert extractor.urls == [f"{SITE}/a"]
    assert stats.extracted == 1
    assert [doc.url for doc in document_store.documents()] == [f"{SITE}/a"]


def test_title_falls_back_to_the_page_title(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/a", ARTICLE, title="Budget approved — City News")
    add_page(store, "/b", UNRELATED, title="Ignored")

    def extract(html: str, url: str) -> Content:
        return Content(title="Deep-sea fish" if url.endswith("/b") else "", text=html)

    ingest(store, document_store, extract=extract)

    assert [doc.title for doc in document_store.documents()] == [
        "Budget approved — City News",
        "Deep-sea fish",
    ]


def test_a_second_run_only_extracts_new_pages_but_redoes_duplicates(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/a", ARTICLE)
    extractor = FakeExtractor()
    ingest(store, document_store, extract=extractor)

    add_page(store, "/copy", ARTICLE)
    stats = ingest(store, document_store, extract=extractor)

    assert extractor.urls == [f"{SITE}/a", f"{SITE}/copy"]
    assert stats.extracted == 1
    assert stats.documents == DocumentStats(
        unique=1, exact_duplicates=1, near_duplicates=0, empty=0
    )


def test_changing_the_distance_needs_no_new_extraction(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    add_page(store, "/a", ARTICLE)
    add_page(store, "/b", EDITED)
    extractor = FakeExtractor()
    ingest(store, document_store, extract=extractor)

    stats = ingest(store, document_store, extract=extractor, max_distance=0)

    assert stats.extracted == 0
    assert len(extractor.urls) == 2
    assert stats.documents == DocumentStats(
        unique=2, exact_duplicates=0, near_duplicates=0, empty=0
    )


def test_force_extracts_everything_again(store: CrawlStore, document_store: DocumentStore) -> None:
    add_page(store, "/a", ARTICLE)
    add_page(store, "/b", UNRELATED)
    extractor = FakeExtractor()
    ingest(store, document_store, extract=extractor)

    stats = ingest(store, document_store, extract=extractor, force=True)

    assert stats.extracted == 2
    assert len(extractor.urls) == 4


def test_decodes_the_body_with_the_charset_of_the_page(
    store: CrawlStore, document_store: DocumentStore
) -> None:
    text = "El año pasado se aprobó una ampliación del presupuesto municipal de transporte."
    store.save_page(
        f"{SITE}/es", 0, text.encode("latin-1"), content_type="text/html; charset=ISO-8859-1"
    )

    ingest(store, document_store, extract=FakeExtractor())

    assert document_store.documents()[0].text == text
