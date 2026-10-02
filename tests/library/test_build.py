from contextlib import closing
from pathlib import Path

import pytest

from digsite.embedding import HashingEmbedder
from digsite.index.analyzer import Language
from digsite.library import BuildError, BuildRequest, BuildStage, build_collection, crawl_scope
from digsite.models import CollectionInfo
from digsite.search.corpus import CorpusSearch, SearchMode
from digsite.store import CollectionStore, connect
from fakes import FakeSite
from library.sites import GUIDE, SITE, guide_site


def build(path: Path, site: FakeSite, url: str = GUIDE, **options: object) -> CollectionInfo:
    request = BuildRequest(url, **options)  # type: ignore[arg-type]
    return build_collection(
        request, path, embedder=HashingEmbedder(), client=site.client, delay_seconds=0
    )


def test_a_website_becomes_a_searchable_collection(tmp_path: Path) -> None:
    site = guide_site()
    path = tmp_path / "guide.db"

    info = build(path, site, language=Language.ENGLISH)

    assert info == CollectionInfo(title="Example Guide", source=GUIDE)
    assert sorted(path.parent.iterdir()) == [path]
    with closing(connect(path, read_only=True)) as connection:
        assert CollectionStore(connection).info() == info
        corpus = CorpusSearch(connection, embedder_factory=lambda _: HashingEmbedder())
        assert corpus.default_mode is SearchMode.HYBRID
        results = corpus.find_documents("which cache entry is evicted", SearchMode.HYBRID)
    assert results[0].url == f"{GUIDE}caching.html"


def test_the_crawl_stays_under_the_directory_of_the_start_page(tmp_path: Path) -> None:
    site = guide_site()

    build(tmp_path / "guide.db", site)

    assert f"{SITE}/blog/news.html" not in site.pages_requested
    assert set(site.pages_requested) == {GUIDE, f"{GUIDE}caching.html", f"{GUIDE}logging.html"}


def test_the_build_tells_how_far_it_has_got(tmp_path: Path) -> None:
    site = guide_site()
    reports: list[tuple[BuildStage, int, int]] = []

    build_collection(
        BuildRequest(GUIDE, max_pages=10),
        tmp_path / "guide.db",
        embedder=HashingEmbedder(),
        client=site.client,
        delay_seconds=0,
        progress=lambda stage, done, total: reports.append((stage, done, total)),
    )

    stages = list(dict.fromkeys(stage for stage, _, _ in reports))
    assert stages == [BuildStage.CRAWLING, BuildStage.EXTRACTING, BuildStage.INDEXING]
    assert (BuildStage.CRAWLING, 3, 10) in reports
    assert (BuildStage.EXTRACTING, 3, 3) in reports
    indexing = [(done, total) for stage, done, total in reports if stage is BuildStage.INDEXING]
    assert indexing[-1][0] == indexing[-1][1] > 0


def test_without_an_embedding_model_only_words_are_matched(tmp_path: Path) -> None:
    path = tmp_path / "guide.db"
    build_collection(
        BuildRequest(GUIDE), path, embedder=None, client=guide_site().client, delay_seconds=0
    )

    with closing(connect(path, read_only=True)) as connection:
        corpus = CorpusSearch(connection)
        assert not corpus.has_embeddings
        assert corpus.find_documents("evicted", SearchMode.LEXICAL)[0].url == f"{GUIDE}caching.html"


def test_a_website_that_gives_nothing_leaves_nothing_behind(tmp_path: Path) -> None:
    site = FakeSite({})

    with pytest.raises(BuildError, match=r"no page could be downloaded from .*\(1 http error\)"):
        build(tmp_path / "empty.db", site, url=f"{SITE}/missing/")

    assert list(tmp_path.iterdir()) == []


def test_an_existing_collection_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "guide.db"
    path.write_bytes(b"precious")

    with pytest.raises(BuildError, match="already a collection"):
        build(path, guide_site())

    assert path.read_bytes() == b"precious"


@pytest.mark.parametrize("pages", [0, 501])
def test_the_number_of_pages_is_bounded(tmp_path: Path, pages: int) -> None:
    with pytest.raises(BuildError, match="between 1 and 500"):
        build(tmp_path / "guide.db", guide_site(), max_pages=pages)


@pytest.mark.parametrize(
    ("url", "seed", "prefix"),
    [
        (GUIDE, GUIDE, GUIDE),
        (f"{GUIDE}caching.html", f"{GUIDE}caching.html", GUIDE),
        ("  https://Example.com  ", "https://example.com/", "https://example.com/"),
        (
            "https://example.com/a/b?x=1#top",
            "https://example.com/a/b?x=1",
            "https://example.com/a/",
        ),
    ],
)
def test_the_crawl_starts_at_the_url_and_stays_under_its_directory(
    url: str, seed: str, prefix: str
) -> None:
    assert crawl_scope(url) == (seed, prefix)


@pytest.mark.parametrize(
    "url", ["", "docs.example.com", "ftp://example.com/", "javascript:alert(1)"]
)
def test_only_web_addresses_can_be_crawled(url: str) -> None:
    with pytest.raises(BuildError, match="not a web address"):
        crawl_scope(url)
