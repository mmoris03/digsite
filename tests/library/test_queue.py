import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from digsite.embedding import HashingEmbedder
from digsite.library import BuildQueue, BuildRequest, BuildStage, Library
from digsite.search.corpus import SearchMode
from fakes import HTML, FakeSite
from library.sites import GUIDE, SITE, guide_site


def make_queue(library: Library, site: FakeSite) -> BuildQueue:
    return BuildQueue(
        library, embedding_model="hashing", client_factory=site.new_client, delay_seconds=0
    )


def stage_of(builds: BuildQueue, build_id: str) -> BuildStage:
    status = builds.status(build_id)
    assert status is not None
    return status.stage


@pytest.fixture
def library(tmp_path: Path) -> Iterator[Library]:
    created = Library(tmp_path, embedder_factory=lambda _: HashingEmbedder())
    yield created
    created.close()


def test_a_build_runs_in_the_background_and_its_collection_can_be_searched(
    library: Library,
) -> None:
    builds = make_queue(library, guide_site())

    asked = builds.submit(BuildRequest(GUIDE))
    ended = builds.wait(asked.id, timeout=30)

    assert asked.stage is BuildStage.QUEUED
    assert asked.collection_id == "docs-example-com-guide"
    assert ended.stage is BuildStage.DONE
    assert ended.error is None
    assert ended.finished is not None
    assert ended.finished >= ended.submitted
    assert not ended.active
    hits = library.corpus(ended.collection_id).find_documents("evicted", SearchMode.HYBRID)
    assert hits[0].url == f"{GUIDE}caching.html"


def test_builds_of_the_same_website_get_their_own_collections(library: Library) -> None:
    builds = make_queue(library, guide_site())

    first = builds.submit(BuildRequest(GUIDE))
    second = builds.submit(BuildRequest(GUIDE))
    builds.wait(second.id, timeout=30)

    assert (first.collection_id, second.collection_id) == (
        "docs-example-com-guide",
        "docs-example-com-guide-2",
    )
    assert library.ids() == ["docs-example-com-guide", "docs-example-com-guide-2"]
    assert [status.id for status in builds.statuses()] == [second.id, first.id]


def test_builds_run_one_at_a_time_in_the_order_asked(library: Library) -> None:
    release = threading.Event()
    order: list[str] = []

    def slow_page(request: httpx.Request) -> httpx.Response:
        order.append(str(request.url))
        release.wait(10)
        return httpx.Response(200, headers=HTML, content=b"<main><p>Slow page text.</p></main>")

    site = FakeSite({f"{SITE}/a/": slow_page, f"{SITE}/b/": slow_page})
    builds = make_queue(library, site)

    first = builds.submit(BuildRequest(f"{SITE}/a/"))
    second = builds.submit(BuildRequest(f"{SITE}/b/"))
    try:
        with pytest.raises(TimeoutError):
            builds.wait(second.id, timeout=0.2)
        assert stage_of(builds, first.id) is BuildStage.CRAWLING
        assert stage_of(builds, second.id) is BuildStage.QUEUED
    finally:
        release.set()
    builds.wait(second.id, timeout=30)

    assert order == [f"{SITE}/a/", f"{SITE}/b/"]


def test_a_failed_build_says_why_and_the_next_one_still_runs(library: Library) -> None:
    builds = make_queue(library, guide_site())

    failed = builds.submit(BuildRequest(f"{SITE}/nowhere/"))
    succeeded = builds.submit(BuildRequest(GUIDE))
    builds.wait(succeeded.id, timeout=30)

    status = builds.status(failed.id)
    assert status is not None
    assert status.stage is BuildStage.FAILED
    assert "no page could be downloaded" in (status.error or "")
    assert stage_of(builds, succeeded.id) is BuildStage.DONE
    assert library.ids() == ["docs-example-com-guide"]


def test_an_unexpected_error_is_reported_and_does_not_stop_the_queue(library: Library) -> None:
    site = guide_site()
    clients = iter([RuntimeError("no network"), site.new_client()])

    def flaky() -> httpx.Client:
        client = next(clients)
        if isinstance(client, Exception):
            raise client
        return client

    builds = BuildQueue(library, embedding_model=None, client_factory=flaky, delay_seconds=0)

    failed = builds.wait(builds.submit(BuildRequest(GUIDE)).id, timeout=30)
    succeeded = builds.wait(builds.submit(BuildRequest(GUIDE)).id, timeout=30)

    assert failed.stage is BuildStage.FAILED
    assert failed.error == "unexpected error: no network"
    assert succeeded.stage is BuildStage.DONE
