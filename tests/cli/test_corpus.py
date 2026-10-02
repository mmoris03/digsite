from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite import __version__
from digsite.cli import main
from digsite.cli.common import CORPUS_FILENAME
from digsite.cli.corpus import format_crawl_summary
from digsite.crawl import CrawlStats
from digsite.models import Link, SkipReason
from digsite.store import AuthorityStore, CrawlStore, DocumentStore, connect
from fakes import article_html


def read_counts(output: str) -> dict[str, int]:
    """Parse the numeric rows printed by `digsite stats`."""
    counts = {}
    for line in output.splitlines():
        name, value = line.rsplit(maxsplit=1)
        if value.isdigit():
            counts[name.strip()] = int(value)
    return counts


def make_linked_corpus(data_dir: Path, extra: dict[str, str] | None = None) -> None:
    """Create a corpus of two pages that link to each other, plus any extra pages."""
    with closing(connect(data_dir / CORPUS_FILENAME)) as connection:
        store = CrawlStore(connection)
        store.save_page(
            f"{SITE}/caching",
            0,
            CACHING.encode(),
            links=[Link(f"{SITE}/logging"), Link(f"{SITE}/not-crawled")],
        )
        store.save_page(f"{SITE}/logging", 0, LOGGING.encode(), links=[Link(f"{SITE}/caching")])
        for path, html in (extra or {}).items():
            store.save_page(f"{SITE}{path}", 0, html.encode())


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"digsite {__version__}"


@pytest.mark.parametrize("command", ["stats", "ingest", "index"])
def test_commands_that_need_a_corpus_fail_with_a_hint(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([command, "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'digsite crawl' first" in capsys.readouterr().err


def test_crawl_rejects_an_invalid_seed_before_touching_the_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["crawl", "--seed", "not-a-url", "--data-dir", str(tmp_path)])

    assert exit_code == 2
    assert "Seeds must be absolute" in capsys.readouterr().err


def test_crawl_requires_a_seed() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["crawl"])

    assert exit_info.value.code == 2


def test_crawl_summary_lists_skips_by_frequency() -> None:
    stats = CrawlStats(
        saved=3,
        links=12,
        redirects=1,
        skipped=Counter({SkipReason.ROBOTS: 1, SkipReason.HTTP_ERROR: 2}),
    )

    assert format_crawl_summary(stats) == (
        "pages stored: 3; links: 12; redirects: 1; skipped: 2 http_error, 1 robots"
    )


def test_crawl_summary_omits_skips_when_there_are_none() -> None:
    assert format_crawl_summary(CrawlStats(saved=1)) == "pages stored: 1; links: 0; redirects: 0"


def test_ingest_extracts_real_pages_and_reports_duplicates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING, "/logging": LOGGING, "/index.html": CACHING})

    exit_code = main(["ingest", "--data-dir", str(tmp_path)])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "pages extracted: 3; unique documents: 2; exact duplicates: 1; near duplicates: 0; empty: 0"
    )
    with closing(connect(tmp_path / CORPUS_FILENAME)) as connection:
        documents = DocumentStore(connection).documents()
    assert [(doc.url, doc.title) for doc in documents] == [
        (f"{SITE}/", "Caching"),
        (f"{SITE}/logging", "Logging"),
    ]
    assert "least recently" in documents[0].text
    assert "All rights reserved" not in documents[0].text


def test_ingest_twice_extracts_nothing_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    main(["ingest", "--data-dir", str(tmp_path)])

    assert capsys.readouterr().out.startswith("pages extracted: 0; unique documents: 1;")


def test_index_reports_what_it_built(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_corpus(tmp_path, {"/": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    exit_code = main(
        ["index", "--data-dir", str(tmp_path), "--language", "english", "--no-embeddings"]
    )

    output = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert output.startswith("documents: 2; chunks: 2; terms: ")
    assert "vectors" not in output


def test_index_embeds_the_chunks_and_reuses_vectors_on_the_next_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    main(["index", "--data-dir", str(tmp_path), "--model", "hashing"])
    first = capsys.readouterr().out.strip()
    main(["index", "--data-dir", str(tmp_path), "--model", "hashing"])
    second = capsys.readouterr().out.strip()

    assert first.endswith("vectors: 2 computed, 0 reused")
    assert second.endswith("vectors: 0 computed, 2 reused")


def test_index_splits_long_documents_into_smaller_chunks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    main(["index", "--data-dir", str(tmp_path), "--no-embeddings", "--max-chars", "200"])

    assert capsys.readouterr().out.startswith("documents: 1; chunks: 3;")


def test_index_without_documents_fails_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING})

    exit_code = main(["index", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'digsite ingest' first" in capsys.readouterr().err


def test_stats_prints_every_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_corpus(tmp_path, {"/": CACHING, "/index.html": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--language", "english", "--model", "hashing"])
    capsys.readouterr()

    exit_code = main(["stats", "--data-dir", str(tmp_path)])

    output = capsys.readouterr().out
    counts = read_counts(output)
    assert exit_code == 0
    assert counts["pages"] == 3
    assert counts["skipped urls"] == 0
    assert counts["unique documents"] == 2
    assert counts["exact duplicates"] == 1
    assert counts["near duplicates"] == 0
    assert counts["empty documents"] == 0
    assert counts["chunks"] == 2
    assert counts["indexed chunks"] == 2
    assert counts["index terms"] > 20
    assert counts["embedded chunks"] == 2
    assert counts["link-scored documents"] == 2
    assert output.splitlines()[-1].split() == ["embedding", "model", "hashing-256"]


def test_index_scores_the_documents_by_their_links(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_linked_corpus(tmp_path)
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])

    # Both pages link to each other; the link to a page that was not crawled does not count.
    assert capsys.readouterr().out.strip().endswith("; links: 2")
    with closing(connect(tmp_path / CORPUS_FILENAME)) as connection:
        scores = AuthorityStore(connection).scores()
    assert list(scores.values()) == pytest.approx([0.5, 0.5])


def test_authority_lists_the_documents_by_link_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    orphan = article_html(
        "Queues",
        "Jobs wait in a queue until a worker is free to take them. Each worker takes one job at "
        "a time and reports back when it has finished, whether the job succeeded or not.",
        "A job that fails is retried three times, waiting longer before each attempt, and is "
        "then moved to a separate queue where an operator can inspect what went wrong.",
    )
    make_linked_corpus(tmp_path, extra={"/queues": orphan})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])
    capsys.readouterr()

    exit_code = main(["authority", "--limit", "2", "--data-dir", str(tmp_path)])

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(lines) == 4
    # The two pages that link to each other outrank the one nobody links to.
    assert lines[0].split()[0] == "1."
    assert {lines[0].split()[-1], lines[2].split()[-1]} == {"Caching", "Logging"}
    assert lines[1].strip() in {f"{SITE}/caching", f"{SITE}/logging"}
    assert float(lines[0].split()[2].rstrip("x")) > 1.0


def test_authority_before_indexing_fails_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING})

    exit_code = main(["authority", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'digsite index' first" in capsys.readouterr().err


def test_stats_before_indexing_shows_an_empty_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/": CACHING})

    main(["stats", "--data-dir", str(tmp_path)])

    output = capsys.readouterr().out
    counts = read_counts(output)
    assert counts["pages"] == 1
    assert counts["chunks"] == 0
    assert counts["indexed chunks"] == 0
    assert counts["embedded chunks"] == 0
    assert counts["link-scored documents"] == 0
    assert output.splitlines()[-1].split() == ["embedding", "model", "(none)"]
