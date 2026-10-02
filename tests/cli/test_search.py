from pathlib import Path

import pytest

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.cli import main


@pytest.fixture
def data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """A corpus with two documents, ingested and indexed in English.

    Documents are split into small chunks, and embedded with the model-free
    embedder so that no test loads a neural model.
    """
    make_corpus(tmp_path, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(
        ["index", "--data-dir", str(tmp_path), "--language", "english", "--model", "hashing",
         "--max-chars", "200"]
    )  # fmt: skip
    capsys.readouterr()
    return str(tmp_path)


def test_search_prints_ranked_results_with_title_url_and_passage(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["search", "least", "recently", "used", "--mode", "lexical", "--data-dir", data_dir]
    )

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].split()[0] == "1."
    assert lines[0].endswith("Caching")
    assert lines[1].strip() == f"{SITE}/caching"
    # The snippet is the passage that matched, not the start of the document.
    assert lines[2].strip().startswith("When the cache is full")


def test_a_document_appears_once_however_many_of_its_chunks_match(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # "cache" is in the title, so every chunk of the caching page matches it.
    main(["search", "cache", "--data-dir", data_dir])

    output = capsys.readouterr().out
    assert output.count(f"{SITE}/caching") == 1


def test_semantic_search_ranks_by_embedding_similarity(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["search", "the entry used least recently", "--mode", "semantic", "--data-dir", data_dir]
    )

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].endswith("Caching")
    # Every document has some similarity to the query, so both are listed.
    assert any(line.endswith("Logging") for line in lines)


def test_search_is_hybrid_by_default(data_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    # No document contains the word, so only the semantic half of hybrid search returns anything.
    main(["search", "zeppelin", "--data-dir", data_dir])
    default = capsys.readouterr()
    main(["search", "zeppelin", "--mode", "hybrid", "--data-dir", data_dir])
    hybrid = capsys.readouterr()

    assert default.out == hybrid.out
    assert default.out.count(SITE) == 2
    assert default.err == ""


def test_hybrid_search_puts_first_what_both_halves_agree_on(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["search", "how long are logs kept", "--data-dir", data_dir])

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].endswith("Logging")


def test_search_without_embeddings_falls_back_to_words_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])
    capsys.readouterr()

    exit_code = main(["search", "cache", "--data-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Caching" in captured.out
    assert "no embeddings; matching words only" in captured.err


def test_hybrid_search_needs_embeddings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])
    capsys.readouterr()

    exit_code = main(["search", "cache", "--mode", "hybrid", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "has no embeddings" in capsys.readouterr().err


def test_hybrid_search_accepts_fusion_and_authority_options(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["search", "cache", "--rank-constant", "10", "--authority-weight", "0.5",
         "--data-dir", data_dir]
    )  # fmt: skip

    assert exit_code == 0
    assert "Caching" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option", [["--authority-weight", "-1"], ["--rank-constant", "-5"], ["--rank-constant", "x"]]
)
def test_fusion_options_must_be_non_negative_numbers(
    option: list[str], data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["search", "cache", *option, "--data-dir", data_dir])

    assert exit_info.value.code == 2
    assert option[0] in capsys.readouterr().err


def test_authority_is_rejected_outside_hybrid_search(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "search",
            "cache",
            "--mode",
            "lexical",
            "--authority-weight",
            "0.5",
            "--data-dir",
            data_dir,
        ]
    )

    assert exit_code == 1
    assert "only applies to hybrid search" in capsys.readouterr().err


def test_hybrid_search_can_expand_its_lexical_half(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["search", "cache", "--expand", "--data-dir", data_dir])

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].startswith("expanded with: ")
    assert lines[1].endswith("Caching")


def test_semantic_search_needs_embeddings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])
    capsys.readouterr()

    exit_code = main(["search", "cache", "--mode", "semantic", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "has no embeddings" in capsys.readouterr().err


def test_expansion_is_rejected_for_semantic_search(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["search", "cache", "--mode", "semantic", "--expand", "--data-dir", data_dir])

    assert exit_code == 1
    assert "does not apply to semantic search" in capsys.readouterr().err


def test_search_uses_the_analyzer_the_index_was_built_with(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # "rotating" only matches "rotated" through the English stemmer stored with the index.
    main(["search", "rotating", "--mode", "lexical", "--data-dir", data_dir])

    assert "Logging" in capsys.readouterr().out


def test_search_respects_the_limit(data_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    main(["search", "request", "--data-dir", data_dir])
    both = capsys.readouterr().out

    main(["search", "request", "--limit", "1", "--data-dir", data_dir])
    one = capsys.readouterr().out

    assert both.count(SITE) == 2
    assert one.count(SITE) == 1


def test_search_without_matches_says_so(data_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search", "zeppelin", "--mode", "lexical", "--data-dir", data_dir])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "no results"


def test_search_accepts_ranking_options(data_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "search",
            "cache",
            "--variant",
            "bm25+",
            "--k1",
            "0.9",
            "--b",
            "0.4",
            "--data-dir",
            data_dir,
        ]
    )

    assert exit_code == 0
    assert "Caching" in capsys.readouterr().out


def test_search_with_expansion_shows_the_added_terms(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["search", "cache", "--mode", "lexical", "--expand", "--expansion-terms", "3",
         "--data-dir", data_dir]
    )  # fmt: skip

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].startswith("expanded with: ")
    added = lines[0].removeprefix("expanded with: ").split(", ")
    assert len(added) == 3
    assert "cach" not in added
    assert lines[1].endswith("Caching")


def test_search_with_expansion_and_no_matches_adds_nothing(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["search", "zeppelin", "--mode", "lexical", "--expand", "--data-dir", data_dir])

    assert capsys.readouterr().out.splitlines() == ["expanded with: (nothing)", "no results"]


def test_search_without_an_index_fails_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    exit_code = main(["search", "cache", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'digsite index' first" in capsys.readouterr().err


def test_search_warns_when_the_index_is_out_of_date(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    new_page = CACHING.replace("Caching", "Queues").replace("cache", "queue")
    make_corpus(Path(data_dir), {"/queues": new_page})
    main(["ingest", "--data-dir", data_dir])
    capsys.readouterr()

    exit_code = main(["search", "cache", "--data-dir", data_dir])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run 'digsite index' to refresh it" in captured.err
    assert "Caching" in captured.out
