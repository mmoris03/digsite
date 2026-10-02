import json
from pathlib import Path

import pytest

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.cli import common as common_module
from digsite.cli import main
from digsite.llm import LanguageModelError
from fakes import ScriptedModel


def test_near_duplicates_prints_one_row_per_distance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    article = (
        "The city council approved the new budget on Tuesday after a long debate. "
        "The plan raises spending on public transport and cuts funding for road building. "
        "Opposition members said the vote was rushed and promised to challenge it in court."
    )
    texts = tmp_path / "texts.txt"
    texts.write_text(
        f"t1 {article}\n"
        f"t2 {article.replace('on Tuesday', 'on Wednesday')}\n"
        "t3 A completely different text about gardening, tomatoes and the arrival of spring.\n",
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("t1 t2\n", encoding="utf-8")

    exit_code = main(
        ["eval", "near-duplicates", "--texts", str(texts), "--pairs", str(pairs),
         "--distance", "0", "--distance", "6"]
    )  # fmt: skip

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0] == "3 texts, 1 duplicate pairs; 64-bit fingerprints of 2-word shingles"
    assert lines[1].split() == ["distance", "precision", "recall", "f1", "tp", "fp", "fn"]
    assert lines[2].split() == ["0", "0.0000", "0.0000", "0.0000", "0", "0", "1"]
    assert lines[3].split() == ["6", "1.0000", "1.0000", "1.0000", "1", "0", "0"]


def test_near_duplicates_reports_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["eval", "near-duplicates", "--texts", str(tmp_path / "no.txt"), "--pairs", "no.txt"]
    )

    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err


def read_table(output: str) -> dict[str, dict[str, str]]:
    """Parse the table printed by the retrieval evaluations: metrics by retriever."""
    lines = output.splitlines()
    columns = lines[1].split()[1:]
    rows = {}
    for line in lines[2:]:
        if line.startswith("p: "):
            break
        name, *values = line.split()
        rows[name] = dict(zip(columns, values, strict=False))
    return rows


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A three-document collection in the BEIR layout."""
    documents = [
        {"_id": "d1", "title": "Caching", "text": "Entries are evicted when the cache is full."},
        {"_id": "d2", "title": "Logging", "text": "Logs rotate daily and are kept for a month."},
        {"_id": "d3", "title": "Install", "text": "Install the package and restart the service."},
    ]
    queries = [
        {"_id": "q1", "text": "when are cache entries evicted"},
        {"_id": "q2", "text": "log rotation"},
    ]
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps(record) for record in documents), encoding="utf-8"
    )
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(record) for record in queries), encoding="utf-8"
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8"
    )
    return tmp_path


def test_retrieval_prints_the_metrics(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval", "retrieval", "--dataset", str(dataset), "--language", "english"])

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert exit_code == 0
    assert lines[0] == "3 documents, 2 queries"
    assert lines[1].split() == ["retriever", "nDCG@10", "MAP", "MRR", "P@10", "R@100"]
    assert len(lines) == 3
    # With stemming, both queries find their document first.
    assert read_table(output) == {
        "lexical": {
            "nDCG@10": "1.0000",
            "MAP": "1.0000",
            "MRR": "1.0000",
            "P@10": "0.1000",
            "R@100": "1.0000",
        }
    }


def test_retrieval_depends_on_the_analyzer(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without stemming, "log rotation" shares no exact word with its document.
    main(["eval", "retrieval", "--dataset", str(dataset)])

    assert read_table(capsys.readouterr().out)["lexical"]["MRR"] == "0.5000"


def test_retrieval_can_evaluate_query_expansion(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["eval", "retrieval", "--dataset", str(dataset), "--language", "english", "--expand"]
    )

    assert exit_code == 0
    # Expansion only adds lower-weighted terms: the right documents stay first.
    assert read_table(capsys.readouterr().out)["lexical"]["MRR"] == "1.0000"


def test_retrieval_can_evaluate_semantic_search(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "eval", "retrieval", "--dataset", str(dataset), "--retriever", "semantic",
        "--model", "hashing",
    ]  # fmt: skip

    exit_code = main(arguments)

    table = read_table(capsys.readouterr().out)
    assert exit_code == 0
    assert list(table) == ["semantic"]
    # The model-free embedder only sees shared words: it finds the document of the
    # first query and has nothing to go on for "log rotation".
    assert 0.5 <= float(table["semantic"]["MRR"]) < 1.0
    # The vectors are kept next to the collection for the next run.
    assert (dataset / ".cache" / "hashing-256.npy").exists()


def test_retrieval_compares_several_retrievers_with_the_first(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "eval", "retrieval", "--dataset", str(dataset), "--language", "english",
        "--model", "hashing", "--retriever", "hybrid", "--retriever", "lexical",
        "--retriever", "semantic",
    ]  # fmt: skip

    exit_code = main(arguments)

    output = capsys.readouterr().out
    table = read_table(output)
    assert exit_code == 0
    assert list(table) == ["hybrid", "lexical", "semantic"]
    assert output.splitlines()[1].split()[-1] == "p"
    # The first row is what the others are compared with, so it has no p-value.
    assert "p" not in table["hybrid"]
    assert 0.0 < float(table["lexical"]["p"]) <= 1.0
    assert 0.0 < float(table["semantic"]["p"]) <= 1.0
    assert output.splitlines()[-1].startswith("p: how likely a difference in nDCG@10")
    # Lexical search with stemming is perfect here, and the fusion does not spoil it.
    assert table["hybrid"]["MRR"] == table["lexical"]["MRR"] == "1.0000"


def test_a_retriever_named_twice_is_evaluated_once(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["eval", "retrieval", "--dataset", str(dataset)]

    main([*arguments, "--retriever", "lexical", "--retriever", "lexical"])

    assert len(capsys.readouterr().out.splitlines()) == 3


def test_expansion_is_rejected_for_semantic_retrieval(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["eval", "retrieval", "--dataset", str(dataset), "--retriever", "semantic", "--expand"]
    )

    assert exit_code == 1
    assert "does not apply to semantic search" in capsys.readouterr().err


def test_retrieval_reports_a_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["eval", "retrieval", "--dataset", str(tmp_path)])

    assert exit_code == 1
    assert "corpus.jsonl does not exist" in capsys.readouterr().err


@pytest.fixture
def data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A corpus with two documents, indexed in English with the model-free embedder."""
    make_corpus(tmp_path, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--language", "english", "--model", "hashing"])
    capsys.readouterr()
    return tmp_path


def write_queries(directory: Path, content: str) -> str:
    path = directory / "queries.tsv"
    path.write_text(content, encoding="utf-8")
    return str(path)


QUERIES = f"""# two queries, one per page
when is a cache entry removed\t{SITE}/caching
how long are logs kept\t{SITE}/logging
"""


def test_search_is_scored_in_every_mode_the_corpus_allows(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = write_queries(data_dir, QUERIES)

    exit_code = main(["eval", "search", "--queries", queries, "--data-dir", str(data_dir)])

    output = capsys.readouterr().out
    table = read_table(output)
    assert exit_code == 0
    assert output.splitlines()[0] == "2 documents, 2 queries"
    assert list(table) == ["lexical", "semantic", "hybrid"]
    assert table["lexical"]["MRR"] == "1.0000"
    assert table["hybrid"]["nDCG@10"] == "1.0000"
    assert "p" in table["hybrid"]


def test_search_evaluation_scores_only_the_modes_asked_for(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = write_queries(data_dir, QUERIES)
    options = ["--retriever", "hybrid", "--authority-weight", "0.5", "--rank-constant", "10"]

    main(["eval", "search", "--queries", queries, *options, "--data-dir", str(data_dir)])

    # Hybrid search with link authority is listed next to plain hybrid search.
    table = read_table(capsys.readouterr().out)
    assert list(table) == ["hybrid", "hybrid+links"]
    assert "p" in table["hybrid+links"]


def test_search_evaluation_without_embeddings_scores_lexical_search(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--language", "english", "--no-embeddings"])
    capsys.readouterr()
    queries = write_queries(tmp_path, QUERIES)

    exit_code = main(["eval", "search", "--queries", queries, "--data-dir", str(tmp_path)])

    assert exit_code == 0
    assert list(read_table(capsys.readouterr().out)) == ["lexical"]


def test_search_evaluation_rejects_queries_written_for_another_corpus(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = write_queries(data_dir, f"cache\t{SITE}/not-crawled\n")

    exit_code = main(["eval", "search", "--queries", queries, "--data-dir", str(data_dir)])

    assert exit_code == 1
    assert f"line 1: not in the corpus: {SITE}/not-crawled" in capsys.readouterr().err


def test_search_evaluation_reports_what_the_corpus_lacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--no-embeddings"])
    capsys.readouterr()
    queries = write_queries(tmp_path, f"cache\t{SITE}/caching\n")
    arguments = ["eval", "search", "--queries", queries, "--data-dir", str(tmp_path)]

    assert main([*arguments, "--retriever", "semantic"]) == 1
    assert "has no embeddings" in capsys.readouterr().err
    assert main([*arguments, "--authority-weight", "1"]) == 1
    assert "only applies to hybrid search" in capsys.readouterr().err
    assert main(["eval", "search", "--queries", "missing.tsv", "--data-dir", str(tmp_path)]) == 1
    assert "missing.tsv does not exist" in capsys.readouterr().err


def test_search_evaluation_needs_a_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["eval", "search", "--queries", "q.tsv", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'digsite crawl' first" in capsys.readouterr().err


def reading(keywords: str) -> dict[str, str]:
    """What a scripted model replies when asked to read a query."""
    return {
        "language": "English",
        "intent": "",
        "kind": "fact",
        "keywords": keywords,
        "english": keywords,
    }


def test_search_evaluation_can_score_the_searches_a_language_model_adds(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedModel(reading("cache entry evicted"), reading("logs rotated kept"))
    monkeypatch.setattr(common_module, "create_language_model", lambda name, *, url: model)
    queries = write_queries(data_dir, QUERIES)
    options = ["--retriever", "hybrid", "--rewrites", "2"]

    exit_code = main(
        ["eval", "search", "--queries", queries, *options, "--data-dir", str(data_dir)]
    )

    table = read_table(capsys.readouterr().out)
    assert exit_code == 0
    assert list(table) == ["hybrid", "hybrid+rewrites"]
    assert "p" in table["hybrid+rewrites"]
    # The model reads each query once, however many times the query is searched for.
    assert len(model.calls) == 2


def test_search_evaluation_with_rewrites_needs_hybrid_search_and_a_model(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = write_queries(data_dir, QUERIES)
    arguments = [
        "eval",
        "search",
        "--queries",
        queries,
        "--rewrites",
        "1",
        "--data-dir",
        str(data_dir),
    ]

    assert main([*arguments, "--retriever", "lexical"]) == 1
    assert "--rewrites only applies to hybrid search" in capsys.readouterr().err

    model = ScriptedModel(LanguageModelError("cannot reach Ollama"))
    monkeypatch.setattr(common_module, "create_language_model", lambda name, *, url: model)
    assert main(arguments) == 1
    assert "cannot reach Ollama" in capsys.readouterr().err
