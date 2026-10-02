"""The rules the packages of digsite follow when they depend on each other.

The pipeline goes crawl → ingest → index → search → answer, with the store
underneath, and two ways in on top: the command line and the HTTP interface.
These tests read the imports of every module, so that a shortcut against that
shape fails here and not in review.
"""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent.parent / "src" / "digsite"


def package_imports() -> dict[str, set[str]]:
    """For each top-level package or module of digsite, the others it imports."""
    imports: dict[str, set[str]] = defaultdict(set)
    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE)
        package = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                parts = name.split(".")
                if parts[0] == "digsite" and len(parts) > 1 and parts[1] != package:
                    imports[package].add(parts[1])
    return dict(imports)


IMPORTS = package_imports()


def test_the_imports_were_found() -> None:
    # Guards the other tests against passing because nothing was read.
    assert {"answer", "api", "cli", "index", "search", "store"} <= set(IMPORTS)


def test_no_two_packages_depend_on_each_other() -> None:
    cycles = sorted(
        f"{a} <-> {b}"
        for a, targets in IMPORTS.items()
        for b in targets
        if a < b and a in IMPORTS.get(b, set())
    )

    assert cycles == []


def test_the_store_depends_only_on_the_domain_types() -> None:
    assert IMPORTS["store"] <= {"models"}


@pytest.mark.parametrize("package", ["models", "text"])
def test_the_foundations_depend_on_nothing_else(package: str) -> None:
    assert IMPORTS.get(package, set()) == set()


def test_only_the_entry_point_uses_the_command_line() -> None:
    users = sorted(package for package, targets in IMPORTS.items() if "cli" in targets)

    assert users == ["__main__"]


@pytest.mark.parametrize(
    ("package", "later_stages"),
    [
        ("crawl", {"ingest", "index", "search", "answer", "evaluation", "api"}),
        ("ingest", {"index", "search", "answer", "evaluation", "api"}),
        ("index", {"search", "answer", "evaluation", "api"}),
        ("search", {"answer", "evaluation", "api"}),
        ("answer", {"evaluation", "api"}),
    ],
)
def test_a_stage_does_not_depend_on_the_stages_after_it(
    package: str, later_stages: set[str]
) -> None:
    assert IMPORTS[package] & later_stages == set()


def test_only_the_command_line_starts_the_http_interface() -> None:
    users = sorted(package for package, targets in IMPORTS.items() if "api" in targets)

    assert users == ["cli"]


def test_the_http_interface_serves_an_indexed_corpus_and_builds_nothing() -> None:
    # It reads what the pipeline built; crawling, ingesting and evaluating are not its job.
    assert IMPORTS["api"] & {"cli", "crawl", "ingest", "index", "evaluation"} == set()


def test_language_models_and_embeddings_know_nothing_of_the_corpus() -> None:
    # They are interchangeable parts: text in, text or vectors out.
    for package in ("llm", "embedding"):
        assert IMPORTS.get(package, set()) <= {"text"}, package
