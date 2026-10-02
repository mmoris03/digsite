"""Download the Cranfield collection and convert it to the BEIR layout.

Cranfield (1,400 aerodynamics abstracts, 225 queries) is the oldest IR test
collection and small enough to evaluate in seconds.

    python scripts/fetch_cranfield.py data/benchmarks/cranfield

Two details of the original files are easy to get wrong:

- Judgements number the queries 1 to 225 by their position in the query file,
  not by the ids written in it (001, 002, 004, 008, ...).
- Relevance codes run backwards: 1 is a complete answer and 4 is of minimum
  interest; -1 marks documents of no interest. They are converted to gains
  where higher is better.
"""

import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
GAINS = {"1": 4, "2": 3, "3": 2, "4": 1, "-1": 0}

_RECORD_RE = re.compile(r"^\.I (\d+)\n(.*?)(?=^\.I \d+\n|\Z)", re.MULTILINE | re.DOTALL)
_FIELD_RE = re.compile(r"^\.([TABW])\n(.*?)(?=^\.[TABW]\n|\Z)", re.MULTILINE | re.DOTALL)


def parse_records(text: str) -> list[tuple[str, dict[str, str]]]:
    """Split a SMART-format file into (id, fields) records, in file order."""
    records = []
    for identifier, body in _RECORD_RE.findall(text):
        fields = {name: " ".join(value.split()) for name, value in _FIELD_RE.findall(body)}
        records.append((identifier, fields))
    return records


def main(output: Path) -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            content = archive.extractfile(member)
            if content is not None:
                # The files do not agree on line endings: the query file uses CRLF.
                text = content.read().decode("ascii", errors="replace")
                files[member.name] = text.replace("\r\n", "\n")

    documents = parse_records(files["cran.all.1400"])
    queries = parse_records(files["cran.qry"])
    judgements = [line.split() for line in files["cranqrel"].splitlines() if line.strip()]
    assert len(documents) == 1400, len(documents)
    assert len(queries) == 225, len(queries)
    assert {int(query) for query, _, _ in judgements} == set(range(1, 226))

    (output / "qrels").mkdir(parents=True, exist_ok=True)
    with (output / "corpus.jsonl").open("w", encoding="utf-8") as file:
        for identifier, fields in documents:
            # The abstract (.W) already starts with the title.
            record = {
                "_id": identifier,
                "title": "",
                "text": fields.get("W") or fields.get("T", ""),
            }
            file.write(json.dumps(record) + "\n")
    with (output / "queries.jsonl").open("w", encoding="utf-8") as file:
        for position, (_, fields) in enumerate(queries, start=1):
            file.write(json.dumps({"_id": str(position), "text": fields["W"]}) + "\n")
    with (output / "qrels" / "test.tsv").open("w", encoding="utf-8") as file:
        file.write("query-id\tcorpus-id\tscore\n")
        for query, document, code in judgements:
            file.write(f"{query}\t{document}\t{GAINS[code]}\n")

    print(f"{len(documents)} documents, {len(queries)} queries, {len(judgements)} judgements")
    print(f"written to {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
