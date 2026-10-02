"""Download a BEIR test collection.

BEIR packages many retrieval benchmarks in one layout, the one
`digsite eval retrieval` reads.

    python scripts/fetch_beir.py trec-covid data/benchmarks

The collection is unpacked into `<directory>/<name>`. Small ones to start with:
scifact (5,183 documents) and nfcorpus (3,633). trec-covid has 171,332.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main(name: str, directory: Path) -> None:
    url = URL.format(name=name)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(directory)
    print(f"{len(data) / 1e6:.1f} MB unpacked into {directory / name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
