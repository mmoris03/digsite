# Labelled queries for the Spanish Python documentation

Queries to score search and answers on a crawl of <https://docs.python.org/es/3/>,
with `digsite eval search` and `digsite eval answers`:

| File | Queries | What they look for |
|---|---|---|
| `queries.tsv` | 70 | A passage that answers a question or covers a topic (142 relevant pages) |
| `navigational.tsv` | 12 | The front page of a section |
| `unanswerable.tsv` | 18 | Nothing the corpus has: the right response is to decline |

Of the unanswerable questions, twelve are about software that the corpus names
in passing at most (each was checked against the text of every page), and six
are general knowledge that a language model knows on its own. The first kind
tests whether an answer is declined when the passages found are only
near-misses; the second, whether the model keeps to the passages.

Most queries are in Spanish. The documentation is only partly translated, so
many of the pages that answer them are in English.

## The corpus

The judgements refer to 86 documents, crawled on 30 September 2026 with:

```powershell
$base = "https://docs.python.org/es/3"
$exclude = "--exclude", "$base/genindex", "--exclude", "$base/py-modindex", "--exclude", "$base/search.html", "--exclude", "$base/contents.html", "--exclude", "$base/download.html"

digsite crawl --seed $base/tutorial/index.html --seed $base/howto/index.html --seed $base/faq/index.html --seed $base/reference/index.html --seed $base/using/index.html --prefix $base/ @exclude --max-depth 1 --max-pages 200
digsite crawl --seed $base/glossary.html --seed $base/whatsnew/3.14.html --seed $base/library/abc.html --seed $base/library/argparse.html --seed $base/library/tkinter.html --seed $base/installing/index.html --prefix $base/ @exclude --max-depth 0 --max-pages 20
digsite ingest
digsite index --language spanish
```

The site changes. A later crawl may add or drop pages, in which case
`digsite eval search` stops and names the judged pages that are missing; and the
content of the pages may no longer match the judgements.

## How the queries were judged

1. The queries were written from the table of contents of the corpus, each with
   the pages expected to answer it.
2. Every query was run in lexical, semantic and hybrid mode. The five best
   documents of each mode were pooled and judged from the passage that matched,
   and the ones that answer the query were added.
3. Pages judged relevant without having been returned were checked against
   their text.

A page is relevant if it has a section about what the query asks, or defines
it. A page that only mentions it is not. Judgements are binary.

## What to keep in mind when reading results

- The queries and the judgements were written while building the system that
  is being scored, knowing how each search mode behaves. There were no
  independent assessors.
- The queries are mostly questions in natural language. Few contain an exact
  identifier, an error message or an option name, which is where matching words
  does best. The set is therefore likely to favour semantic search more than
  real traffic would.
- Pages that no mode returned among its five best were only judged if they had
  been expected beforehand. Some relevant pages are probably unjudged, which
  counts against every mode alike.
- 70 and 12 queries are few. Differences of a few hundredths are within noise;
  the p-values that `digsite eval search` prints say how far within.
