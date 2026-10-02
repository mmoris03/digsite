# Decision log

Short records of choices that shaped the project and would not be obvious from
the code alone. Newest last.

## 1. No orchestration frameworks

**Decision.** Retrieval, ranking and RAG orchestration are implemented in this
repository. Libraries are used for what is not the subject of the project
(HTTP, the embedding model, the web server).

**Why.** The goal is to understand and be able to measure every stage. A
framework would replace exactly the parts worth building.

**Cost.** More code to own and test.

## 2. The corpus is one SQLite file that keeps the original HTML

**Decision.** Pages, the link graph and the raw response bodies (zlib-compressed)
live in a single SQLite database.

**Why.** Keeping the bodies as received lets every later stage be re-run
without crawling again. SQLite gives transactions, random access by URL and a
corpus that is one file to copy or delete. WARC, the web-archiving standard,
was considered and rejected: it needs an extra dependency and offset
bookkeeping, and nothing here needs to interoperate with archiving tools.

**Cost.** Response headers other than the content type are not preserved.

## 3. The schema is versioned inside the database

**Decision.** `PRAGMA user_version` holds the schema version, and an ordered
list of migrations moves a database forward one version at a time, each in its
own transaction.

**Why.** Later milestones add tables. An existing corpus should be upgraded in
place rather than crawled again.

## 4. The crawler is sequential

**Decision.** One request at a time, with a minimum interval per host.

**Why.** Politeness, not the network, bounds the crawl rate for a single site,
so concurrency would add complexity without speeding anything up.

**Cost.** Crawling many hosts at once is slower than it could be.

## 5. Redirects are reported, not followed, by the HTTP layer

**Decision.** The fetcher returns the redirect and the crawler queues the
target as a URL of its own.

**Why.** A followed redirect would reach its target without the scope,
`robots.txt` and pacing checks. Recording source and target also lets the link
graph resolve links that point at the old URL.

## 6. `nofollow` links are not crawled

**Decision.** Links marked `rel="nofollow"`, and every link on a page with a
`nofollow` robots meta tag, are stored in the graph with a flag but not visited.

**Why.** It is the conservative reading of the site owner's intent, and the
flag keeps the information for ranking.

## 7. Python 3.14 is the minimum version

**Decision.** The project targets the current stable Python only.

**Why.** It is an application, not a library, so there are no downstream users
to keep compatible with.

## 8. Main-content extraction is delegated to a library, in its fast mode

**Decision.** Boilerplate removal uses trafilatura with `fast=True`, producing
Markdown. The title is the first top-level heading, falling back to the page's
`<title>`.

**Why.** Separating an article from its navigation is a research problem of its
own and not what this project is about. Measured on 65 documentation pages, the
default mode (which compares several extractors and keeps one) returned 30%
less text than the fast mode and was slower; asking for metadata to get the
title doubled the time per page. Markdown keeps the headings at no extra cost,
which chunking will need.

**Cost.** A page that is only a list of links yields that list as its content.

## 9. Duplicates: exact hash first, then SimHash over distinct shingles

**Decision.** A document is an exact duplicate if the SHA-256 of its
whitespace-normalised text matches an earlier one, and a near-duplicate if its
64-bit SimHash, computed over its distinct two-word shingles, is within 6 bits
of an earlier one. The earliest crawled page of a group is the canonical one.

**Why these settings.** Measured on 10,000 news articles with 80 labelled
duplicate pairs:

| Features | Bits | Distance | Found | False positives |
|---|---|---|---|---|
| single words, by frequency | 128 | 3 | 77 | 0 |
| single words, by frequency | 64 | 3 | 80 | 249 |
| word pairs, by frequency | 64 | 6 | 80 | 0 |
| distinct word pairs | 64 | 6 | 79 | 0 |
| distinct word pairs | 64 | 10 | 80 | 0 |

Single words are too common to tell articles apart with a short fingerprint.
Word pairs fix that at half the storage of 128 bits.

**Why distinct shingles.** Weighting pairs by frequency scored best on the
articles but failed on the first real crawl: alphabetical index pages, which
list different entries but repeat phrases such as "in module" on every line,
collapsed into the same fingerprint (43 false pairs among 64 documents). With
each distinct pair counted once, only the true duplicate remains and the next
closest pair is 18 bits away. The lesson is recorded as a regression test.

**Why distance 6 and not 10.** Dropping a distinct page from the index is worse
than keeping a redundant one, and the chance of two unrelated fingerprints
colliding grows quickly with the distance.

**Cost.** One labelled pair in 80 is missed. The settings were chosen and
scored on the same collection.

## 10. Extraction is incremental; duplicate marks are recomputed every run

**Decision.** `ingest` extracts only pages that have no document yet, then
recomputes the duplicate marks for the whole corpus.

**Why.** Extraction costs about 0.3 s per page and does not depend on other
pages. Duplicate detection takes a fraction of a second and depends on all of
them, and on a threshold the user may change between runs.

**Cost.** Fingerprints are stored, so changing how they are computed requires
`ingest --force`.

## 11. The inverted index is built in one pass and held in memory

**Decision.** Postings are kept as one pair of packed `uint32` arrays per term
(item positions and term frequencies). The index is built from all documents at
once, stored in SQLite as one row per term, and loaded whole at query time.
Scores are accumulated with NumPy.

**Why.** Packed arrays make the 171,332-document TREC-COVID collection (16
million postings) index in about 30 seconds and answer in about 10 ms per
query, without a Python object per posting. Loading everything keeps the query
path to a dictionary lookup and a few array operations.

**Cost.** Memory grows with the corpus, and adding one document means
rebuilding the index. Both are acceptable for corpora of up to a few hundred
thousand documents; beyond that the index would need to be read per term and
updated incrementally.

## 12. The analyzer is a property of the index

**Decision.** Language, stop-word removal and stemming are chosen when the
index is built and stored with it. Searching reads them back; there is no way
to pass different ones to a query.

**Why.** A term only matches if documents and query were analyzed identically.
Making that impossible to get wrong is worth more than the flexibility.

**Cost.** One language per index. A corpus that mixes languages is analyzed
with the rules of one of them.

## 13. BM25 defaults: Lucene's formulation, k1 = 1.2, b = 0.75

**Decision.** Five formulations are implemented (Lucene, Robertson, ATIRE,
BM25L, BM25+). Lucene's is the default, with the usual parameters.

**Why.** nDCG@10, stop words removed and stemming applied:

| Formulation | Cranfield | TREC-COVID |
|---|---|---|
| Lucene | 0.369 | 0.662 |
| ATIRE | 0.369 | 0.662 |
| Robertson | 0.365 | 0.638 |
| BM25L | 0.349 | 0.635 |
| BM25+ | 0.343 | 0.609 |

What the text goes through matters more than the formula. With Lucene's BM25 on
TREC-COVID: plain words 0.581, stop words removed 0.595, stemming 0.605, both
0.662.

Robertson's original IDF is negative for a term that occurs in more than half
of the items. If stop words are kept it drops to 0.518 on TREC-COVID and 0.278
on Cranfield, because a query's common words then push matching items down.

Tuning k1 and b on Cranfield moved nDCG@10 between 0.341 and 0.377. That is too
small a collection to choose parameters on, so the defaults were left alone.

The order of the formulations is not universal. The `bm25s` library, on the
same files and metric code, puts BM25L first on TREC-COVID (0.628 against 0.609
for its Lucene formulation), and its BM25+ returns exactly the numbers of its
ATIRE in every configuration tried. Here BM25L and BM25+ add their lower bound
only to items that contain the term, as their authors define them, which
rewards matching common query words more. The differences between libraries
are smaller than the effect of the analysis in either of them.

## 14. Test collections use the BEIR layout

**Decision.** `eval retrieval` reads one layout: `corpus.jsonl`,
`queries.jsonl` and `qrels/test.tsv`. Scripts download collections into it;
Cranfield, which predates it, is converted.

**Why.** One loader covers every BEIR collection, and results can be compared
with published ones.

## 15. Query expansion is opt-in, with few feedback documents and a low weight

**Decision.** Pseudo-relevance feedback adds to the query the terms with the
highest signed-root log-likelihood ratio in its top results. By default it
reads 2 results, adds 40 terms and gives each a weight of 0.15 against 1 for
the terms the user typed. It only runs when asked for.

**Why a low weight.** nDCG@10 on TREC-COVID, 3 feedback documents, 20 terms:

| Weight of added terms | nDCG@10 | MAP |
|---|---|---|
| no expansion | 0.662 | 0.232 |
| 1.0 | 0.583 | 0.141 |
| 0.5 | 0.636 | 0.215 |
| 0.25 | 0.677 | 0.251 |
| 0.1 | 0.694 | 0.249 |

Added at full weight, as if the user had typed them, the new terms outvote the
query and results get worse. The same happens as more results are trusted: on
Cranfield, 20 feedback documents at full weight take nDCG@10 from 0.369 down
to 0.286.

**Why opt-in.** It roughly triples the time per query, it does not improve the
first result on either collection, and it assumes feedback documents of
similar length: their term counts are pooled, so a long one dominates. On a
crawl with pages of very different lengths it helped some queries and derailed
others. Indexing passages of bounded size (milestone 5) removes that
assumption.

**Cost.** The defaults were picked on the two collections they are reported
on, so the reported gains are optimistic. On TREC-COVID only the gain in MAP
is statistically significant.

## 16. The indexed unit is a chunk of at most 1,000 characters

**Decision.** Documents are split into passages along their own structure:
never across a heading; paragraphs packed together while they fit; a paragraph
that is too long split at line breaks, then between sentences, then between
words. Both the lexical and the vector index are built over these chunks, and
results are shown per document, each represented by its best chunk.

**Why chunks.** A passage about one thing is a better answer, and a better
input for a language model, than a whole page. Having both indexes over the
same units is also what lets their rankings be fused.

**Why characters and not words.** The first version limited chunks to 200
words. On the example corpus, which is full of identifiers, a word costs 2.6
model tokens on average, so 40% of those chunks exceeded the 512 tokens the
embedding model reads and would have been cut silently. Characters are a
steadier measure: a 1,000-character piece exceeds 500 tokens in 1.2% of cases.

**Cost.** No overlap between chunks: a fact split across a boundary is in
neither chunk whole.

## 17. Every chunk is indexed with its heading path in front

**Decision.** What is analyzed and embedded for a chunk is its heading path
(`Title > Section > Subsection`) followed by its text. A chunk outside any
heading takes the document title.

**Why.** A passage taken out of its page often does not say what it is about.
Generating that context with a language model works well and costs a model
call per chunk; the headings are free and already written by the author.

**Cost.** Words of the title count for every chunk of the document. On a first
crawl this let hundreds of chunks from alphabetical index pages, all titled
with the product name and version, outrank real content for any query that
mentioned the version. The fix was to leave those pages out of the corpus
(`crawl --exclude`), not to drop the context.

## 18. Embeddings: a small multilingual model on CPU, exact search, cached by content

**Decision.** Chunks are embedded with `intfloat/multilingual-e5-small` (384
dimensions) through sentence-transformers. Nearest neighbours are found by
comparing the query with every vector. Vectors are stored under the hash of the
text they were computed from.

**Why this model.** The corpus mixes Spanish and English and there is no GPU.
Any model can be used behind the `Embedder` interface; a model-free hashing
embedder implements it too, so that tests and smoke runs need no download.

**Why exact search.** One matrix product over a few thousand vectors takes
milliseconds. An approximate index would add a dependency and a recall
trade-off to save time that is not being spent.

**Why cache by content.** Embedding is the slowest step by far: about 5 chunks
per second on a laptop CPU. Chunk ids change on every re-index, but most text
does not, so re-indexing only embeds what is new.

**Cost.** Exact search grows linearly with the corpus, and a command-line
query pays some 15 seconds to load the model. A long-running server loads it
once.

**What the model does not do.** It scores a passage written in the language of
the query 0.02 to 0.04 higher than the same passage in another language, and
all its scores sit between about 0.74 and 0.90. So for a Spanish question, an
unrelated Spanish passage that repeats one of its words (0.815) can outrank
the English passage that answers it (0.796). Among passages in one language
the answer wins. This, and the queries where matching words is simply the
better signal, is the case for fusing both rankings in milestone 6.

## 19. Code blocks are tracked by counting fences wherever they are in a line

**Decision.** The chunker keeps a fenced code block whole, and does not take a
line starting with `#` inside one for a heading. Whether a line is inside a
block is decided by counting the fence markers seen so far, anywhere in a line,
not only at its start.

**Why.** Python comments look exactly like Markdown headings. Treating them as
such gave chunks a heading path like `Optionally run the configuration checker`
and hid the real one for the rest of the page.

**How it was got wrong first.** The first version looked for fences at the start
of a line. On the example corpus 10 of 86 documents then seemed to have an
unclosed block, and a rule was added to recover: inside code, a heading-like
line standing alone between blank lines is a heading after all. The documents
were fine. Content extraction leaves some opening fences at the end of a line
of text (14 in the corpus), and they were not being counted. Counting every
marker, all 86 documents are balanced, the 379 heading-like lines inside code
are all comments and the 1,423 outside are all headings. The recovery rule was
removed: it answered a problem that did not exist and would have turned any
comment set apart by blank lines into a heading.

**Cost.** A block that is really left unclosed swallows the headings after it:
the rest of the document becomes one section.

## 20. PageRank is implemented here and checked against networkx

**Decision.** The importance of a document is its PageRank over the links
between documents, computed by power iteration with NumPy arrays: damping 0.85,
and a page without outgoing links spreading its score over all pages.

**Why write it.** It is one short function, and the rest of the ranking is
written here too. What a library would add is confidence that it is right, and that is
obtained another way: the tests compare it with `networkx` on random graphs
with dangling and isolated nodes, to nine decimal places. `networkx` is a
development dependency only.

**What the graph is.** Links are reduced to links between indexed documents
before ranking: followed through redirects, attributed to the document that
stands for a duplicate, and dropped if they leave the corpus, point to the
document itself or are marked `nofollow`. On the example corpus, 880 of the
3,728 links crawled remain.

## 21. Hybrid search fuses rankings by position, with the standard constant

**Decision.** Hybrid search asks the lexical and the semantic retriever for
their best 1,000 chunks and fuses the two rankings by reciprocal rank
(Cormack et al., 2009): a chunk scores `1 / (60 + position)` in each ranking it
appears in. It is the default search mode when the corpus has embeddings.

**Why positions and not scores.** BM25 scores are unbounded and depend on the
query; cosine similarities from this model sit between 0.74 and 0.90. Combining
them needs a normalisation and a weight, both to be tuned. Positions need
neither.

**What it buys** (nDCG@10, p from a paired randomisation test against hybrid):

| Collection | Lexical | Semantic | Hybrid |
|---|---|---|---|
| Cranfield | 0.369 (p = 0.015) | 0.354 (p < 0.001) | 0.389 |
| SciFact | 0.687 (p = 0.005) | 0.677 (p = 0.006) | 0.716 |
| Example corpus, 70 queries | 0.752 (p < 0.001) | 0.852 (p = 0.17) | 0.825 |

On the two public collections it beats both of its parts. On the example
corpus it beats lexical search by a wide margin and is below semantic search
alone, by a difference that 70 queries cannot tell from noise. There the
lexical half is the weak one: the corpus mixes two languages, its index is
analyzed as one, and the queries are questions in natural language. Hybrid is
the default because it is never the worst of the three and is the best on the
collections nobody here wrote the queries for.

**Nothing was tuned.** The constant is the one from the paper and both
retrievers weigh the same. Smaller constants reward the top of each ranking
more, which should suit a fusion of only two. Measured:

| Rank constant | Cranfield | SciFact | Example corpus |
|---|---|---|---|
| 0 | 0.3812 | 0.7228 | 0.8360 |
| 10 | 0.3895 | 0.7225 | 0.8242 |
| 60 | 0.3892 | 0.7155 | 0.8247 |
| 200 | 0.3888 | 0.7126 | 0.8244 |

No value is best everywhere and the differences are small, so the standard one
stays. `--rank-constant` is there to repeat this. Weights were tried once, with
the library. Counting the lexical retriever double was marginally better on
the two public collections (by 0.003 and 0.001) and worse on the example
corpus; counting the semantic one double, the reverse. Equal weights stay.

**Fusing chunks, not documents.** Each retriever's chunk ranking could first be
collapsed into a ranking of documents, and those fused. On the example corpus
the two orders gave the same quality in a one-off comparison (nDCG@10 0.825
fusing documents, 0.821 fusing chunks, p = 0.76). Chunks are fused because the
answer stage needs ranked passages, and one fusion then serves both.

**The depth is fixed.** The fusion always uses the top 1,000 of each retriever,
however many results are asked for. The first ten results are therefore the
same whether ten or a thousand are requested, and what the evaluation scores is
what `digsite search` returns.

**Cost.** A retriever that is confidently wrong pulls the fusion down: a chunk
that both rank thirtieth outscores one that only one of them ranks first. For
"novedades de la versión 3.14", semantic search puts the release notes first
and hybrid search fourth, behind pages that repeat "versión 3.14".

## 22. Link authority is an opt-in prior, off by default

**Decision.** PageRank is computed and stored by `digsite index`, shown by
`digsite authority`, and used by hybrid search only when `--authority-weight`
is given. It then enters the fusion as a third ranking, of documents by
PageRank, with that weight. It reorders what the retrievers found and never
adds a result.

**Why off.** Measured on the example corpus with `digsite eval search` (nDCG@10
of hybrid search, p against no authority):

| Authority weight | 70 questions | 12 front-page searches |
|---|---|---|
| 0 | 0.825 | 0.724 |
| 0.1 | 0.829 (p = 0.44) | 0.802 (p = 0.12) |
| 0.5 | 0.801 (p = 0.06) | 0.886 (p = 0.03) |

Links help to find the front page of a section and do not help to answer a
question. The TREC web track found the same at scale: link evidence improved
entry-page finding and not topic relevance. On this site the pages with the
highest PageRank are the home page and the bug-reporting, copyright and
feedback pages, which every page links to: four documents hold 48% of the
score. That is what makes a front page
findable, and what would push those four up for any query if the weight were
large.

A weight of 0.1 costs nothing measurable on questions and gains on front-page
searches, but that was read off 82 queries written here, after looking at the
results. It is not enough to make it the default.

**What was tried and dropped.** Removing the links to pages that more than
half the corpus links to, as template boilerplate, gives a PageRank led by the
glossary and the language reference. As a prior it did worse than the
unfiltered one on both kinds of query: the links it removes are the ones that
point at front pages.

## 23. The example corpus has its own labelled queries

**Decision.** `benchmarks/python-docs-es/` holds 70 questions and 12 front-page
searches over the example corpus, each with the pages that answer it, and
`digsite eval search` scores the search modes against them through the same
code path as `digsite search`.

**Why.** The public collections have no links and one language, so they say
nothing about PageRank or about a bilingual corpus, and they are not what the
system is demonstrated on. Without judgements for the real corpus, the choice
between search modes there would rest on a handful of examples.

**Cost.** The queries were written and judged while building the system, not
by independent assessors, and they are mostly natural-language questions,
which favours semantic search. The README of that directory says how they were
made and what follows from it. Results on them are reported next to the
results on the public collections, not instead of them.

## 24. Language models sit behind a protocol and are reached over plain HTTP

**Decision.** The rest of the system knows a language model as `LanguageModel`:
a name and `generate(prompt, system, schema)`. The one implementation,
`OllamaModel`, posts to Ollama's `/api/chat` with `httpx`, which the crawler
already uses. `create_language_model` is the one place that picks the client.

**Why no SDK and no framework.** One endpoint and one reply shape do not need
either. The client is a hundred lines, and its tests run against a simulated
server, covering the failures seen in practice: server not running, model not
pulled, prompt larger than the context window. A hosted model would be one more
class behind the same protocol.

**Why a JSON schema for every reply the code reads.** Ollama constrains the
model's output to the schema, so even a 4B model returns a document that parses.
The code then checks the values, not the syntax.

**Why the context size is always sent.** Ollama's default window is small and
it cuts longer prompts without an error. The client asks for 8,192 tokens and
logs a warning when a prompt fills them.

**Cost.** Text inside JSON needs escaping. The 4B model sometimes writes Windows
paths with backslashes that JSON reads as control characters, and the path in
the answer comes out mangled.

## 25. The answer stage reads the question, searches, and writes: two calls, kept simple

**Decision.** `Answerer.ask` follows the pipeline of the course material,
shortened because chunks already exist at index time:

1. The model reads the question: its language, its intent, the kind of answer
   it calls for, its key terms, and the same in English.
2. The question and the two added queries are searched for with the corpus's
   search mode, and the rankings are fused by reciprocal rank.
3. The best six chunks are kept, at most three from one document.
4. The model writes the answer from them, citing them by number, or declines.

**Why named fields for the reading.** Asked for "two queries, the second in
English", the model gave two Spanish queries. Asked for a `keywords` field and
an `english` field, it fills each as asked.

**What the added queries buy** (hybrid search, 70 questions of the example
corpus, one-off measurement repeatable with `digsite eval search --rewrites 2`):
nDCG@10 went from 0.825 to 0.873 (p = 0.06) and the share of questions with a
relevant page among the six passages from 92.9% to 95.7%. Nearly all of it comes
from the English query, in a corpus that is half English. It costs about 14
seconds per question on the laptop CPU.

**Why the language is named in the answer prompt.** "Write in the language of
the question" was ignored: the first real answer to a Spanish question came
back in English. "Write the answer in Spanish" is followed.

**What the kind of question is used for.** It chooses between short, stepwise
and explanatory answers. It was not reliable enough for anything else: of the
12 front-page searches, only 4 were recognised as looking for a page, so it
does not switch on link authority.

**Why the prompts are plain.** The model is the weakest and the most
replaceable part. Effort went into the code around it: what it is given, what
is checked in its replies, and how it is measured.

## 26. What the model replies is checked by code

**Decision.** An answer counts as given only if the model says it can answer
and writes something. Citations are read from the text: `[1]`, `[1][2]` and
`[1, 2]` are citations, numbers that match no passage are ignored, and a
bracket right after a name, as in `sys.argv[1]`, is code. The terminal lists
only the passages the answer cites, and warns about an answer that cites
nothing. Without passages, the model is not asked to write at all.

**What was tried to make a 4B model decline better, and dropped.** It answers
questions whose passages only share words with them: it wrote code for "cómo
crear un DataFrame con pandas" from a page that merely names pandas.

- Asking it to list the passages that answer before giving its verdict made it
  list nearly all of them, and it began answering general-knowledge questions
  that it had declined before.
- A separate judging call, sharing the prompt prefix so that Ollama's cache
  kept it cheap, gave one-line reasons that were right and verdicts that
  contradicted them: "it does not directly address creating a DataFrame",
  then `answerable: true`.
- A threshold on the best semantic similarity tells general-knowledge questions
  apart (all six below 0.84, the lowest answerable question at 0.845) but not
  questions about software the corpus names in passing ("requests" at 0.898).

None of them is kept. With this model the gain did not repay the complexity,
and a larger model is one option away. How often it answers what it should
decline is measured, not hidden: see `digsite eval answers`.

## 27. Answers are scored by where they point

**Decision.** `digsite eval answers` asks the labelled questions and reports
how many were answered, how many answers cite a page judged relevant, how many
cite nothing, what share of citations point to a relevant page, how often a
relevant page was among the passages at all, and how many of the unanswerable
questions were declined. Every answer is saved, and a run can be resumed.

**Why no language model as a judge.** The course material asks for a judge
different from the model that answers. Here that means a second local model and
hours more per run, and the scores would only be as good as that judge. The
metrics used here are cheap, repeatable and objective, given the judgements.

**Cost.** They measure grounding, not correctness: an answer that cites the
right page and misreads it scores as well as one that reads it right. Reading
the saved answers is the only check of that.

## 28. The dependencies between packages are a tested rule

**Decision.** `tests/test_architecture.py` reads the imports of every module
and fails if two packages depend on each other, if the store depends on
anything but the domain types, if a stage depends on a later one, if the
language model or embedding packages depend on the corpus, or if anything but
the entry point uses the command line.

**Why.** The shape of the code (crawl → ingest → index → search → answer, the
store underneath, the command line on top) is what makes each stage testable
alone and replaceable. A rule that is only written down erodes one convenient
import at a time.

**How it paid off at once.** Mapping the imports found that `index` and `store`
depended on each other: the chunk store used the passage type defined in the
chunker, and the store of the inverted index lived in `store`. The passage
became a domain type in `models.py`, and the inverted index is now stored by
the `index` package that owns it.

## 29. The HTTP interface is a thin layer, handed what it serves

*Since decision 34 it serves many collections and can add new ones, through
the `library` package; what follows still holds for each collection.*

**Decision.** `digsite serve` runs a FastAPI application with three endpoints
(`/api/search`, `/api/ask`, `/api/status`) and a page. `create_app` receives
the corpus and the language model already built; it opens no file and makes no
decision about the pipeline. Search and answers go through the same two
objects as the command line: `CorpusSearch.find_documents` and
`Answerer.for_corpus`.

**Why FastAPI.** Request validation, an OpenAPI description and a test client
come with it, and its endpoints are typed functions that mypy checks. Nothing
in it is needed beyond that, and nothing outside `api/` imports it.

**Why separate request and response types.** `api/schemas.py` defines what the
interface accepts and returns, and converts from the domain types. A field
renamed in `Answer` then breaks a conversion that mypy and the tests see, not a
client that nobody here sees.

**Why the logic moved out of the command line.** Turning matching chunks into
documents, each shown by its best passage, used to live in the `search`
command. The HTTP interface needed it too, so it became
`CorpusSearch.find_documents` and `search/results.py`, and the command now only
prints what it returns. A search gives the same documents in the terminal and
in the browser because there is only one place that decides them.

**Why plain threads.** The endpoints are ordinary functions, which FastAPI runs
in a pool of threads. Asking takes one to two minutes of a language model
running on the CPU; it does not need an event loop, and searching meanwhile
works because each request has its own thread.

**Why the page has no framework and no build step.** It is three static files.
Everything the server returns is put on the page with `textContent`, never as
HTML, so a document cannot inject markup; links are only made from `http(s)`
addresses; and the page is served with a content security policy that allows
scripts from the server alone. A framework would add a toolchain to a page
whose whole job is a form, a list and an answer.

**Cost.** No streaming: an answer arrives whole after a minute or two, with a
timer meanwhile. No authentication and no rate limit: the server listens on
127.0.0.1 unless told otherwise. Exposing it on the internet would need both,
and is not planned (decision 31).

## 30. One read-only connection serves every request, without the statement cache

**Decision.** `digsite serve` opens the corpus with `connect(path,
read_only=True)`: SQLite's read-only mode, `check_same_thread=False`, and
`cached_statements=0`. The request threads share that one connection.

**Why one connection.** The indexes and the embedding model are loaded once
and kept by `CorpusSearch`; what requests read from the database afterwards is
a few chunks and documents. One connection, shared, keeps the stores simple:
they take a connection and do not care where it came from. Read-only, the
server cannot damage the corpus, and opening it does not upgrade its schema
behind the user's back: it asks for another command to do that.

**Why without the statement cache.** SQLite is compiled thread-safe
(`sqlite3.threadsafety == 3`), so sharing a connection looked safe. A first
test, 4 threads reading the same row 8 times in all, failed in about one run
out of five, returning `None` for a row that exists. A stress test with 16 threads gave malformed rows and
`InterfaceError: bad parameter or other API misuse` with the default cache, and
0 errors in 64,000 queries with `cached_statements=0`. The `sqlite3` module
caches prepared statements per connection and hands the same one to two
threads; without the cache, each query prepares its own. The cost is preparing
each statement again, microseconds next to the rest of a request.

**The test that guards it** runs 16 tasks on 8 threads, each reading 200 rows
through the one connection, and passed 20 runs out of 20.

## 31. No public demo: the project runs locally, with Docker

**Decision.** The last milestone packages the system to run with one command
(Docker Compose, with Ollama alongside) and checks it on every change
(continuous integration). It is not deployed on the internet. Screenshots in
the README are to show the page working instead, once the page is final.

**Why.** What makes a public demo costly is the language model, not the page.
Without a GPU, an answer takes one to two minutes and concurrent questions wait
for each other; a GPU or a paid API costs money every month, and either needs
rate limiting, abuse protection and someone to keep it up. None of that work
says anything about how the system is built, which is what this project is
meant to show; and the measured results in the README say more about its
quality than a few questions tried on a demo would.

**What would change it.** A search-only demo is cheap: search does not need the
language model, answers in milliseconds and runs on a small CPU instance from
the same image. It can be added later without touching the code.

## 32. One image for every command, and the data outside it

*The corpus is no longer mounted from `./data`: see decision 37.*

**Decision.** The `Dockerfile` packages the command line, not only the server:
`docker compose run --rm digsite crawl …` works like `digsite crawl …`, and
the image's default command is `serve`. The corpus lives in `./data`, mounted
into the container, and the embedding model in a Docker volume; neither is
part of the image. `compose.yaml` adds Ollama and a one-off service that pulls
the language model.

**Why the CPU build of PyTorch.** The embedding model needs PyTorch, and the
default Linux wheel brings the CUDA libraries with it: gigabytes that a CPU
container never uses. It is installed first, from PyTorch's CPU index, so that
`sentence-transformers` finds it already there. The image is about 2 GB, of
which PyTorch is 0.8 and the rest of the embedding model's libraries
(transformers, scipy, scikit-learn) most of the remainder.

**Why the corpus is not in the image.** It is built by the user, from the site
they choose, and it changes when they crawl or index again. In a volume it
outlives the containers, and a corpus built without Docker can be served with
it and the other way round: it is the same SQLite file.

**Why the language model is configured through the environment.**
`DIGSITE_LLM` and `DIGSITE_LLM_URL` set the defaults of `--llm` and
`--llm-url`, and an option on the command line still wins. Compose sets them
once, so `ask`, `serve` and `eval answers` all find Ollama at `ollama:11434`
without repeating it in every command.

**Why nothing is exposed beyond this computer.** The port is published on
127.0.0.1, and the container runs as an unprivileged user. The server has no
authentication (decision 29), and there is no public deployment (decision 31).

## 33. Continuous integration runs everything that does not need the network

**Decision.** On every push to `main` and every pull request, GitHub Actions
runs, on Linux: `ruff check`, `ruff format --check`, `mypy` and `pytest`; and,
in a second job, builds the container image and runs `digsite --version` in it.

**What is left out.** The six tests marked `model` load the real embedding
model, which means downloading it from Hugging Face on every run. Everything
else runs offline: the crawler against an in-memory site, the language model
scripted, a hashing embedder in place of the neural one. That is what keeps the
suite fast and deterministic in CI; the model tests are run by hand.

**Why the image is built in CI.** A Dockerfile that is not built breaks
without anyone noticing, usually through a dependency. Building it on every
push costs a few minutes and catches that.

**Why on Linux.** The project was developed on Windows. CI is the first place
it runs on another system, which is what a user of the container gets.

## 34. One collection per website, each in its own file

**Decision.** The server serves a directory of collections, `<id>.db` each,
and the page lets the user choose one and add new ones from an address. A
collection is exactly the corpus of the earlier milestones, with a title and
the address it came from stored in it. The new `library` package lists,
opens and builds collections; the command line works on one of them with
`--collection ID`, whose default, `corpus`, is the file every earlier corpus
already was.

**Why a file per website, not one database for all.** Every stage was built for
one corpus: one inverted index, one set of vectors, one link graph. Separate
files keep all of that as it is, keep each website's results apart without a
column added to every table and query, and make removing a website deleting a
file. Above all, building a new website never writes to a file that the server
is reading: the hard part of updating an index in use simply does not arise.

**Why a new package.** Building a collection runs crawl, ingest and index; the
HTTP interface must not run them itself (decision 28's rules, which
`tests/test_architecture.py` enforces). `library` is the one place that does,
below both ways in, so the server and the command line build collections the
same way.

**Cost.** Searching several websites at once would mean searching several
collections and fusing the results; it is not done.

## 35. Websites are built in the background, one at a time, and appear complete

**Decision.** Adding a website answers at once (202) with the id the new
collection will have. A single background thread then crawls the website,
extracts its text and indexes it, into `<id>.db.partial`, and renames the file
to `<id>.db` only when everything has succeeded. The page polls
`GET /api/collections` to show how far each build has got.

**Why one at a time.** Crawling waits a second between requests, and embedding
uses every core of the processor. Two builds at once would compete for both and
finish no sooner; queued, each finishes as soon as it can.

**Why a temporary file.** A collection is listed by its file name. Renaming
only a complete file means a half-built collection is never listed, and a
failed build leaves nothing behind. A build cut short by stopping the server
leaves its temporary file, which the server deletes when it starts again.

**Why a daemon thread and no task system.** A build takes minutes and nothing
else needs to run in the background. A thread, a queue and a lock are all of
it; a task queue with its broker would be the largest part of the system. The
cost is that a build in progress is lost when the server stops.

**What the crawl may visit.** The pages under the folder of the address given:
from `https://docs.python.org/3/tutorial/`, the tutorial and nothing else. It is
the scope a person usually means, and keeps a build from wandering over a whole
site.

**A bug the tests caught.** If making the HTTP client for a build failed, the
error escaped the worker thread, which died: later builds would have stayed
queued forever. Every error now ends the build that raised it and nothing
else, and a test asks for a build after a failing one.

## 36. Searches and builds share one embedding model

**Decision.** The library loads each embedding model once and hands the same
instance to every collection and to the builds.

**Why it is safe.** The tokenizer library behind the model has a reputation for
failing when called from several threads at once. Measured before relying on
it: 8 threads embedding queries and 2 embedding batches of passages, at the
same time on one instance, gave no error and no wrong vector in two runs. One
instance instead of one per collection saves about 470 MB per collection.

**Cost.** While a website is being indexed, searches compete with it for the
processor and take longer.

## 37. The containers keep the collections in a Docker volume

**Decision.** `compose.yaml` mounts a named volume on `/app/data`, not the
host's `./data` folder.

**Why.** The server now writes: every website added is a new file. A folder of
the host that does not exist yet is created by Docker as root on Linux, and the
container's unprivileged user then cannot write to it. A named volume is
initialised from the image, where `/app/data` belongs to that user.

**Cost.** The files are not visible as a folder of the host. A corpus built
outside Docker can still be copied in with `docker compose cp`.
