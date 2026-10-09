# Decisions

Why things are the way they are. Each entry says what was decided, why, what
else was considered, and what would make it worth deciding again.

## 1. A token repeated in a query counts once

**Decision.** `search_and`, `search_or` and `rank` ignore repetitions in the
query: `"oil oil"` finds the same documents as `"oil"`, in the same order. The
tokens of a query are collected in a dict keyed by token (`postings_of_query`
in `digsite/index.py`), so a repeated token is looked up once and its
repetitions are not kept.

**Why.** The boolean searches ask a yes-or-no question about each document:
does it have every token of the query, or at least one? Asking twice about the
same token cannot change the answer (`p and p` is `p`, and so is `p or p`).

`rank` does give each document a score, so there a repeated token could weigh
more. It does not, because the same search should come out in the same order
however it is typed, and nothing so far shows that people repeat a token to
make it count more.

**Alternatives.** Keep how many times each token was typed, so that `"oil oil"`
weighs twice as much as `"oil"` in `rank`. Two ways of typing the same search
would then give different orders, on the strength of a guess about why
someone typed a token twice.

**Revisit when.** There is evidence, from a measurement and not a hunch, that
repeating a token should raise its weight. Changing it means
`postings_of_query` also returns how many times each token was typed. The
current behaviour is pinned by the `a repeated token` cases in
`tests/test_index.py` and by `a repeated query token counts once` in
`tests/test_ranking.py`.

## 2. How digsite searches is something you can choose

**Decision.** digsite is meant to show how a search engine works inside, not
only to find things. So the way it searches is a setting: each way of
matching or ordering documents is kept as an option, and a new one is added
next to the ones that already exist instead of replacing them. Today there are
three: finding the documents that have every token of the query (`search_and`)
or any of them (`search_or`), and ordering the documents by how many times the
tokens of the query occur in them (`rank`).

**Why.** Running two approaches on the same documents with the same query,
and seeing where their results differ, explains each of them better than
either one alone. It also makes them easy to measure against each other.

**Alternatives.** A single, fixed way of searching: less code and a simpler
interface, but every improvement would erase the approach before it, and with
it the chance to compare them. Keeping the options has a cost too: each one
has to keep working and stay tested, and not every combination of settings
makes sense.

**Revisit when.** An option costs more to keep than it shows, for instance
one that never gives a result different from another option.
