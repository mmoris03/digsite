"""Domain types shared by the pipeline stages and the store."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

# Told how far a long task has got: (done, total). The total may be an upper bound.
type Progress = Callable[[int, int], None]


class SkipReason(StrEnum):
    """Why a requested URL was not stored."""

    ROBOTS = "robots"
    NOT_HTML = "not_html"
    TOO_LARGE = "too_large"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    BAD_REDIRECT = "bad_redirect"


@dataclass(frozen=True, slots=True)
class Link:
    """An outgoing hyperlink whose target is already canonical."""

    url: str
    anchor: str = ""
    nofollow: bool = False


@dataclass(frozen=True, slots=True)
class Page:
    """Metadata of a page whose HTML is stored."""

    id: int
    url: str
    depth: int
    title: str
    noindex: bool
    content_type: str = ""


@dataclass(frozen=True, slots=True)
class SkippedUrl:
    """A URL that was requested but not stored."""

    url: str
    reason: SkipReason
    status: int
    detail: str


@dataclass(frozen=True, slots=True)
class Edge:
    """A link in the corpus graph. The target may not have been crawled."""

    source: str
    target: str
    nofollow: bool


@dataclass(frozen=True, slots=True)
class CorpusStats:
    pages: int
    skipped: int
    redirects: int
    links: int
    internal_links: int
    compressed_html_bytes: int


class DuplicateKind(StrEnum):
    """How a document was found to repeat another one."""

    EXACT = "exact"
    NEAR = "near"


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What de-duplication needs to know about a document with content."""

    page_id: int
    content_hash: str
    simhash: int


@dataclass(frozen=True, slots=True)
class Duplicate:
    """A document that repeats the content of `canonical_id`."""

    canonical_id: int
    kind: DuplicateKind


@dataclass(frozen=True, slots=True)
class Document:
    """The main content of a page, as Markdown. One per stored page."""

    page_id: int
    url: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class DuplicateDocument:
    """A duplicate and the document that stands for it."""

    url: str
    canonical_url: str
    kind: DuplicateKind


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """Counts of ingested pages. `unique` is what later stages index."""

    unique: int
    exact_duplicates: int
    near_duplicates: int
    empty: int


@dataclass(frozen=True, slots=True)
class Passage:
    """A piece of a document.

    Attributes:
        headings: The headings the passage sits under, outermost first.
        text: The passage itself, as Markdown.
    """

    headings: tuple[str, ...]
    text: str

    @property
    def context(self) -> str:
        """The heading path as one line, e.g. 'argparse > Arguments > nargs'."""
        return " > ".join(self.headings)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A passage of a document: the unit that is indexed and retrieved.

    Attributes:
        id: Identifier of the chunk. It changes when the corpus is re-indexed.
        page_id: The document the chunk belongs to.
        ordinal: Position of the chunk within its document, from 0.
        context: The headings the passage sits under, as 'Title > Section'.
        text: The passage, as Markdown.
    """

    id: int
    page_id: int
    ordinal: int
    context: str
    text: str

    @property
    def indexed_text(self) -> str:
        """What is indexed and embedded: the passage preceded by its context."""
        return indexed_text(self.context, self.text)


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """What a collection is, as people see it.

    Attributes:
        title: Name to show, usually the title of the start page.
        source: Where its documents come from: the URL the crawl started at.
    """

    title: str
    source: str


def indexed_text(context: str, text: str) -> str:
    """Put a passage's heading path in front of it.

    A passage taken out of its document often does not say what it is about;
    its headings usually do.
    """
    return f"{context}\n{text}" if context else text
