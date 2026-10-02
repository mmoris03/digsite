"""HTML parsing for the crawler: links, title, robots directives and decoding."""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from digsite.crawl.urls import canonicalize
from digsite.models import Link

_WHITESPACE_RE = re.compile(r"\s+")
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-]+)", re.I)
_HEADER_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-]+)", re.I)


@dataclass(slots=True)
class ParsedPage:
    title: str = ""
    links: list[Link] = field(default_factory=list)
    noindex: bool = False
    nofollow: bool = False


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.base_href: str | None = None
        self.title: str | None = None
        self.robots: set[str] = set()
        self.raw_links: list[tuple[str, str, bool]] = []
        self._title_parts: list[str] | None = None
        self._href: str | None = None
        self._nofollow = False
        self._anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if tag == "a":
            self._close_anchor()
            href = attributes.get("href", "").strip()
            if href:
                self._href = href
                self._nofollow = "nofollow" in attributes.get("rel", "").lower().split()
        elif tag == "base" and self.base_href is None and attributes.get("href"):
            self.base_href = attributes["href"].strip()
        elif tag == "title" and self.title is None:
            self._title_parts = []
        elif tag == "meta" and attributes.get("name", "").lower() == "robots":
            content = attributes.get("content", "").lower()
            self.robots.update(token.strip() for token in content.split(","))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._close_anchor()
        elif tag == "title" and self._title_parts is not None:
            # Only the first <title> counts: inline SVGs carry their own.
            self.title = _collapse("".join(self._title_parts))
            self._title_parts = None

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._anchor_parts.append(data)
        if self._title_parts is not None:
            self._title_parts.append(data)

    def close(self) -> None:
        super().close()
        self._close_anchor()

    def _close_anchor(self) -> None:
        if self._href is not None:
            anchor = _collapse("".join(self._anchor_parts))
            self.raw_links.append((self._href, anchor, self._nofollow))
        self._href = None
        self._nofollow = False
        self._anchor_parts = []


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def parse_page(html: str, page_url: str) -> ParsedPage:
    """Extract the title, outgoing links and `<meta name="robots">` directives.

    Links come back canonical and de-duplicated. Links that are not HTTP(S) and
    links to the page itself (in-page anchors) are dropped.

    Args:
        html: Decoded HTML.
        page_url: Canonical URL of the page, used to resolve relative links.
    """
    parser = _PageParser()
    parser.feed(html)
    parser.close()

    base = canonicalize(parser.base_href, base=page_url) if parser.base_href else None
    base = base or page_url
    page_nofollow = "nofollow" in parser.robots or "none" in parser.robots

    links: dict[str, Link] = {}
    for href, anchor, nofollow in parser.raw_links:
        url = canonicalize(href, base=base)
        if url is None or url == page_url:
            continue
        previous = links.get(url)
        if previous is None:
            links[url] = Link(url, anchor, nofollow or page_nofollow)
        elif not previous.anchor and anchor:
            links[url] = Link(url, anchor, previous.nofollow)

    return ParsedPage(
        title=parser.title or "",
        links=list(links.values()),
        noindex="noindex" in parser.robots or "none" in parser.robots,
        nofollow=page_nofollow,
    )


def decode_body(body: bytes, content_type: str = "") -> str:
    """Decode the body of an HTML response.

    Uses the charset from the Content-Type header; failing that, the one
    declared in a `<meta charset>` near the top of the document; and finally
    UTF-8. Never raises: undecodable bytes become replacement characters.
    """
    candidates = []
    header_match = _HEADER_CHARSET_RE.search(content_type)
    if header_match:
        candidates.append(header_match.group(1))
    meta_match = _META_CHARSET_RE.search(body[:4096])
    if meta_match:
        candidates.append(meta_match.group(1).decode("ascii"))
    candidates.append("utf-8")

    for charset in candidates:
        try:
            return body.decode(charset, errors="replace")
        except LookupError:  # unknown charset: try the next one
            continue
    return body.decode("utf-8", errors="replace")
