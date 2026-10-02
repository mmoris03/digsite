"""Main-content extraction: the text of a page without navigation or boilerplate."""

import re
from dataclasses import dataclass

import trafilatura

# Documentation generators append a pilcrow permalink to every heading.
_PERMALINK_RE = re.compile(r"[ \t]*¶[ \t]*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^# +(.+)$", re.MULTILINE)
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class Content:
    """The main content of a page.

    Attributes:
        title: Text of the first top-level heading, or "" if there is none.
        text: The content as Markdown.
    """

    title: str
    text: str


def extract_content(html: str, url: str) -> Content | None:
    """Extract the main content of an HTML page as Markdown.

    Menus, sidebars, footers and other boilerplate are left out. Headings are
    kept, which later allows splitting a document by section.

    Args:
        html: Decoded HTML.
        url: Canonical URL of the page.

    Returns:
        The content, or None if the page has no main content.
    """
    # `fast` skips the fallback extractors: on documentation pages they were
    # slower and returned less of the page.
    markdown = trafilatura.extract(
        html, url=url, output_format="markdown", include_comments=False, fast=True
    )
    if not markdown:
        return None
    text = _BLANK_LINES_RE.sub("\n\n", _PERMALINK_RE.sub("", markdown)).strip()
    if not text:
        return None
    heading = _HEADING_RE.search(text)
    title = heading.group(1).replace("`", "").replace("*", "").strip() if heading else ""
    return Content(title=title, text=text)
