"""URL canonicalisation and filtering.

Two URLs that point to the same resource must end up as the same string, so
that the crawler never downloads a page twice and the link graph has no
duplicate nodes.
"""

import re
from dataclasses import dataclass
from typing import Self
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}
_TRACKING_PARAMS = frozenset({"fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "igshid"})
_TRACKING_PREFIXES = ("utm_",)
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT_RE = re.compile(r"%([0-9a-fA-F]{2})")
# Characters quote() must leave alone in a path. '%' is included so that what is
# already percent-encoded is not encoded twice.
_PATH_SAFE = "/%:@!$&'()*+,;=-._~"

# Resources that are not pages: not worth requesting at all.
_SKIPPED_EXTENSIONS = frozenset(
    {
        ".7z", ".avi", ".bz2", ".css", ".csv", ".doc", ".docx", ".epub", ".exe", ".gif",
        ".gz", ".ico", ".jpeg", ".jpg", ".js", ".json", ".mov", ".mp3", ".mp4", ".pdf",
        ".png", ".ppt", ".pptx", ".rar", ".svg", ".tar", ".tgz", ".txt", ".wav", ".webp",
        ".xls", ".xlsx", ".xml", ".zip",
    }
)  # fmt: skip


def _normalize_percent_encoding(text: str) -> str:
    """Decode %XX escapes of unreserved characters and upper-case the rest."""

    def replace(match: re.Match[str]) -> str:
        char = chr(int(match.group(1), 16))
        return char if char in _UNRESERVED else "%" + match.group(1).upper()

    return _PERCENT_RE.sub(replace, text)


def _remove_dot_segments(path: str) -> str:
    """Resolve '.' and '..' path segments (RFC 3986, section 5.2.4)."""
    segments = path.split("/")
    output: list[str] = []
    for segment in segments:
        if segment == ".":
            continue
        if segment == "..":
            if len(output) > 1:
                output.pop()
            continue
        output.append(segment)
    # A path ending in '.' or '..' refers to a directory.
    if segments[-1] in (".", ".."):
        output.append("")
    return "/".join(output)


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Return the canonical form of a URL, or None if it cannot be crawled.

    Resolves relative URLs against `base`, rejects schemes other than HTTP(S),
    drops the fragment, credentials, default port and tracking parameters, and
    sorts the query string.

    Example:
        >>> canonicalize("#history", base="https://example.com/")
        'https://example.com/'
        >>> canonicalize("mailto:info@example.com", base="https://example.com/")
    """
    url = url.strip()
    if not url:
        return None
    try:
        parts = urlsplit(urljoin(base, url) if base else url)
        port = parts.port
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None

    host = parts.hostname.lower().rstrip(".")
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = host if port is None or port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"

    path = _normalize_percent_encoding(quote(parts.path, safe=_PATH_SAFE))
    path = _remove_dot_segments(path) or "/"
    if not path.startswith("/"):
        path = "/" + path

    params = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    query = urlencode(sorted(params))

    return urlunsplit((scheme, netloc, path, query, ""))


def has_skipped_extension(url: str) -> bool:
    """Tell whether the URL, judging by its extension, is not a page."""
    path = urlsplit(url).path.lower()
    dot = path.rfind(".")
    return dot > path.rfind("/") and path[dot:] in _SKIPPED_EXTENSIONS


def origin_of(url: str) -> str:
    """Return the scheme and host of a canonical URL, e.g. 'https://example.com'."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True, slots=True)
class Scope:
    """The URLs the crawler is allowed to visit.

    Attributes:
        prefixes: A URL is in scope only if it starts with one of these.
        excluded: A URL that starts with one of these is out of scope even so.
    """

    prefixes: tuple[str, ...]
    excluded: tuple[str, ...] = ()

    @classmethod
    def from_seeds(
        cls,
        seeds: tuple[str, ...],
        prefixes: tuple[str, ...] = (),
        excluded: tuple[str, ...] = (),
    ) -> Self:
        """Build the scope from explicit prefixes or, failing that, from the seed hosts."""
        sources = prefixes or tuple(origin_of(seed) + "/" for seed in seeds)
        canonical = tuple(dict.fromkeys(filter(None, map(canonicalize, sources))))
        if not canonical:
            raise ValueError("No valid URL prefix to bound the crawl")
        canonical_excluded = tuple(canonicalize(prefix) for prefix in excluded)
        if None in canonical_excluded:
            raise ValueError(f"Excluded prefixes must be absolute HTTP(S) URLs; got {excluded}")
        return cls(canonical, tuple(prefix for prefix in canonical_excluded if prefix))

    def contains(self, url: str) -> bool:
        return url.startswith(self.prefixes) and not url.startswith(self.excluded)
