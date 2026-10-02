"""Per-site robots.txt lookup and caching."""

import logging
from urllib.robotparser import RobotFileParser

import httpx

from digsite.crawl.urls import origin_of

logger = logging.getLogger(__name__)

_ALLOW_ALL: list[str] = []
_DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


class RobotsCache:
    """Downloads each site's robots.txt once and answers questions about it.

    Follows RFC 9309: if the file does not exist (4xx) everything may be
    crawled; if the server is failing (5xx, 429 or a network error) nothing on
    that site is crawled.
    """

    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        return self._parser_for(url).can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """Return the Crawl-delay the site asks of this agent, if it declares one."""
        delay = self._parser_for(url).crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None

    def _parser_for(self, url: str) -> RobotFileParser:
        origin = origin_of(url)
        parser = self._parsers.get(origin)
        if parser is None:
            parser = self._fetch(origin)
            self._parsers[origin] = parser
        return parser

    def _fetch(self, origin: str) -> RobotFileParser:
        robots_url = origin + "/robots.txt"
        try:
            response = self._client.get(robots_url, follow_redirects=True)
        except httpx.HTTPError as exc:
            logger.warning(
                "robots.txt unreachable at %s (%s): site will not be crawled", origin, exc
            )
            return _parser_for_rules(robots_url, _DISALLOW_ALL)

        status = response.status_code
        if status >= 500 or status == 429:
            logger.warning("robots.txt returned %d at %s: site will not be crawled", status, origin)
            return _parser_for_rules(robots_url, _DISALLOW_ALL)
        if status >= 400:
            return _parser_for_rules(robots_url, _ALLOW_ALL)
        return _parser_for_rules(robots_url, response.text.splitlines())


def _parser_for_rules(robots_url: str, lines: list[str]) -> RobotFileParser:
    parser = RobotFileParser(robots_url)
    parser.parse(lines)
    return parser
