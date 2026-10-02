"""Reading what a language model replies."""

import json
import re

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n(.*)\n```$", re.S)


def parse_json_object(reply: str) -> dict[str, object]:
    """Read a reply that should be one JSON object.

    Models asked for JSON sometimes wrap it in a Markdown code block; the
    wrapping is removed.

    Raises:
        ValueError: If the reply is not a JSON object.
    """
    text = reply.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The reply is not JSON: {error}") from None
    if not isinstance(value, dict):
        raise ValueError("The reply is JSON, but not an object")
    return value
