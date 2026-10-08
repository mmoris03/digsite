import re

def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\w+", text.casefold())
