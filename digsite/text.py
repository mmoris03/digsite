import re

def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    text -> [token, ...]
    """
    return re.findall(r"\w+", text.casefold())
