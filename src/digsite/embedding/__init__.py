"""Embedding models: text to vectors."""

from digsite.embedding.base import Embedder, Vector, Vectors
from digsite.embedding.hashing import HashingEmbedder

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

__all__ = ["DEFAULT_MODEL", "Embedder", "HashingEmbedder", "Vector", "Vectors", "create_embedder"]


def create_embedder(name: str = DEFAULT_MODEL) -> Embedder:
    """Build the embedder with the given name.

    `hashing` or `hashing-<dimension>` gives the model-free embedder. Any other
    name is taken as a sentence-transformers model.
    """
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("hashing-") and name.removeprefix("hashing-").isdigit():
        return HashingEmbedder(int(name.removeprefix("hashing-")))

    from digsite.embedding.sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(name)
