"""Semantic similarity via sentence-transformers embeddings.

Used as an alternative to TF-IDF for measuring how close a generated
description is to the dataset's official description. Embeddings capture
meaning (semantic similarity) rather than just shared words (lexical overlap).
"""
from __future__ import annotations

import numpy as np

import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def encode(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    return model.encode(list(texts), batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)


def cosine_from_embeddings(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equal-length embedding matrices.

    Embeddings are L2-normalized on encode, so cosine == dot product.
    """
    return (a * b).sum(axis=1)
