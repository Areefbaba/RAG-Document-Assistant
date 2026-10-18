from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = DEFAULT_RERANKER_MODEL):
    """Lazily load and cache the cross-encoder reranker model."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)
    except Exception as exc:
        logger.warning(f"Could not load CrossEncoder ({model_name}): {exc}. Falling back to lexical reranker.")
        return None


def _lexical_similarity_score(query: str, text: str) -> float:
    """Compute normalized token overlap and frequency score for fallback reranking."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return 0.0

    text_lower = text.lower()
    text_tokens = re.findall(r"\w+", text_lower)
    if not text_tokens:
        return 0.0

    matches = sum(1 for token in text_tokens if token in query_tokens)
    overlap = len(query_tokens.intersection(set(text_tokens))) / len(query_tokens)

    # Combined term frequency and token coverage score
    return overlap * 0.7 + (matches / len(text_tokens)) * 0.3


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = 4,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[Document]:
    """Rerank retrieved document chunks by relevance to the query.
    
    Uses CrossEncoder when available, with automatic fallback to
    lexical-semantic overlap scoring.
    
    Args:
        query: User question string.
        documents: Candidate retrieved documents from vector search.
        top_n: Number of top reranked documents to return.
        model_name: CrossEncoder model identifier.
        
    Returns:
        List of reranked Document instances sorted by highest relevance.
    """
    if not documents:
        return []

    doc_list = list(documents)
    if len(doc_list) <= 1:
        return doc_list[:top_n]

    encoder = get_cross_encoder(model_name)

    if encoder is not None:
        try:
            pairs = [[query, doc.page_content] for doc in doc_list]
            scores = encoder.predict(pairs)

            scored_docs = []
            for doc, score in zip(doc_list, scores):
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": float(score)},
                )
                scored_docs.append((doc_copy, float(score)))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_n]]

        except Exception as exc:
            logger.warning(f"CrossEncoder inference failed: {exc}. Using fallback reranker.")

    # Fallback lexical reranker
    scored_docs = []
    for doc in doc_list:
        score = _lexical_similarity_score(query, doc.page_content)
        doc_copy = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": round(score, 4)},
        )
        scored_docs.append((doc_copy, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]
