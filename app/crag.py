from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


@dataclass
class RetrievalEvaluation:
    status: str
    best_score: float
    documents: list[Document]


def evaluate_retrieval(
    question: str,
    vector_store: FAISS,
    top_k: int = 4,
    threshold: float = 1.0,
) -> RetrievalEvaluation:
    results = vector_store.similarity_search_with_score(
        question,
        k=top_k,
    )

    if not results:
        return RetrievalEvaluation(
            status="low",
            best_score=float("inf"),
            documents=[],
        )

    documents = [document for document, _ in results]
    scores = [score for _, score in results]
    best_score = min(scores)

    high_confidence = threshold * 0.75

    if best_score <= high_confidence:
        status = "high"
    elif best_score <= threshold:
        status = "medium"
    else:
        status = "low"

    return RetrievalEvaluation(
        status=status,
        best_score=best_score,
        documents=documents,
    )


def web_correction_needed(
    evaluation: RetrievalEvaluation,
) -> bool:
    return evaluation.status == "medium"
