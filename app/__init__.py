from __future__ import annotations

import re
from typing import Callable

from app.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
from app.crag import (
    RetrievalEvaluation,
    evaluate_retrieval,
    web_correction_needed,
)
from app.generation import (
    DEFAULT_LLM,
    HYBRID_PROMPT,
    PDF_PROMPT,
    answer_from_hybrid,
    answer_from_pdf,
    answer_from_web,
    check_and_revise_answer,
    create_llm,
    generate_grounded_answer,
    web_search,
)
from app.loading_data import clean_text, load_pdfs
from app.reranking import rerank_documents
from app.retrieval import (
    EMBEDDING_MODEL,
    FETCH_K,
    MMR_LAMBDA,
    PDF_RELEVANCE_THRESHOLD,
    TOP_K,
    build_vector_store,
    create_embeddings,
    format_docs,
    get_retriever,
)
from app.self_rag import (
    REVISE_PROMPT,
    REWRITE_PROMPT,
    SUPPORT_PROMPT,
    USEFULNESS_PROMPT,
    check_answer_support,
    check_answer_useful,
    revise_answer,
    rewrite_question,
    self_reflect_and_correct,
)


def _is_casual_query(question: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", "", question.lower()).strip()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
    }


def answer_question(
    question: str,
    vector_store,
    retriever,
    llm_name: str = DEFAULT_LLM,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Answer question by coordinating retrieval, reranking, crag, generation, and self-rag."""
    query = question.strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "source_type": None,
            "documents": [],
            "web_sources": [],
        }

    if status_callback:
        status_callback("decide_retrieval")

    if _is_casual_query(query):
        if status_callback:
            status_callback("generate_direct")
        return {
            "answer": "I'm ready to help. Upload a document and ask a question about its contents.",
            "source_type": "direct",
            "documents": [],
            "web_sources": [],
        }

    # Step 1: Retrieval
    if status_callback:
        status_callback("retrieve")
    retrieved_docs = retriever.invoke(query)

    # Step 2: Reranking
    reranked_docs = rerank_documents(query, retrieved_docs, top_n=TOP_K)

    # Step 3: CRAG Evaluation
    if status_callback:
        status_callback("is_relevant")
    evaluation = evaluate_retrieval(
        query,
        vector_store,
        top_k=TOP_K,
        threshold=PDF_RELEVANCE_THRESHOLD,
    )

    documents = reranked_docs or evaluation.documents
    web_sources = []

    # Routing based on CRAG evaluation
    if evaluation.status == "low" or not documents:
        if status_callback:
            status_callback("no_answer_found")
        answer, web_sources = answer_from_web(query)
        return {
            "answer": answer,
            "source_type": "web",
            "documents": [],
            "web_sources": web_sources,
        }

    if evaluation.status == "medium":
        if status_callback:
            status_callback("crag_correct")
        web_sources = web_search(query)

    # Step 4: Generation
    if status_callback:
        status_callback("generate_from_context")

    if web_sources:
        answer = answer_from_hybrid(query, documents, web_sources, llm_name=llm_name)
        source_type = "hybrid"
    else:
        answer = generate_grounded_answer(query, documents, llm_name=llm_name)
        source_type = "pdf"

    # Step 5: Self-RAG Support Check & Revision
    if status_callback:
        status_callback("is_sup")

    llm = create_llm(llm_name)
    context = format_docs(documents)
    supported, _ = check_answer_support(query, answer, context, llm)

    if not supported:
        if status_callback:
            status_callback("revise_answer")
        answer = revise_answer(query, answer, context, llm)

    # Step 6: Self-RAG Usefulness Check & Query Rewrite (if not useful)
    if status_callback:
        status_callback("is_use")
    useful, _ = check_answer_useful(query, answer, llm)

    if not useful:
        if status_callback:
            status_callback("rewrite_question")
        rewritten_q = rewrite_question(query, llm)
        new_docs = retriever.invoke(rewritten_q)
        if new_docs:
            documents = rerank_documents(rewritten_q, new_docs, top_n=TOP_K)
            answer = generate_grounded_answer(rewritten_q, documents, llm_name=llm_name)

    return {
        "answer": answer,
        "source_type": source_type,
        "documents": documents,
        "web_sources": web_sources,
    }


__all__ = [
    # Loading data
    "clean_text",
    "load_pdfs",
    # Chunking
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "split_documents",
    # Retrieval
    "EMBEDDING_MODEL",
    "TOP_K",
    "FETCH_K",
    "MMR_LAMBDA",
    "PDF_RELEVANCE_THRESHOLD",
    "create_embeddings",
    "build_vector_store",
    "get_retriever",
    "format_docs",
    # Reranking
    "rerank_documents",
    # CRAG
    "evaluate_retrieval",
    "web_correction_needed",
    "RetrievalEvaluation",
    # Generation
    "DEFAULT_LLM",
    "create_llm",
    "generate_grounded_answer",
    "answer_from_hybrid",
    "answer_from_web",
    "answer_from_pdf",
    "check_and_revise_answer",
    "web_search",
    "PDF_PROMPT",
    "HYBRID_PROMPT",
    # Self-RAG
    "check_answer_support",
    "check_answer_useful",
    "revise_answer",
    "rewrite_question",
    "self_reflect_and_correct",
    "SUPPORT_PROMPT",
    "USEFULNESS_PROMPT",
    "REVISE_PROMPT",
    "REWRITE_PROMPT",
    # Main QA coordination
    "answer_question",
]
