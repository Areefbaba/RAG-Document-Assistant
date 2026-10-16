from __future__ import annotations

import re
from typing import Callable, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

# Connected pipeline parts:
from app.ingestion import clean_text, load_pdfs
from app.chunking import CHUNK_SIZE, CHUNK_OVERLAP, split_documents
from app.retrieval import (
    EMBEDDING_MODEL,
    TOP_K,
    FETCH_K,
    MMR_LAMBDA,
    PDF_RELEVANCE_THRESHOLD,
    create_embeddings,
    build_vector_store,
    get_retriever,
    format_docs,
)
from app.generation import (
    DEFAULT_LLM,
    PDF_PROMPT,
    HYBRID_PROMPT,
    create_llm,
    generate_grounded_answer,
    web_search,
    answer_from_web,
    answer_from_hybrid,
    answer_from_pdf,
    check_and_revise_answer,
)
from app.crag import evaluate_retrieval
from app.self_rag import (
    check_answer_support,
    check_answer_useful,
    revise_answer as revise_grounded_answer,
    rewrite_question as rewrite_retrieval_query,
)

MAX_REWRITES = 1


class RAGState(TypedDict, total=False):
    question: str
    original_question: str
    route: str
    retrieval_status: str
    best_score: float
    documents: list[Document]
    web_sources: list[dict]
    context: str
    answer: str
    source_type: str
    support_ok: bool
    usefulness_ok: bool
    rewrite_count: int


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


def build_rag_graph(
    vector_store: FAISS,
    retriever,
    llm_name: str = DEFAULT_LLM,
    status_callback: Callable[[str], None] | None = None,
):
    def status(name: str) -> None:
        if status_callback:
            status_callback(name)

    def decide_retrieval(state: RAGState):
        status("decide_retrieval")

        if _is_casual_query(state["question"]):
            return {"route": "direct"}

        return {"route": "retrieve"}

    def generate_direct(state: RAGState):
        status("generate_direct")

        return {
            "answer": (
                "I'm ready to help. Upload a document and ask a question "
                "about its contents."
            ),
            "source_type": "direct",
        }

    def retrieve(state: RAGState):
        status("retrieve")

        evaluation = evaluate_retrieval(
            state["question"],
            vector_store,
            top_k=TOP_K,
            threshold=PDF_RELEVANCE_THRESHOLD,
        )

        documents = retriever.invoke(state["question"])
        web_sources = []

        if evaluation.status == "medium":
            status("crag_correct")
            web_sources = web_search(state["question"])

        return {
            "retrieval_status": evaluation.status,
            "best_score": evaluation.best_score,
            "documents": documents or evaluation.documents,
            "web_sources": web_sources,
        }

    def is_relevant(state: RAGState):
        status("is_relevant")

        return {
            "route": (
                "relevant"
                if state.get("retrieval_status") != "low"
                else "not_relevant"
            )
        }

    def generate_from_context(state: RAGState):
        status("generate_from_context")

        documents = state.get("documents", [])
        web_sources = state.get("web_sources", [])
        question = state["question"]

        pdf_context = format_docs(documents)

        if web_sources:
            answer = answer_from_hybrid(
                question,
                documents,
                web_sources,
                llm_name,
            )

            context = (
                pdf_context
                + "\n\n"
                + "\n\n".join(
                    item.get("snippet", "")
                    for item in web_sources
                )
            )

            source_type = "hybrid"

        else:
            answer = generate_grounded_answer(
                question,
                documents,
                llm_name,
            )

            context = pdf_context
            source_type = "pdf"

        return {
            "answer": answer,
            "context": context,
            "source_type": source_type,
        }

    def is_sup(state: RAGState):
        status("is_sup")

        supported, _ = check_answer_support(
            state["question"],
            state.get("answer", ""),
            state.get("context", ""),
            create_llm(llm_name),
        )

        return {
            "support_ok": supported,
            "route": "supported" if supported else "unsupported",
        }

    def revise_answer(state: RAGState):
        status("revise_answer")

        answer = revise_grounded_answer(
            state["question"],
            state.get("answer", ""),
            state.get("context", ""),
            create_llm(llm_name),
        )

        return {
            "answer": answer,
            "support_ok": True,
        }

    def is_use(state: RAGState):
        status("is_use")

        useful, _ = check_answer_useful(
            state["question"],
            state.get("answer", ""),
            create_llm(llm_name),
        )

        rewrite_count = state.get("rewrite_count", 0)

        if useful:
            route = "end"
        elif rewrite_count < MAX_REWRITES:
            route = "rewrite"
        else:
            route = "no_answer"

        return {
            "usefulness_ok": useful,
            "route": route,
        }

    def rewrite_question(state: RAGState):
        status("rewrite_question")

        rewritten = rewrite_retrieval_query(
            state["question"],
            create_llm(llm_name),
        )

        return {
            "question": rewritten,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def no_answer_found(state: RAGState):
        status("no_answer_found")

        answer, sources = answer_from_web(
            state["original_question"],
        )

        return {
            "answer": answer,
            "source_type": "web",
            "web_sources": sources,
        }

    graph = StateGraph(RAGState)

    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("retrieve", retrieve)
    graph.add_node("is_relevant", is_relevant)
    graph.add_node("generate_from_context", generate_from_context)
    graph.add_node("is_sup", is_sup)
    graph.add_node("revise_answer", revise_answer)
    graph.add_node("is_use", is_use)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("no_answer_found", no_answer_found)
    graph.add_node("generate_direct", generate_direct)

    graph.add_edge(START, "decide_retrieval")

    graph.add_conditional_edges(
        "decide_retrieval",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "direct": "generate_direct",
        },
    )

    graph.add_edge("retrieve", "is_relevant")

    graph.add_conditional_edges(
        "is_relevant",
        lambda state: state["route"],
        {
            "relevant": "generate_from_context",
            "not_relevant": "no_answer_found",
        },
    )

    graph.add_edge("generate_from_context", "is_sup")

    graph.add_conditional_edges(
        "is_sup",
        lambda state: state["route"],
        {
            "supported": "is_use",
            "unsupported": "revise_answer",
        },
    )

    graph.add_edge("revise_answer", "is_use")

    graph.add_conditional_edges(
        "is_use",
        lambda state: state["route"],
        {
            "end": END,
            "rewrite": "rewrite_question",
            "no_answer": "no_answer_found",
        },
    )

    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("no_answer_found", END)
    graph.add_edge("generate_direct", END)

    return graph.compile()


def answer_question(
    question: str,
    vector_store: FAISS,
    retriever,
    llm_name: str = DEFAULT_LLM,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "source_type": None,
            "documents": [],
            "web_sources": [],
        }

    graph = build_rag_graph(
        vector_store=vector_store,
        retriever=retriever,
        llm_name=llm_name,
        status_callback=status_callback,
    )

    result = graph.invoke(
        {
            "question": question,
            "original_question": question,
            "rewrite_count": 0,
        }
    )

    return {
        "answer": result.get(
            "answer",
            "I couldn't generate an answer.",
        ),
        "source_type": result.get(
            "source_type",
            "pdf",
        ),
        "documents": result.get(
            "documents",
            [],
        ),
        "web_sources": result.get(
            "web_sources",
            [],
        ),
    }
