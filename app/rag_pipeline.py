from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
from typing import Callable, Iterable, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from app.crag import evaluate_retrieval
from app.self_rag import (
    check_answer_support,
    check_answer_useful,
    revise_answer as revise_grounded_answer,
    rewrite_question as rewrite_retrieval_query,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "qwen3:4b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
FETCH_K = 12
MMR_LAMBDA = 0.5
PDF_RELEVANCE_THRESHOLD = 1.0
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


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs(paths: Iterable[str | Path]) -> list[Document]:
    documents = []

    for path in paths:
        path = Path(path)

        pages = PyPDFLoader(
            str(path),
            extraction_mode="layout",
        ).load()

        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_file"] = path.name

            if page.page_content:
                documents.append(page)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def build_vector_store(chunks: list[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No document chunks were created.")

    return FAISS.from_documents(
        chunks,
        create_embeddings(),
    )


def get_retriever(vector_store: FAISS):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )


def format_docs(documents: list[Document]) -> str:
    parts = []

    for document in documents:
        page = int(document.metadata.get("page", 0)) + 1
        filename = document.metadata.get(
            "source_file",
            "document.pdf",
        )

        parts.append(
            f"[Source: {filename}, Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(parts)


PDF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grounded PDF question-answering assistant.

Answer using ONLY the provided document context.

Rules:
- Do not use outside knowledge.
- Do not invent facts, numbers, or citations.
- Use only information supported by the context.
- If the context does not contain enough information, reply exactly:
"I don't know based on the provided documents."
- Keep the answer concise and useful.
- Cite supporting information as [filename, Page N].

Context:

{context}
""",
        ),
        ("human", "{question}"),
    ]
)


HYBRID_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grounded question-answering assistant.

Use the supplied PDF context and web search context to answer the question.

Rules:
- Prefer the uploaded PDF when it directly supports the answer.
- Use web information only to fill gaps in retrieval.
- Do not invent facts.
- Keep the answer concise.
- For PDF claims, cite [filename, Page N].
""",
        ),
        ("human", "Question: {question}\n\nPDF Context:\n{pdf_context}\n\nWeb Context:\n{web_context}"),
    ]
)


@lru_cache(maxsize=4)
def create_llm(llm_name: str = DEFAULT_LLM) -> ChatOllama:
    return ChatOllama(
        model=llm_name,
        temperature=0.1,
    )


def generate_grounded_answer(
    question: str,
    documents: list[Document],
    llm_name: str = DEFAULT_LLM,
) -> str:
    context = format_docs(documents)

    return (
        {
            "context": lambda _: context,
            "question": lambda _: question,
        }
        | PDF_PROMPT
        | create_llm(llm_name)
        | StrOutputParser()
    ).invoke(question).strip()


def web_search(
    question: str,
    max_results: int = 5,
) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []

    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                question,
                max_results=max_results,
            )

            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    }
                )

        return results

    except Exception:
        return []


def answer_from_web(
    question: str,
) -> tuple[str, list[dict]]:
    results = web_search(question)

    if not results:
        return (
            "The answer wasn't found in the uploaded document, "
            "and I couldn't find relevant information on the web.",
            [],
        )

    answer_parts = [
        "The answer wasn't found in the uploaded document, "
        "so I searched the web for relevant information.",
        "",
        "### 🌐 Web Search Results",
    ]

    for index, item in enumerate(results, 1):
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()

        if not snippet:
            continue

        answer_parts.append(
            f"**{index}. {title}**\n\n{snippet}"
        )

    return "\n\n".join(answer_parts), results


def answer_from_hybrid(
    question: str,
    pdf_documents: list[Document],
    web_sources: list[dict],
    llm_name: str = DEFAULT_LLM,
) -> str:
    pdf_context = format_docs(pdf_documents)

    web_context = "\n\n".join(
        f"[Web Source {index}]\n"
        f"Title: {item.get('title', '')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Content: {item.get('snippet', '')}"
        for index, item in enumerate(web_sources, 1)
    )

    return (
        HYBRID_PROMPT
        | create_llm(llm_name)
        | StrOutputParser()
    ).invoke(
        {
            "question": question,
            "pdf_context": pdf_context,
            "web_context": web_context,
        }
    ).strip()


def answer_from_pdf(
    question: str,
    retriever,
    llm_name: str = DEFAULT_LLM,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, list[Document]]:
    documents = retriever.invoke(question)

    if not documents:
        return (
            "I don't know based on the provided documents.",
            [],
        )

    context = format_docs(documents)
    answer = generate_grounded_answer(
        question=question,
        documents=documents,
        llm_name=llm_name,
    )

    answer = check_and_revise_answer(
        question,
        answer,
        context,
        llm_name,
        status_callback,
    )

    return answer, documents


def check_and_revise_answer(
    question: str,
    answer: str,
    context: str,
    llm_name: str,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    llm = create_llm(llm_name)

    if status_callback:
        status_callback("is_sup")

    supported, _ = check_answer_support(
        question,
        answer,
        context,
        llm,
    )

    if supported:
        return answer

    if status_callback:
        status_callback("revise_answer")

    return revise_grounded_answer(
        question,
        answer,
        context,
        llm,
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
