from __future__ import annotations

from functools import lru_cache
from typing import Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.retrieval import format_docs
from app.self_rag import (
    check_answer_support,
    revise_answer as revise_grounded_answer,
)

DEFAULT_LLM = "qwen3:4b"

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
