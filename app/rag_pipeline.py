from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
from typing import Callable, Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from app.crag import evaluate_retrieval
from app.self_rag import self_reflect_and_correct


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "qwen3:4b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
FETCH_K = 12
MMR_LAMBDA = 0.5
PDF_RELEVANCE_THRESHOLD = 1.0
SELF_RAG_MAX_RETRIES = 1


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
- Use web information only for gaps or corrections.
- Do not invent facts.
- Keep the answer concise.
- For PDF claims, cite [filename, Page N].
- For web claims, do not invent URLs or source details.

PDF Context:
{pdf_context}

Web Context:
{web_context}
""",
        ),
        ("human", "{question}"),
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
    status_callback: Callable[[str], None] | None = None,
) -> str:
    context = format_docs(documents)

    answer = (
        {
            "context": lambda _: context,
            "question": lambda _: question,
        }
        | PDF_PROMPT
        | create_llm(llm_name)
        | StrOutputParser()
    ).invoke(question).strip()

    return self_reflect_and_correct(
        question=question,
        answer=answer,
        context=context,
        llm=create_llm(llm_name),
        max_retries=SELF_RAG_MAX_RETRIES,
        status_callback=status_callback,
    )


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


def answer_from_pdf(
    question: str,
    documents: list[Document],
    llm_name: str = DEFAULT_LLM,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    return generate_grounded_answer(
        question=question,
        documents=documents,
        llm_name=llm_name,
        status_callback=status_callback,
    )


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

    if status_callback:
        status_callback("crag_evaluate")

    evaluation = evaluate_retrieval(
        question,
        vector_store,
        top_k=TOP_K,
        threshold=PDF_RELEVANCE_THRESHOLD,
    )

    if evaluation.status == "high":
        if status_callback:
            status_callback("pdf_answer")

        answer = answer_from_pdf(
            question=question,
            documents=retriever.invoke(question),
            llm_name=llm_name,
            status_callback=status_callback,
        )

        return {
            "answer": answer,
            "source_type": "pdf",
            "documents": evaluation.documents,
            "web_sources": [],
        }

    if status_callback:
        status_callback("crag_correct")

    web_sources = web_search(question)

    if evaluation.status == "medium" and web_sources:
        answer = answer_from_hybrid(
            question=question,
            pdf_documents=evaluation.documents,
            web_sources=web_sources,
            llm_name=llm_name,
        )

        return {
            "answer": (
                "The document contained related information, "
                "so I checked the web to improve the retrieved context.\n\n"
                + answer
            ),
            "source_type": "hybrid",
            "documents": evaluation.documents,
            "web_sources": web_sources,
        }

    if status_callback:
        status_callback("web_search")

    if not web_sources:
        return {
            "answer": (
                "The answer wasn't found in the uploaded document, "
                "and I couldn't find relevant information on the web."
            ),
            "source_type": "web",
            "documents": [],
            "web_sources": [],
        }

    answer, web_sources = answer_from_web(question)

    if status_callback:
        status_callback("web_done")

    return {
        "answer": answer,
        "source_type": "web",
        "documents": [],
        "web_sources": web_sources,
    }
