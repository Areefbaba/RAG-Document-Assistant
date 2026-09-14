from __future__ import annotations

import re
from functools import lru_cache
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


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "qwen3:4b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
FETCH_K = 12
MMR_LAMBDA = 0.5
PDF_RELEVANCE_THRESHOLD = 1.0


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


def create_pdf_chain(
    retriever,
    llm_name: str = DEFAULT_LLM,
):
    llm = create_llm(llm_name)

    return (
        {
            "context": retriever | format_docs,
            "question": lambda question: question,
        }
        | PDF_PROMPT
        | llm
        | StrOutputParser()
    )


@lru_cache(maxsize=4)
def create_llm(llm_name: str = DEFAULT_LLM) -> ChatOllama:
    return ChatOllama(
        model=llm_name,
        temperature=0.1,
    )


def check_pdf_relevance(
    question: str,
    vector_store: FAISS,
    threshold: float = PDF_RELEVANCE_THRESHOLD,
) -> tuple[bool, list[Document]]:

    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    if not results:
        return False, []

    documents = [document for document, _ in results]
    scores = [score for _, score in results]
    best_score = min(scores)

    return best_score <= threshold, documents


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
    retriever,
    llm_name: str = DEFAULT_LLM,
) -> tuple[str, list[Document]]:

    docs = retriever.invoke(question)

    if not docs:
        return (
            "I don't know based on the provided documents.",
            [],
        )

    chain = create_pdf_chain(
        retriever,
        llm_name,
    )

    answer = chain.invoke(question)

    return answer, docs


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
        status_callback("pdf_search")

    pdf_relevant, matched_docs = check_pdf_relevance(
        question,
        vector_store,
    )

    if pdf_relevant:
        if status_callback:
            status_callback("pdf_answer")

        answer, docs = answer_from_pdf(
            question,
            retriever,
            llm_name,
        )

        return {
            "answer": answer,
            "source_type": "pdf",
            "documents": docs or matched_docs,
            "web_sources": [],
        }

    if status_callback:
        status_callback("web_search")

    answer, web_sources = answer_from_web(question)

    if status_callback:
        status_callback("web_done")

    return {
        "answer": answer,
        "source_type": "web",
        "documents": [],
        "web_sources": web_sources,
    }
