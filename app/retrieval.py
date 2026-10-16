from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
FETCH_K = 12
MMR_LAMBDA = 0.5
PDF_RELEVANCE_THRESHOLD = 1.0


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
