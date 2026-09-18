from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import (
    DEFAULT_LLM,
    answer_question,
    build_vector_store,
    get_retriever,
    load_pdfs,
    split_documents,
)


app = FastAPI(
    title="RAG Document Assistant API",
    version="1.0.0",
    description="FastAPI backend for PDF RAG with web fallback.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    llm_name: str = DEFAULT_LLM


vector_store = None
retriever = None
loaded_files: list[str] = []
page_count = 0
chunk_count = 0


def pdf_sources(documents) -> list[dict]:
    sources = []
    seen = set()

    for document in documents or []:
        filename = document.metadata.get("source_file", "document.pdf")
        page = int(document.metadata.get("page", 0)) + 1
        key = (filename, page)

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "file": filename,
                "page": page,
            }
        )

    return sources


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": bool(loaded_files),
        "files": loaded_files,
    }


@app.get("/documents/status")
def documents_status():
    return {
        "loaded": bool(loaded_files),
        "files": loaded_files,
        "pages": page_count,
        "chunks": chunk_count,
    }


@app.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
):
    global vector_store, retriever
    global loaded_files, page_count, chunk_count

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF.",
        )

    temp_paths = []
    original_names = []

    try:
        for uploaded_file in files:
            filename = uploaded_file.filename or ""

            if not filename:
                continue

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is not a PDF.",
                )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:
                temp_file.write(await uploaded_file.read())
                temp_paths.append(temp_file.name)
                original_names.append(filename)

        documents = load_pdfs(temp_paths)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the uploaded PDFs.",
            )

        for document, fallback_name in zip(
            documents,
            original_names,
            strict=False,
        ):
            document.metadata.setdefault(
                "source_file",
                fallback_name,
            )

        chunks = split_documents(documents)
        new_vector_store = build_vector_store(chunks)
        new_retriever = get_retriever(new_vector_store)

        vector_store = new_vector_store
        retriever = new_retriever
        loaded_files = original_names
        page_count = len(documents)
        chunk_count = len(chunks)

        return {
            "message": "Documents processed successfully.",
            "files": loaded_files,
            "pages": page_count,
            "chunks": chunk_count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        ) from exc

    finally:
        for path in temp_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/chat")
def chat(request: ChatRequest):
    if vector_store is None or retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents are loaded. Upload PDF documents first.",
        )

    try:
        result = answer_question(
            question=request.question,
            vector_store=vector_store,
            retriever=retriever,
            llm_name=request.llm_name,
        )

        return {
            "answer": result["answer"],
            "source_type": result["source_type"],
            "documents": pdf_sources(
                result.get("documents", [])
            ),
            "web_sources": result.get(
                "web_sources",
                [],
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {exc}",
        ) from exc
