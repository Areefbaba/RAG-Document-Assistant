# LOCAL RAG DOCUMENT ASSISTANT

An end-to-end Retrieval-Augmented Generation (RAG) application for grounded PDF question answering with local LLM inference, intelligent retrieval routing, answer verification, and web fallback.

## Overview

RAG Document Assistant allows users to upload PDF documents and ask questions about their content.

The system:
- Extracts and cleans PDF text
- Splits documents into overlapping chunks
- Generates semantic embeddings
- Stores document vectors in FAISS
- Retrieves relevant and diverse context using MMR
- Generates grounded answers using Qwen3 through Ollama
- Evaluates retrieval quality using a CRAG-style routing layer
- Verifies generated answers using Self-RAG-style support and usefulness checks
- Revises unsupported answers
- Rewrites queries when retrieval or answer quality is insufficient
- Falls back to web search when the uploaded documents do not contain enough information
- Returns document and page-level source information

## Architecture

```text
User
  │
  ▼
Streamlit Frontend
  │ HTTP
  ▼
FastAPI Backend
  │
  ▼
PDF Processing → Cleaning → Chunking → Embeddings → FAISS
                                             │
                                             ▼
                                        MMR Retrieval
                                             │
                                             ▼
                                  CRAG-style Evaluation
                                      ┌──────┴──────┐
                                      │             │
                                     PDF           Web
                                      │             │
                                      └──────┬──────┘
                                             ▼
                                      Qwen3 / Ollama
                                             │
                                             ▼
                                  Self-RAG-style Checks
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                              Supported            Unsupported
                                  │                     │
                                  │                  Revise
                                  └──────────┬──────────┘
                                             ▼
                                      Usefulness Check
                                             │
                                    ┌────────┴────────┐
                                    │                 │
                                  Useful          Not Useful
                                    │                 │
                                   END          Rewrite Query
                                                     │
                                                     ▼
                                                 Retrieval
```

## Core Technologies

- Python
- LangChain
- LangGraph
- FAISS
- Hugging Face Sentence Transformers
- Qwen3
- Ollama
- FastAPI
- Streamlit
- PyPDF
- DDGS Web Search
- Pydantic
- Pytest

## Project Structure

```text
RAG-Document-Assistant/
│
├── app/
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── self_rag.py
│   └── crag.py
│
├── backend/
│   └── main.py
│
├── evaluation/
│   └── test_cases.json
│
├── requirements.txt
└── README.md
```

## RAG Pipeline

### 1. Document Ingestion

Uploaded PDFs are processed using `PyPDFLoader`. Extracted text is cleaned and normalized before retrieval preparation.

### 2. Chunking

Documents are split using `RecursiveCharacterTextSplitter` with:

- Chunk size: `800`
- Chunk overlap: `150`

### 3. Embeddings

The system uses `sentence-transformers/all-MiniLM-L6-v2` to convert document chunks and queries into dense vector representations.

### 4. Vector Search

Embeddings are stored in a FAISS index. The retriever uses Maximal Marginal Relevance (MMR) to balance semantic relevance and result diversity.

Configuration:

```text
Top K: 4
Fetch K: 12
MMR Lambda: 0.5
```

### 5. Grounded Generation

Relevant document chunks are injected into a constrained prompt. The LLM is instructed to use only retrieved context, avoid unsupported claims, avoid fabricated citations, and return an explicit uncertainty response when the documents do not contain the answer.

## CRAG-Style Retrieval Routing

The application evaluates retrieval quality using FAISS similarity scores and classifies retrieval into high, medium, or low confidence.

```text
High
  → Answer using document context

Medium
  → Perform web correction
  → Generate using PDF + web context

Low
  → Use web fallback
```

This is a practical CRAG-style implementation focused on retrieval evaluation and corrective routing.

## Self-RAG-Style Answer Verification

After answer generation, the system evaluates the response before returning it.

### Support Evaluation

The system checks whether important factual claims in the answer are supported by the available context. Unsupported answers are sent through an answer revision step.

### Usefulness Evaluation

The system checks whether the answer directly addresses the user's question. If the response is not useful, the system can rewrite the query and perform another retrieval attempt.

The number of rewrite attempts is limited to prevent unnecessary loops and repeated inference.

## Web Fallback

When the uploaded documents do not provide sufficient information, the application can perform web search using DDGS.

The web path provides:
- Search result title
- URL
- Result snippet

For medium-confidence retrieval, web results can be combined with document context to create a hybrid answer.

## Local LLM

The application uses Qwen3 4B through Ollama. Using a locally served model keeps the core generation pipeline independent of a hosted LLM API and supports local document processing.

## Backend API

### Health Check

```http
GET /health
```

Returns backend and document-loading status.

### Document Status

```http
GET /documents/status
```

Returns currently loaded files, page count, and chunk count.

### Upload Documents

```http
POST /documents/upload
```

Processes one or more PDF files and builds the vector index.

### Chat

```http
POST /chat
```

Accepts a question and executes the RAG workflow.

Example request:

```json
{
  "question": "What is the bias-variance trade-off?"
}
```

## Frontend

The Streamlit interface provides PDF upload, document processing, conversational question answering, source type display, PDF page references, web source links, and backend connectivity status.

## Evaluation

The repository includes a basic evaluation test set covering document-grounded questions and questions that should route to web search.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama pull qwen3:4b
```

### Start FastAPI

```bash
uvicorn backend.main:app --reload
```

### Start Streamlit

In another terminal:

```bash
streamlit run app/app.py
```

## Key Features

- End-to-end PDF RAG pipeline
- Semantic vector retrieval with FAISS
- MMR-based diverse retrieval
- LangGraph stateful orchestration
- CRAG-style retrieval evaluation
- Self-RAG-style answer verification
- Answer revision
- Query rewriting
- PDF + web hybrid retrieval
- Web fallback for missing information
- Local Qwen3 inference with Ollama
- FastAPI backend
- Streamlit frontend
- Source and page traceability

## Current Limitations

- The FAISS index is currently maintained in process memory.
- Document state is lost when the backend restarts.
- The evaluation setup is currently a basic test set rather than a full RAG benchmarking framework.
- Scanned or image-only PDFs require an OCR pipeline that is not included in the current implementation.
- Web fallback currently exposes search results rather than implementing a full web-content extraction and synthesis pipeline.

## Future Improvements

- Persistent vector storage
- OCR for scanned documents
- Hybrid keyword + vector retrieval
- Cross-encoder reranking
- Automated RAG evaluation
- Streaming responses
- Authentication and multi-user document isolation
- Background document processing
- Production monitoring and observability

## Summary

RAG Document Assistant combines document retrieval, semantic search, local LLM generation, retrieval-quality routing, answer verification, and corrective workflows into a modular AI application.

The project demonstrates practical implementation of modern RAG and AI engineering concepts using LangChain, LangGraph, FAISS, FastAPI, Streamlit, and local LLM inference.
