# LOCAL RAG DOCUMENT ASSISTANT

An end-to-end Retrieval-Augmented Generation (RAG) application for grounded PDF question answering with local LLM inference, Cross-Encoder reranking, Corrective RAG (CRAG) retrieval routing, Self-RAG answer verification, and live web fallback.

---

## 🏛️ Project Structure

The pipeline is organized into modular files inside `app/`:

```text
RAG-Document-Assistant/
├── app/
│   ├── app.py           # Streamlit frontend user interface
│   ├── loading_data.py  # Document loading (PyPDFLoader) & text cleaning
│   ├── chunking.py      # Text splitting & chunking configuration
│   ├── retrieval.py     # Embeddings, FAISS vector store & MMR retriever
│   ├── reranking.py     # Cross-Encoder & lexical relevance reranking
│   ├── crag.py          # Corrective RAG (CRAG) retrieval evaluation
│   ├── generation.py    # Grounded answer generation, hybrid synthesis & web search
│   ├── self_rag.py      # Self-RAG verification (support check, usefulness, rewrite)
│   └── __init__.py      # Package interface & answer_question coordinator
│
├── backend/
│   ├── main.py          # FastAPI backend REST API
│   └── __init__.py
│
├── evaluation/
│   └── test_cases.json  # Benchmark evaluation questions
│
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🔄 Modular Pipeline Components

### 1. Loading Data (`app/loading_data.py`)
- `clean_text`: Sanitizes extracted text, removing null bytes (`\x00`), non-breaking spaces (`\xa0`), and excessive whitespace.
- `load_pdfs`: Loads PDF files using `PyPDFLoader` with `layout` extraction mode and attaches page metadata.

### 2. Chunking (`app/chunking.py`)
- `CHUNK_SIZE = 800`
- `CHUNK_OVERLAP = 150`
- `split_documents`: Splits input documents using `RecursiveCharacterTextSplitter`.

### 3. Retrieval (`app/retrieval.py`)
- `create_embeddings`: Cached HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with normalization.
- `build_vector_store`: Builds in-memory FAISS vector index from document chunks.
- `get_retriever`: Maximal Marginal Relevance (MMR) retriever (`top_k=4`, `fetch_k=12`, `lambda_mult=0.5`).
- `format_docs`: Formats retrieved chunks with citation tags `[Source: filename, Page N]`.

### 4. Reranking (`app/reranking.py`)
- `rerank_documents`: Reranks retrieved document chunks using `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with automatic fallback to token overlap & term frequency scoring.

### 5. CRAG Layer (`app/crag.py`)
- `evaluate_retrieval`: Evaluates FAISS similarity distance scores into `high`, `medium`, or `low` confidence.
- `web_correction_needed`: Flags medium-confidence retrieval for web correction.

### 6. Generation (`app/generation.py`)
- `create_llm`: Cached ChatOllama client (`qwen3:4b`).
- `PDF_PROMPT` & `HYBRID_PROMPT`: Strict prompt templates prohibiting hallucinations.
- `generate_grounded_answer`: Generates answer strictly supported by document context.
- `web_search`: Live search using DuckDuckGo.
- `answer_from_hybrid`: Combines PDF context and web search results.
- `answer_from_web`: Web search fallback when documents lack information.
- `check_and_revise_answer`: Checks answer grounding and revises unsupported answers.

### 7. Self-RAG Layer (`app/self_rag.py`)
- `check_answer_support`: Verifies if every factual claim is grounded in the context.
- `revise_answer`: Corrects unsupported claims.
- `check_answer_useful`: Verifies whether the response directly addresses the question.
- `rewrite_question`: Rewrites user query for improved retrieval.

---

## 🚀 Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
ollama pull qwen3:4b
```

### 3. Start FastAPI Backend
```bash
uvicorn backend.main:app --reload
```

### 4. Start Streamlit Frontend
In a new terminal:
```bash
streamlit run app/app.py
```
