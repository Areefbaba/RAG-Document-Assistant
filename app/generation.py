from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.retrieval import format_docs
from app.self_rag import (
    check_answer_support,
    revise_answer as revise_grounded_answer,
)

load_dotenv()

DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen3:4b")

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


@lru_cache(maxsize=8)
def create_llm(llm_name: str = DEFAULT_LLM):
    """Instantiate chat model supporting Ollama, OpenAI, Groq, Anthropic, or Gemini based on .env keys."""
    load_dotenv()
    llm_lower = llm_name.lower()

    # 1. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and (llm_lower.startswith(("gpt-", "o1-", "o3-", "chatgpt")) or "openai" in llm_lower):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=llm_name,
                temperature=0.1,
                api_key=openai_key,
            )
        except Exception:
            pass

    # 2. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and ("groq" in llm_lower or llm_lower.startswith(("llama-", "mixtral-", "gemma-"))):
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=llm_name,
                temperature=0.1,
                api_key=groq_key,
            )
        except Exception:
            pass

    # 3. Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and "claude" in llm_lower:
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=llm_name,
                temperature=0.1,
                api_key=anthropic_key,
            )
        except Exception:
            pass

    # 4. Google Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key and "gemini" in llm_lower:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=llm_name,
                temperature=0.1,
                google_api_key=gemini_key,
            )
        except Exception:
            pass

    # 5. Default to local Ollama
    ollama_base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    kwargs = {
        "model": llm_name,
        "temperature": 0.1,
    }
    if ollama_base_url:
        kwargs["base_url"] = ollama_base_url

    return ChatOllama(**kwargs)


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
    """Perform live web search using Tavily (if TAVILY_API_KEY is present) or DuckDuckGo fallback."""
    load_dotenv()
    tavily_key = os.getenv("TAVILY_API_KEY")

    # 1. Attempt Tavily Search if API key is provided
    if tavily_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            response = client.search(
                query=question,
                max_results=max_results,
                search_depth="basic",
            )
            results = []
            for item in response.get("results", []):
                snippet = item.get("content", "").strip()
                if snippet:
                    results.append(
                        {
                            "title": item.get("title", "").strip(),
                            "url": item.get("url", "").strip(),
                            "snippet": snippet,
                        }
                    )
            if results:
                return results
        except Exception:
            # Fall back to Tavily REST API via requests
            try:
                import requests

                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": question,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                    timeout=10,
                )
                if resp.ok:
                    data = resp.json()
                    results = []
                    for item in data.get("results", []):
                        snippet = item.get("content", "").strip()
                        if snippet:
                            results.append(
                                {
                                    "title": item.get("title", "").strip(),
                                    "url": item.get("url", "").strip(),
                                    "snippet": snippet,
                                }
                            )
                    if results:
                        return results
            except Exception:
                pass

    # 2. Fall back to DuckDuckGo search (free, no API key required)
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
                        "title": result.get("title", "").strip(),
                        "url": result.get("href", "").strip(),
                        "snippet": result.get("body", "").strip(),
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
