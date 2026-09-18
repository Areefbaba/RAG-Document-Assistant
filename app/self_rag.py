from __future__ import annotations

import json
import re
from typing import Callable

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


SUPPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict RAG answer evaluator.

Determine whether every important factual claim in the proposed answer is supported by the supplied context.

Return JSON only:
{"supported": true, "reason": "brief reason"}

or

{"supported": false, "reason": "brief reason"}

Do not use outside knowledge.
""",
        ),
        (
            "human",
            """Question:
{question}

Context:
{context}

Proposed answer:
{answer}
""",
        ),
    ]
)


USEFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict answer-quality evaluator.

Determine whether the proposed answer directly and usefully answers the user's question.

Return JSON only:
{"useful": true, "reason": "brief reason"}

or

{"useful": false, "reason": "brief reason"}

An answer is not useful when it is evasive, incomplete, unrelated, or does not address the requested information.
""",
        ),
        (
            "human",
            """Question:
{question}

Proposed answer:
{answer}
""",
        ),
    ]
)


REVISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grounded document question-answering assistant.

Correct the proposed answer using ONLY the supplied context.

Rules:
- Every factual claim must be supported by the context.
- Do not use outside knowledge.
- Do not invent facts or citations.
- If the context is insufficient, say:
"I don't know based on the provided documents."
- Keep the answer concise.
- Cite document claims as [filename, Page N].
""",
        ),
        (
            "human",
            """Question:
{question}

Context:
{context}

Proposed answer:
{answer}

Write the corrected answer.
""",
        ),
    ]
)


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's question into a clearer retrieval query.

Preserve the user's intent.
Do not answer the question.
Return only the rewritten question.
""",
        ),
        ("human", "{question}"),
    ]
)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def check_answer_support(
    question: str,
    answer: str,
    context: str,
    llm: ChatOllama,
) -> tuple[bool, str]:
    response = (
        SUPPORT_PROMPT
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "question": question,
            "answer": answer,
            "context": context,
        }
    )

    payload = _parse_json(response)

    if "supported" in payload:
        return bool(payload["supported"]), str(
            payload.get("reason", "")
        )

    return True, "Support evaluation could not be parsed."


def check_answer_useful(
    question: str,
    answer: str,
    llm: ChatOllama,
) -> tuple[bool, str]:
    response = (
        USEFULNESS_PROMPT
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "question": question,
            "answer": answer,
        }
    )

    payload = _parse_json(response)

    if "useful" in payload:
        return bool(payload["useful"]), str(
            payload.get("reason", "")
        )

    return bool(answer.strip()), "Usefulness evaluation could not be parsed."


def revise_answer(
    question: str,
    answer: str,
    context: str,
    llm: ChatOllama,
) -> str:
    return (
        REVISE_PROMPT
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "question": question,
            "answer": answer,
            "context": context,
        }
    ).strip()


def rewrite_question(
    question: str,
    llm: ChatOllama,
) -> str:
    rewritten = (
        REWRITE_PROMPT
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "question": question,
        }
    ).strip()

    return rewritten or question


def self_reflect_and_correct(
    question: str,
    answer: str,
    context: str,
    llm: ChatOllama,
    max_retries: int = 1,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    current_answer = answer

    for attempt in range(max_retries + 1):
        if status_callback:
            status_callback("is_sup")

        supported, _ = check_answer_support(
            question,
            current_answer,
            context,
            llm,
        )

        if supported or attempt >= max_retries:
            return current_answer

        if status_callback:
            status_callback("revise_answer")

        current_answer = revise_answer(
            question,
            current_answer,
            context,
            llm,
        )

    return current_answer
