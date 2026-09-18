from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


SELF_RAG_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict answer verifier for a grounded RAG system.

Check whether the proposed answer is fully supported by the supplied context.

Return JSON only:
{"supported": true, "reason": "brief reason"}

or

{"supported": false, "reason": "brief reason"}

An answer is supported only when its factual claims can be directly supported by the supplied context.
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


STRICT_REGEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grounded document question-answering assistant.

Generate the answer using ONLY the supplied context.

Rules:
- Every factual claim must be supported by the context.
- Do not use outside knowledge.
- Do not invent facts or citations.
- If the context is insufficient, say:
"I don't know based on the provided documents."
- Keep the answer concise.
- Cite sources as [filename, Page N].
""",
        ),
        (
            "human",
            """Question:
{question}

Context:
{context}

Previous answer:
{answer}

Write a corrected answer that is fully grounded in the context.
""",
        ),
    ]
)


def _parse_critique(text: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            payload = json.loads(match.group(0))
            supported = bool(payload.get("supported", False))
            reason = str(payload.get("reason", ""))
            return supported, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    lowered = text.lower()

    if '"supported": false' in lowered or '"supported":false' in lowered:
        return False, text.strip()

    if '"supported": true' in lowered or '"supported":true' in lowered:
        return True, text.strip()

    return True, "Critique could not be parsed; keeping the grounded answer."


def self_reflect_and_correct(
    question: str,
    answer: str,
    context: str,
    llm: ChatOllama,
    max_retries: int = 1,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    if not answer.strip():
        return answer

    for attempt in range(max_retries + 1):
        if status_callback:
            status_callback("self_reflect")

        critique_response = (
            SELF_RAG_CRITIQUE_PROMPT
            | llm
            | StrOutputParser()
        ).invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

        supported, _ = _parse_critique(critique_response)

        if supported or attempt >= max_retries:
            return answer

        if status_callback:
            status_callback("self_retry")

        answer = (
            STRICT_REGEN_PROMPT
            | llm
            | StrOutputParser()
        ).invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        ).strip()

    return answer
