from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean and normalize extracted raw text from PDF documents."""
    text = text.replace("\x00", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs(paths: Iterable[str | Path]) -> list[Document]:
    """Load and preprocess multiple PDF documents."""
    documents = []

    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue

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
