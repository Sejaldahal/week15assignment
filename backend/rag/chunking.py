"""
Text cleaning and chunking.

Chunk size defaults to 800 characters with 120 characters (~15%) overlap:
large enough to usually contain a full paragraph/thought (reducing
fragmentation of ideas across chunks), small enough that 4-6 retrieved
chunks comfortably fit the prompt budget alongside the system prompt and
chat history. The overlap prevents a sentence spanning a chunk boundary
from being lost from both neighboring chunks.
"""
from __future__ import annotations

import re
import uuid


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # try to end on a sentence/paragraph boundary if one exists nearby
        if end < length:
            boundary = text.rfind("\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - chunk_overlap
    return chunks


def new_document_id() -> str:
    return uuid.uuid4().hex[:12]


def new_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-c{index}"
