"""
Document loading: extracts raw text (and page numbers where available)
from PDF, TXT, and DOCX files. Kept separate from chunking/embedding so
each stage is independently testable.

CPU-bound work (PDF/DOCX parsing) is offloaded to a thread pool via
`asyncio.to_thread` so it never blocks the event loop while other chat
requests are being served concurrently.
"""
from __future__ import annotations

import asyncio
import os

from backend.utils.errors import FileTooLargeError, UnsupportedFileTypeError


class ExtractedPage:
    def __init__(self, page_number: int | None, text: str):
        self.page_number = page_number
        self.text = text


def validate_upload(filename: str, size_bytes: int, allowed_ext: tuple[str, ...], max_mb: int) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")
    if size_bytes > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"File exceeds {max_mb}MB limit")


def _extract_txt(path: str) -> list[ExtractedPage]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ExtractedPage(None, f.read())]


def _extract_pdf(path: str) -> list[ExtractedPage]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(ExtractedPage(i, text))
    return pages


def _extract_docx(path: str) -> list[ExtractedPage]:
    import docx

    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [ExtractedPage(None, text)]


_EXTRACTORS = {".txt": _extract_txt, ".pdf": _extract_pdf, ".docx": _extract_docx}


async def extract_pages(path: str) -> list[ExtractedPage]:
    ext = os.path.splitext(path)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if not extractor:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")
    # CPU-bound parsing -> run off the event loop.
    return await asyncio.to_thread(extractor, path)
