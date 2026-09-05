from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile

from backend.assistant.schemas import UploadResponse
from backend.rag.chunking import chunk_text, clean_text, new_chunk_id, new_document_id
from backend.rag.embeddings import embed_texts
from backend.rag.ingestion import extract_pages, validate_upload
from backend.utils.errors import FileTooLargeError, UnsupportedFileTypeError

logger = logging.getLogger("ai_assistant.api.documents")
router = APIRouter()


async def _ingest(path: str, document_id: str, document_name: str, request: Request) -> int:
    settings = request.app.state.settings
    provider = request.app.state.llm_chain._primary
    store = request.app.state.vector_store

    pages = await extract_pages(path)
    all_chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    idx = 0
    for page in pages:
        cleaned = clean_text(page.text)
        for piece in chunk_text(cleaned, settings.chunk_size, settings.chunk_overlap):
            chunk_id = new_chunk_id(document_id, idx)
            all_chunks.append(piece)
            ids.append(chunk_id)
            # ChromaDB metadata values must be str/int/float/bool - None is
            # rejected, so page is omitted entirely when not available
            # (e.g. for .txt/.docx which have no page concept).
            meta: dict = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_id": chunk_id,
                "source": document_name,
            }
            if page.page_number is not None:
                meta["page"] = page.page_number
            metadatas.append(meta)
            idx += 1

    if not all_chunks:
        return 0

    embeddings = await embed_texts(provider, all_chunks)
    store.add(ids=ids, embeddings=embeddings, documents=all_chunks, metadatas=metadatas)
    logger.info("document_ingested", extra={"event": "document_ingested"})
    return len(all_chunks)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile
) -> UploadResponse:
    settings = request.app.state.settings
    contents = await file.read()

    try:
        validate_upload(file.filename, len(contents), settings.allowed_upload_extensions, settings.max_upload_size_mb)
    except (UnsupportedFileTypeError, FileTooLargeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = new_document_id()
    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(settings.upload_dir, f"{document_id}{ext}")
    with open(saved_path, "wb") as f:
        f.write(contents)

    # Ingestion (extraction/chunking/embedding) is CPU/IO heavy; run it now
    # for small dev-scale files but keep it decoupled via a plain function
    # so it can trivially be swapped for background_tasks.add_task(...) or
    # a task queue for larger documents without touching this endpoint's
    # contract.
    try:
        chunk_count = await _ingest(saved_path, document_id, file.filename, request)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed", exc_info=True)
        return UploadResponse(
            document_id=document_id,
            document_name=file.filename,
            status="failed",
            message=str(exc),
        )

    return UploadResponse(
        document_id=document_id,
        document_name=file.filename,
        status="ingested",
        chunk_count=chunk_count,
    )
