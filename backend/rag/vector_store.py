"""
ChromaDB-backed vector store, wrapped behind a small interface so it can
later be swapped for pgvector/Qdrant/etc. without touching ingestion or
retrieval code.
"""
from __future__ import annotations

from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int) -> dict:
        return self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self._collection.count()

    def health_check(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:  # noqa: BLE001
            return False
