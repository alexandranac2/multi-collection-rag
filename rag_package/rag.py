import logging
import os
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

from .collection import CollectionManager

load_dotenv()
logger = logging.getLogger(__name__)

# OpenAI accepts up to 2048 inputs per embeddings request
EMBED_BATCH_SIZE = 256


class MultiCollectionRAG:
    """Several independently chunked document collections behind one query interface."""

    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        cache_dir: str = ".",
    ):
        # Setup OpenAI
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

        # Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir, settings=Settings(anonymized_telemetry=False)
        )

        self.cache_dir = Path(cache_dir)
        self.collections: dict[str, CollectionManager] = {}

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches (large documents can exceed one request)."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = self.openai_client.embeddings.create(
                input=texts[start : start + EMBED_BATCH_SIZE],
                model=self.embedding_model,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def add_collection(
        self,
        collection_name: str,
        docs_path: str,
        doc_type: Literal["manual", "policy", "contract", "general"] = "general",
        chunk_size: int = 512,
        chunking_strategy: Literal["hybrid", "hierarchical"] = "hybrid",
        save_recognized: bool = False,
    ):
        """Add a document collection."""
        # Create ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            # Chunk settings are persisted so a restart re-attaches the collection as created
            metadata={
                "hnsw:space": "cosine",
                "doc_type": doc_type,
                "chunk_size": chunk_size,
                "chunking_strategy": chunking_strategy,
            },
        )

        # Create collection manager
        self.collections[collection_name] = CollectionManager(
            collection=chroma_collection,
            collection_name=collection_name,
            docs_path=docs_path,
            doc_type=doc_type,
            chunk_size=chunk_size,
            chunking_strategy=chunking_strategy,
            embed_func=self._embed_texts,
            save_recognized=save_recognized,
            cache_dir=self.cache_dir,
        )
        logger.info("Added collection %s (%s)", collection_name, doc_type)
        return self

    def ingest_collection(self, collection_name: str, force_reindex: bool = False) -> dict[str, int]:
        """Ingest a specific collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        return self.collections[collection_name].ingest(force_reindex)

    def ingest_all(self, force_reindex: bool = False):
        """Ingest all collections."""
        for coll in self.collections.values():
            coll.ingest(force_reindex)

    def query(
        self,
        query_text: str,
        collections: list[str] | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 5,
    ) -> dict[str, list[dict]]:
        """Query collections."""
        target_colls = collections or list(self.collections.keys())

        # Build filter
        where_clause = {}
        if doc_types:
            where_clause["doc_type"] = {"$in": doc_types}

        # Embed query
        query_embedding = self._embed_texts([query_text])[0]

        # Query each collection
        results = {}
        for coll_name in target_colls:
            if coll_name in self.collections:
                results[coll_name] = self.collections[coll_name].query(
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where_clause=where_clause if where_clause else None,
                )

        return results

    def query_combined(
        self,
        query_text: str,
        collections: list[str] | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """Query and combine results."""
        all_results = self.query(query_text, collections, doc_types, n_results)

        # Combine and sort
        combined = []
        for results in all_results.values():
            combined.extend(results)

        combined.sort(key=lambda x: x["distance"])
        return combined[:n_results]

    def list_collections(self) -> dict[str, int]:
        """Chunk count per collection."""
        return {name: coll.count() for name, coll in self.collections.items()}
