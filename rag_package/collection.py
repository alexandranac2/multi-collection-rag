import hashlib
import logging
from pathlib import Path

from .cache import CacheManager
from .processor import DocumentProcessor

logger = logging.getLogger(__name__)


class CollectionManager:
    """Manages a single document collection (one folder ↔ one Chroma collection)."""

    def __init__(
        self,
        collection,
        collection_name: str,
        docs_path: str,
        doc_type: str,
        chunk_size: int,
        chunking_strategy: str,
        embed_func,
        save_recognized: bool = False,
        cache_dir: Path = Path("."),
    ):
        self.collection = collection
        self.name = collection_name
        self.docs_path = Path(docs_path)
        self.doc_type = doc_type
        self.chunk_size = chunk_size
        self.chunking_strategy = chunking_strategy
        self.embed_func = embed_func
        self.save_recognized = save_recognized

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = CacheManager(cache_dir / f".{collection_name}_cache.json")
        self.processor = DocumentProcessor()

    def ingest(self, force_reindex: bool = False) -> dict[str, int]:
        """
        Ingest every supported file under docs_path.

        One bad file does not stop the batch: failures are logged and counted.
        """
        if not self.docs_path.exists():
            raise ValueError(f"Path {self.docs_path} does not exist")

        files = [
            path
            for ext in self.processor.get_supported_extensions()
            for path in self.docs_path.rglob(f"*{ext}")
            if "recognize" not in path.parts  # OCR text dumps written by save_recognized
        ]
        logger.info("Collection %s: %d documents found", self.name, len(files))

        stats = {"indexed": 0, "skipped": 0, "failed": 0}
        for file_path in files:
            try:
                stats["indexed" if self.process_file(file_path, force_reindex) else "skipped"] += 1
            except Exception:
                stats["failed"] += 1
                logger.exception("Failed to ingest %s", file_path)
        return stats

    def process_file(self, file_path: Path, force_reindex: bool = False) -> int:
        """
        Convert, chunk, embed and upsert one file. Returns the number of chunks
        written (0 when the file is unchanged since the last run).

        Raises on conversion or embedding failure so callers can report it.
        """
        file_hash = self.cache.calculate_file_hash(file_path)
        file_key = str(file_path)

        if not force_reindex and self.cache.get(file_key) == file_hash:
            logger.debug("Unchanged, skipping %s", file_path.name)
            return 0

        chunks = self.processor.process(
            file_path=file_path,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            doc_type=self.doc_type,
            collection_name=self.name,
            save_recognized=self.save_recognized,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from {file_path.name}")

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_func(texts)

        # Replace, don't append: drop this file's previous chunks first
        self.collection.delete(where={"source": file_key})
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
            ids=[self.chunk_id(file_path, i) for i in range(len(chunks))],
        )

        self.cache.set(file_key, file_hash)
        self.cache.save()
        logger.info("Indexed %s (%d chunks)", file_path.name, len(chunks))
        return len(chunks)

    def chunk_id(self, file_path: Path, index: int) -> str:
        # Keyed on the full path, so report.pdf and report.docx never collide
        path_hash = hashlib.sha1(str(file_path).encode()).hexdigest()[:12]
        return f"{self.name}_{path_hash}_{index}"

    def count_source(self, file_path: Path) -> int:
        """Number of chunks stored for one source file."""
        return len(self.collection.get(where={"source": str(file_path)}, include=[])["ids"])

    def remove_source(self, file_path: Path) -> None:
        """Delete a file's chunks and forget its hash."""
        self.collection.delete(where={"source": str(file_path)})
        self.cache.remove(str(file_path))
        self.cache.save()

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where_clause: dict | None = None,
    ) -> list[dict]:
        """Nearest chunks in this collection (cosine distance, lower = closer)."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": chunk_id,
                "collection": self.name,
            }
            for i, chunk_id in enumerate(results["ids"][0])
        ]

    def count(self) -> int:
        """Number of chunks in the collection."""
        return self.collection.count()
