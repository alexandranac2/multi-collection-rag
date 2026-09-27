"""Document service: upload, index, move and delete documents."""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from app.config import settings
from app.models import DocumentResponse, DocumentUpdate
from app.services.langfuse_service import langfuse_service
from app.validation import ensure_within, safe_filename, validate_collection_name
from rag_package import MultiCollectionRAG
from rag_package.processor import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentError(ValueError):
    """A client-side problem with an upload (bad type, too large, unreadable)."""


class DocumentService:
    """Keeps the file on disk, its chunks in Chroma and its registry entry in sync."""

    def __init__(self, rag_instance: MultiCollectionRAG):
        self.rag = rag_instance
        self.documents_base_path = Path(settings.DOCUMENTS_BASE_PATH)
        self.documents_base_path.mkdir(parents=True, exist_ok=True)

        state_dir = Path(settings.STATE_DIR)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = state_dir / "document_registry.json"
        self.document_registry: dict[str, dict] = self._load_registry()

    def _load_registry(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text())
        except json.JSONDecodeError:
            logger.error("Document registry %s is corrupt; starting empty", self.registry_path)
            return {}

    def _save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self.document_registry, indent=2, default=str))

    def _collection_folder(self, collection_name: str) -> Path:
        validate_collection_name(collection_name)
        return ensure_within(self.documents_base_path, self.documents_base_path / collection_name)

    def _ensure_collection(self, collection_name: str, folder: Path, doc_type: str):
        if collection_name not in self.rag.collections:
            self.rag.add_collection(
                collection_name=collection_name,
                docs_path=str(folder),
                doc_type=doc_type,
                chunking_strategy="hierarchical",
            )
        return self.rag.collections[collection_name]

    def upload_document(
        self,
        file: BinaryIO,
        filename: str,
        collection_name: str,
        doc_type: str = "general",
        title: str | None = None,
        description: str | None = None,
    ) -> DocumentResponse:
        """Save an upload into its collection folder and index it."""
        clean_name = safe_filename(filename)
        if Path(clean_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentError(f"Unsupported file type. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}")

        trace = langfuse_service.create_trace(
            name="document_upload",
            metadata={"filename": clean_name, "collection": collection_name},
        )

        folder = self._collection_folder(collection_name)
        # One sub-folder per upload: the file keeps its real name (which is what
        # citations show) and two uploads of "report.pdf" never collide.
        upload_dir = ensure_within(folder, folder / uuid.uuid4().hex[:12])
        upload_dir.mkdir(parents=True)
        file_path = upload_dir / clean_name

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        written = _copy_limited(file, file_path, max_bytes)
        if written is None:
            shutil.rmtree(upload_dir)
            raise DocumentError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit")

        collection = self._ensure_collection(collection_name, folder, doc_type)
        try:
            chunk_count = collection.process_file(file_path, force_reindex=True)
        except Exception as exc:
            # Leave nothing half-indexed behind
            shutil.rmtree(upload_dir)
            collection.remove_source(file_path)
            trace.update(output={"success": False, "error": type(exc).__name__})
            if isinstance(exc, ValueError):
                raise DocumentError(str(exc)) from exc
            raise

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        self.document_registry[doc_id] = {
            "id": doc_id,
            "filename": clean_name,
            "collection": collection_name,
            "doc_type": doc_type,
            "title": title or clean_name,
            "description": description,
            "source_path": str(file_path),
            "chunk_count": chunk_count,
            "upload_date": datetime.now().isoformat(),
            "metadata": {},
        }
        self._save_registry()
        trace.update(output={"document_id": doc_id, "chunks": chunk_count, "success": True})
        return self._to_response(self.document_registry[doc_id])

    def _to_response(self, doc_data: dict) -> DocumentResponse:
        data = dict(doc_data)
        collection = self.rag.collections.get(data["collection"])
        if collection:
            data["chunk_count"] = collection.count_source(Path(data["source_path"]))
        return DocumentResponse(**data)

    def get_document(self, document_id: str) -> DocumentResponse | None:
        doc_data = self.document_registry.get(document_id)
        return self._to_response(doc_data) if doc_data else None

    def list_documents(self, collection: str | None = None) -> list[DocumentResponse]:
        return [
            self._to_response(doc)
            for doc in self.document_registry.values()
            if not collection or doc["collection"] == collection
        ]

    def update_document(self, document_id: str, update_data: DocumentUpdate) -> DocumentResponse | None:
        """Update metadata; changing collection_name moves and re-indexes the file."""
        doc_data = self.document_registry.get(document_id)
        if not doc_data:
            return None

        if update_data.title is not None:
            doc_data["title"] = update_data.title
        if update_data.description is not None:
            doc_data["description"] = update_data.description
        if update_data.doc_type is not None:
            doc_data["doc_type"] = update_data.doc_type

        new_collection = update_data.collection_name
        if new_collection and new_collection != doc_data["collection"]:
            old_path = Path(doc_data["source_path"])
            new_folder = self._collection_folder(new_collection)
            new_folder.mkdir(parents=True, exist_ok=True)
            new_dir = ensure_within(new_folder, new_folder / old_path.parent.name)
            if new_dir.exists():
                raise DocumentError("Target collection already holds this upload")

            shutil.move(str(old_path.parent), str(new_dir))
            new_path = new_dir / old_path.name
            old = self.rag.collections.get(doc_data["collection"])
            if old:
                old.remove_source(old_path)

            target = self._ensure_collection(new_collection, new_folder, doc_data["doc_type"])
            target.process_file(new_path, force_reindex=True)
            doc_data["collection"] = new_collection
            doc_data["source_path"] = str(new_path)

        self._save_registry()
        return self._to_response(doc_data)

    def delete_document(self, document_id: str) -> bool:
        """Remove the file, its chunks and its registry entry."""
        doc_data = self.document_registry.pop(document_id, None)
        if not doc_data:
            return False

        file_path = Path(doc_data["source_path"])
        shutil.rmtree(file_path.parent, ignore_errors=True)
        collection = self.rag.collections.get(doc_data["collection"])
        if collection:
            collection.remove_source(file_path)
        self._save_registry()
        return True


def _copy_limited(src: BinaryIO, dest: Path, max_bytes: int) -> int | None:
    """Stream src to dest; returns bytes written, or None if max_bytes was exceeded."""
    written = 0
    with open(dest, "wb") as out:
        while chunk := src.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                return None
            out.write(chunk)
    return written
