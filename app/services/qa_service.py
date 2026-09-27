"""Q&A service: curated question/answer pairs stored as vectors next to the documents."""

import uuid
from datetime import datetime

from app.config import settings
from app.models import QACreate, QAResponse
from app.services.langfuse_service import langfuse_service


class QAService:
    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.collection_name = settings.QA_COLLECTION_NAME
        self.qa_collection = self.rag.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "doc_type": "qa"},
        )

    def _embed_text(self, text: str) -> list[float]:
        return self.rag._embed_texts([text])[0]

    @staticmethod
    def _to_response(qa_id: str, metadata: dict) -> QAResponse:
        tags = metadata.get("tags", "")
        return QAResponse(
            id=qa_id,
            question=metadata.get("question", ""),
            answer=metadata.get("answer", ""),
            tags=[t for t in tags.split(",") if t],
            category=metadata.get("category") or None,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            metadata=metadata,
        )

    def create_qa(self, qa_data: QACreate) -> QAResponse:
        """Embed question + answer together so either side can match a query."""
        trace = langfuse_service.create_trace(name="qa_creation", metadata={"question": qa_data.question[:100]})
        combined_text = f"Q: {qa_data.question}\nA: {qa_data.answer}"
        embedding = self._embed_text(combined_text)
        langfuse_service.track_embedding(trace=trace, texts=[combined_text], model=settings.OPENAI_EMBEDDING_MODEL)

        qa_id = f"qa_{uuid.uuid4().hex[:12]}"
        metadata = {
            "question": qa_data.question,
            "answer": qa_data.answer,
            "qa_id": qa_id,
            "doc_type": "qa",
            "tags": ",".join(qa_data.tags or []),  # Chroma metadata values must be scalars
            "category": qa_data.category or "",
            "created_at": datetime.now().isoformat(),
        }
        self.qa_collection.add(embeddings=[embedding], documents=[combined_text], metadatas=[metadata], ids=[qa_id])
        trace.update(output={"qa_id": qa_id, "success": True})
        return self._to_response(qa_id, metadata)

    def get_qa(self, qa_id: str) -> QAResponse | None:
        result = self.qa_collection.get(ids=[qa_id])
        if not result["ids"]:
            return None
        return self._to_response(qa_id, result["metadatas"][0])

    def list_qa(self, limit: int = 100, offset: int = 0) -> list[QAResponse]:
        result = self.qa_collection.get(limit=limit, offset=offset)
        return [self._to_response(qa_id, meta) for qa_id, meta in zip(result["ids"], result["metadatas"], strict=True)]

    def delete_qa(self, qa_id: str) -> bool:
        if not self.qa_collection.get(ids=[qa_id], include=[])["ids"]:
            return False
        self.qa_collection.delete(ids=[qa_id])
        return True

    def query_qa(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Nearest Q&A pairs to the query (cosine distance, lower = closer)."""
        if self.qa_collection.count() == 0:
            return []

        trace = langfuse_service.create_trace(name="qa_query", metadata={"query": query_text[:100]})
        query_embedding = self._embed_text(query_text)
        langfuse_service.track_embedding(trace=trace, texts=[query_text], model=settings.OPENAI_EMBEDDING_MODEL)

        results = self.qa_collection.query(query_embeddings=[query_embedding], n_results=n_results)
        langfuse_service.track_retrieval(
            trace=trace,
            query=query_text,
            results_count=len(results["ids"][0]),
            collection=self.collection_name,
        )
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": qa_id,
                "collection": self.collection_name,
                "question": results["metadatas"][0][i].get("question", ""),
                "answer": results["metadatas"][0][i].get("answer", ""),
                "qa_id": qa_id,
            }
            for i, qa_id in enumerate(results["ids"][0])
        ]

    def count(self) -> int:
        return self.qa_collection.count()
