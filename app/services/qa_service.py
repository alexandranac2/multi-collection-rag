"""Q&A service for managing question/answer pairs in ChromaDB."""
import uuid
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os

from app.config import settings
from app.models import QACreate, QAResponse
from app.services.langfuse_service import langfuse_service


class QAService:
    """Service for managing Q&A pairs in vector database."""
    
    def __init__(self, rag_instance):
        """Initialize Q&A service with RAG instance."""
        self.rag = rag_instance
        self.collection_name = settings.QA_COLLECTION_NAME
        
        # Get or create Q&A collection
        self.qa_collection = self.rag.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "doc_type": "qa"}
        )
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.rag._embed_texts([text])[0]
    
    def create_qa(self, qa_data: QACreate) -> QAResponse:
        """Create and store a Q&A pair."""
        trace = None
        if langfuse_service.enabled:
            trace = langfuse_service.create_trace(
                name="qa_creation",
                metadata={"question": qa_data.question[:100]}
            )
        
        try:
            # Create combined text for embedding
            combined_text = f"Q: {qa_data.question}\nA: {qa_data.answer}"
            
            # Generate embedding
            embedding = self._embed_text(combined_text)
            
            if trace and langfuse_service.enabled:
                langfuse_service.track_embedding(
                    trace_id=trace.id,
                    texts=[combined_text],
                    model=settings.OPENAI_EMBEDDING_MODEL
                )
            
            # Generate unique ID
            qa_id = f"qa_{uuid.uuid4().hex[:12]}"
            
            # Prepare metadata
            metadata = {
                "question": qa_data.question,
                "answer": qa_data.answer,
                "qa_id": qa_id,
                "doc_type": "qa",
                "tags": ",".join(qa_data.tags) if qa_data.tags else "",
                "category": qa_data.category or "",
                "created_at": datetime.now().isoformat()
            }
            
            # Store in ChromaDB
            self.qa_collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[metadata],
                ids=[qa_id]
            )
            
            if trace and langfuse_service.enabled:
                trace.update(output={"qa_id": qa_id, "success": True})
            
            return QAResponse(
                id=qa_id,
                question=qa_data.question,
                answer=qa_data.answer,
                tags=qa_data.tags or [],
                category=qa_data.category,
                created_at=datetime.now(),
                metadata=metadata
            )
        except Exception as e:
            if trace and langfuse_service.enabled:
                trace.update(output={"success": False, "error": str(e)})
            raise
    
    def get_qa(self, qa_id: str) -> Optional[QAResponse]:
        """Get a specific Q&A pair by ID."""
        try:
            result = self.qa_collection.get(ids=[qa_id])
            if not result['ids']:
                return None
            
            metadata = result['metadatas'][0]
            return QAResponse(
                id=qa_id,
                question=metadata.get("question", ""),
                answer=metadata.get("answer", ""),
                tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                category=metadata.get("category"),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                metadata=metadata
            )
        except Exception:
            return None
    
    def list_qa(self, limit: int = 100, offset: int = 0) -> List[QAResponse]:
        """List all Q&A pairs."""
        try:
            # Get all Q&A pairs (ChromaDB doesn't have pagination, so we get all and slice)
            result = self.qa_collection.get()
            
            qa_pairs = []
            for i, qa_id in enumerate(result['ids']):
                if i < offset:
                    continue
                if len(qa_pairs) >= limit:
                    break
                
                metadata = result['metadatas'][i]
                qa_pairs.append(QAResponse(
                    id=qa_id,
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    category=metadata.get("category"),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                    metadata=metadata
                ))
            
            return qa_pairs
        except Exception:
            return []
    
    def delete_qa(self, qa_id: str) -> bool:
        """Delete a Q&A pair."""
        try:
            self.qa_collection.delete(ids=[qa_id])
            return True
        except Exception:
            return False
    
    def query_qa(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query Q&A collection for similar pairs."""
        trace = None
        if langfuse_service.enabled:
            trace = langfuse_service.create_trace(
                name="qa_query",
                metadata={"query": query_text[:100]}
            )
        
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query_text)
            
            if trace and langfuse_service.enabled:
                langfuse_service.track_embedding(
                    trace_id=trace.id,
                    texts=[query_text],
                    model=settings.OPENAI_EMBEDDING_MODEL
                )
            
            # Query ChromaDB
            results = self.qa_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if trace and langfuse_service.enabled:
                langfuse_service.track_retrieval(
                    trace_id=trace.id,
                    query=query_text,
                    results_count=len(results['ids'][0]) if results['ids'][0] else 0,
                    collection=self.collection_name
                )
            
            # Format results
            formatted = []
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    formatted.append({
                        'text': results['documents'][0][i],
                        'metadata': metadata,
                        'distance': results['distances'][0][i],
                        'id': results['ids'][0][i],
                        'collection': self.collection_name,
                        'question': metadata.get('question', ''),
                        'answer': metadata.get('answer', ''),
                        'qa_id': metadata.get('qa_id', results['ids'][0][i])
                    })
            
            if trace and langfuse_service.enabled:
                trace.update(output={"results_count": len(formatted), "success": True})
            
            return formatted
        except Exception as e:
            if trace and langfuse_service.enabled:
                trace.update(output={"success": False, "error": str(e)})
            raise
    
    def count(self) -> int:
        """Get total number of Q&A pairs."""
        return self.qa_collection.count()

