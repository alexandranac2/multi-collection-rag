"""FastAPI application entry point."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, List
from pathlib import Path
import uvicorn

from rag_package import MultiCollectionRAG

from app.config import settings
from app.models import (
    DocumentCreate, DocumentUpdate, DocumentResponse, DocumentListResponse,
    QACreate, QAResponse, QAListResponse,
    QueryRequest, QueryResponse,
    CollectionCreate, CollectionResponse, CollectionListResponse
)
from app.services.document_service import DocumentService
from app.services.qa_service import QAService
from app.services.query_service import QueryService
from app.services.langfuse_service import langfuse_service
from app.context import set_user_id


# Global instances
rag_instance: Optional[MultiCollectionRAG] = None
document_service: Optional[DocumentService] = None
qa_service: Optional[QAService] = None
query_service: Optional[QueryService] = None


def load_existing_collections(rag: MultiCollectionRAG):
    """Load existing collections from ChromaDB and documents folder."""
    # Get all existing ChromaDB collections (excluding qa_history)
    try:
        all_collections = rag.chroma_client.list_collections()
        print(f"📚 Found {len(all_collections)} existing ChromaDB collections")
        
        for chroma_coll in all_collections:
            # Skip Q&A collection
            if chroma_coll.name == settings.QA_COLLECTION_NAME:
                continue
            
            # Check if collection is already loaded
            if chroma_coll.name in rag.collections:
                continue
            
            # Get collection metadata
            metadata = chroma_coll.metadata or {}
            doc_type = metadata.get('doc_type', 'general')
            
            # Try to find the documents folder for this collection
            # Check common locations
            possible_paths = [
                Path(settings.DOCUMENTS_BASE_PATH) / chroma_coll.name,
                Path(settings.DOCUMENTS_BASE_PATH) / "pdf",  # For the pdf folder
                Path("./documents") / chroma_coll.name,
                Path("./documents") / "pdf",
            ]
            
            docs_path = None
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    docs_path = str(path)
                    break
            
            # If no path found, use a default based on collection name
            if not docs_path:
                docs_path = str(Path(settings.DOCUMENTS_BASE_PATH) / chroma_coll.name)
                Path(docs_path).mkdir(parents=True, exist_ok=True)
            
            # Load the collection
            try:
                rag.add_collection(
                    collection_name=chroma_coll.name,
                    docs_path=docs_path,
                    doc_type=doc_type,
                    chunk_size=512,
                    chunk_overlap=128,
                    chunking_strategy="hierarchical"
                )
                print(f"✅ Loaded existing collection: {chroma_coll.name} ({doc_type}) - {chroma_coll.count()} chunks")
            except Exception as e:
                print(f"⚠️  Warning: Could not load collection {chroma_coll.name}: {e}")
    except Exception as e:
        print(f"⚠️  Warning: Could not list existing collections: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global rag_instance, document_service, qa_service, query_service
    
    # Startup
    print("🚀 Starting RAG API...")
    
    # Initialize RAG instance
    rag_instance = MultiCollectionRAG(
        chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL
    )
    
    # Load existing collections from ChromaDB
    print("📂 Loading existing collections...")
    load_existing_collections(rag_instance)
    
    # Also check if there's a pdf folder and add it if not already loaded
    pdf_folder = Path(settings.DOCUMENTS_BASE_PATH) / "pdf"
    if pdf_folder.exists() and "pdf" not in rag_instance.collections:
        # Check if there's a collection in ChromaDB that might match
        try:
            all_collections = rag_instance.chroma_client.list_collections()
            # Try to find a collection that might be for pdf folder
            for coll in all_collections:
                if coll.name != settings.QA_COLLECTION_NAME and coll.name not in rag_instance.collections:
                    # Load it with pdf path
                    metadata = coll.metadata or {}
                    rag_instance.add_collection(
                        collection_name=coll.name,
                        docs_path=str(pdf_folder),
                        doc_type=metadata.get('doc_type', 'general'),
                        chunk_size=512,
                        chunk_overlap=128,
                        chunking_strategy="hierarchical"
                    )
                    print(f"✅ Loaded collection '{coll.name}' with pdf folder path")
                    break
        except:
            pass
    
    # Initialize services
    document_service = DocumentService(rag_instance)
    qa_service = QAService(rag_instance)
    query_service = QueryService(rag_instance, qa_service)
    
    print(f"✅ RAG API started successfully with {len(rag_instance.collections)} collections loaded")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down RAG API...")
    if langfuse_service.enabled:
        langfuse_service.shutdown()  # Use shutdown instead of flush for final cleanup
    print("✅ RAG API shut down")


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="RAG API with document and Q&A management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to flush LangFuse events after each request
@app.middleware("http")
async def flush_langfuse_middleware(request, call_next):
    """Flush LangFuse events after each request to ensure traces appear immediately."""
    response = await call_next(request)
    # Flush LangFuse events after each request
    if langfuse_service.enabled and langfuse_service.client:
        try:
            # Flush synchronously to ensure events are sent
            langfuse_service.client.flush()
        except Exception as e:
            print(f"❌ Warning: Failed to flush LangFuse events in middleware: {e}")
    return response


# Dependency to get services
def get_document_service() -> DocumentService:
    """Get document service instance."""
    if document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")
    return document_service


def get_qa_service() -> QAService:
    """Get Q&A service instance."""
    if qa_service is None:
        raise HTTPException(status_code=500, detail="Q&A service not initialized")
    return qa_service


def get_query_service() -> QueryService:
    """Get query service instance."""
    if query_service is None:
        raise HTTPException(status_code=500, detail="Query service not initialized")
    return query_service


def get_rag_instance() -> MultiCollectionRAG:
    """Get RAG instance."""
    if rag_instance is None:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    return rag_instance


def get_request_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> str:
    """
    Extract user_id from X-User-Id header and set it in context.
    If no header is provided, defaults to 'system'.
    Clients should send: X-User-Id: <user_id>
    """
    # Use default 'system' if no header provided
    user_id = x_user_id if x_user_id else "system"
    set_user_id(user_id)
    return user_id


# Health check
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/langfuse")
async def health_langfuse():
    """LangFuse diagnostic endpoint."""
    langfuse_status = {
        "enabled": langfuse_service.enabled,
        "client_initialized": langfuse_service.client is not None,
        "public_key_set": bool(settings.LANGFUSE_PUBLIC_KEY),
        "secret_key_set": bool(settings.LANGFUSE_SECRET_KEY),
        "host": settings.LANGFUSE_HOST,
        "public_key_preview": settings.LANGFUSE_PUBLIC_KEY[:10] + "..." if settings.LANGFUSE_PUBLIC_KEY else None,
    }
    return {
        "langfuse": langfuse_status,
        "message": "✅ LangFuse is ready" if langfuse_service.enabled else "⚠️ LangFuse is disabled - check your .env file"
    }


# Document endpoints
@app.post("/api/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    folder_name: str = Form(...),
    doc_type: str = Form("general"),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    service: DocumentService = Depends(get_document_service),
    _user_id: str = Depends(get_request_user_id)
):
    """Upload and ingest a document."""
    try:
        return service.upload_document(
            file=file,
            folder_name=folder_name,
            doc_type=doc_type,
            title=title,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    collection: Optional[str] = None,
    service: DocumentService = Depends(get_document_service)
):
    """List all documents."""
    try:
        documents = service.list_documents(collection=collection)
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get document by ID."""
    document = service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.put("/api/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    update_data: DocumentUpdate,
    service: DocumentService = Depends(get_document_service)
):
    """Update document metadata."""
    document = service.update_document(document_id, update_data)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a document."""
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return None


# Q&A endpoints
@app.post("/api/qa", response_model=QAResponse, status_code=201)
async def create_qa(
    qa_data: QACreate,
    service: QAService = Depends(get_qa_service),
    _user_id: str = Depends(get_request_user_id)
):
    """Create a Q&A pair."""
    try:
        return service.create_qa(qa_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qa", response_model=QAListResponse)
async def list_qa(
    limit: int = 100,
    offset: int = 0,
    service: QAService = Depends(get_qa_service)
):
    """List all Q&A pairs."""
    try:
        qa_pairs = service.list_qa(limit=limit, offset=offset)
        return QAListResponse(qa_pairs=qa_pairs, total=len(qa_pairs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qa/{qa_id}", response_model=QAResponse)
async def get_qa(
    qa_id: str,
    service: QAService = Depends(get_qa_service)
):
    """Get Q&A pair by ID."""
    qa = service.get_qa(qa_id)
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    return qa


@app.delete("/api/qa/{qa_id}", status_code=204)
async def delete_qa(
    qa_id: str,
    service: QAService = Depends(get_qa_service)
):
    """Delete a Q&A pair."""
    success = service.delete_qa(qa_id)
    if not success:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    return None


# Query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
    _user_id: str = Depends(get_request_user_id)
):
    """Query the RAG system (searches both documents and Q&A)."""
    try:
        return service.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Collection endpoints
@app.get("/api/collections", response_model=CollectionListResponse)
async def list_collections(
    rag: MultiCollectionRAG = Depends(get_rag_instance)
):
    """List all collections."""
    try:
        collections = []
        for name, coll_manager in rag.collections.items():
            collections.append(CollectionResponse(
                name=name,
                doc_type=coll_manager.doc_type,
                document_count=0,  # Would need to track this separately
                chunk_count=coll_manager.count(),
                docs_path=str(coll_manager.docs_path)
            ))
        return CollectionListResponse(collections=collections, total=len(collections))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collections", response_model=CollectionResponse, status_code=201)
async def create_collection(
    collection_data: CollectionCreate,
    rag: MultiCollectionRAG = Depends(get_rag_instance)
):
    """Create a new collection."""
    try:
        folder_path = f"{settings.DOCUMENTS_BASE_PATH}/{collection_data.collection_name}"
        rag.add_collection(
            collection_name=collection_data.collection_name,
            docs_path=folder_path,
            doc_type=collection_data.doc_type,
            chunk_size=collection_data.chunk_size,
            chunk_overlap=collection_data.chunk_overlap,
            chunking_strategy=collection_data.chunking_strategy
        )
        
        coll_manager = rag.collections[collection_data.collection_name]
        return CollectionResponse(
            name=collection_data.collection_name,
            doc_type=collection_data.doc_type,
            document_count=0,
            chunk_count=coll_manager.count(),
            docs_path=str(coll_manager.docs_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections/{collection_name}", response_model=CollectionResponse)
async def get_collection(
    collection_name: str,
    rag: MultiCollectionRAG = Depends(get_rag_instance)
):
    """Get collection details."""
    if collection_name not in rag.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    coll_manager = rag.collections[collection_name]
    return CollectionResponse(
        name=collection_name,
        doc_type=coll_manager.doc_type,
        document_count=0,
        chunk_count=coll_manager.count(),
        docs_path=str(coll_manager.docs_path)
    )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

