"""FastAPI application entry point."""

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.context import set_user_id
from app.models import (
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    DocType,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
    QACreate,
    QAListResponse,
    QAResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.document_service import DocumentError, DocumentService
from app.services.langfuse_service import langfuse_service
from app.services.qa_service import QAService
from app.services.query_service import QueryService
from app.validation import validate_collection_name
from rag_package import MultiCollectionRAG

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

rag_instance: MultiCollectionRAG | None = None
document_service: DocumentService | None = None
qa_service: QAService | None = None
query_service: QueryService | None = None


def load_existing_collections(rag: MultiCollectionRAG) -> None:
    """Re-attach every persisted Chroma collection to its folder under DOCUMENTS_BASE_PATH."""
    for chroma_coll in rag.chroma_client.list_collections():
        name = getattr(chroma_coll, "name", chroma_coll)  # Chroma >=0.6 returns plain names
        if name == settings.QA_COLLECTION_NAME or name in rag.collections:
            continue
        metadata = rag.chroma_client.get_collection(name).metadata or {}
        docs_path = Path(settings.DOCUMENTS_BASE_PATH) / name
        docs_path.mkdir(parents=True, exist_ok=True)
        rag.add_collection(
            collection_name=name,
            docs_path=str(docs_path),
            doc_type=metadata.get("doc_type", "general"),
            chunk_size=metadata.get("chunk_size", 512),
            chunking_strategy=metadata.get("chunking_strategy", "hierarchical"),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance, document_service, qa_service, query_service

    rag_instance = MultiCollectionRAG(
        chroma_persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        cache_dir=settings.STATE_DIR,
    )
    load_existing_collections(rag_instance)
    document_service = DocumentService(rag_instance)
    qa_service = QAService(rag_instance)
    query_service = QueryService(rag_instance, qa_service)
    logger.info("RAG API started with %d collections", len(rag_instance.collections))

    yield

    langfuse_service.shutdown()  # blocks until queued trace events are sent


app = FastAPI(
    title="Multi-Collection RAG API",
    description="Document collections + curated Q&A behind one semantic search endpoint.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,  # header-based API key, no cookies
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# --- dependencies -------------------------------------------------------------


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    """When API_KEY is configured, every /api route requires a matching X-API-Key header."""
    if settings.API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, settings.API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_request_user_id(x_user_id: str | None = Header(None, alias="X-User-Id")) -> str:
    """
    Caller label for LangFuse traces. This is attribution only, not authentication:
    the header is client-supplied and never used for access control.
    """
    user_id = (x_user_id or "anonymous")[:64]
    set_user_id(user_id)
    return user_id


def _ready(service):
    if service is None:
        raise HTTPException(status_code=503, detail="Service is starting up")
    return service


def get_document_service() -> DocumentService:
    return _ready(document_service)


def get_qa_service() -> QAService:
    return _ready(qa_service)


def get_query_service() -> QueryService:
    return _ready(query_service)


def get_rag_instance() -> MultiCollectionRAG:
    return _ready(rag_instance)


# --- errors -------------------------------------------------------------------


@app.exception_handler(DocumentError)
async def document_error_handler(request, exc: DocumentError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception):
    # Log the detail, never return it: exception text can contain paths or upstream errors
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# --- health -------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "RAG API is running", "version": app.version, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "ready": rag_instance is not None}


@app.get("/health/langfuse")
def health_langfuse():
    return {
        "enabled": langfuse_service.enabled,
        "host": settings.LANGFUSE_HOST if langfuse_service.enabled else None,
    }


# Handlers below are plain `def`: embedding calls, Docling and Chroma all block,
# so FastAPI runs them in its threadpool instead of on the event loop.
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key), Depends(get_request_user_id)])


# --- documents ----------------------------------------------------------------


@api.post("/documents", response_model=DocumentResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    folder_name: str = Form(..., description="Target collection (created if missing)"),
    doc_type: DocType = Form("general"),
    title: str | None = Form(None, max_length=200),
    description: str | None = Form(None, max_length=2000),
    service: DocumentService = Depends(get_document_service),
):
    """Upload a document into a collection and index it."""
    try:
        validate_collection_name(folder_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return service.upload_document(
        file=file.file,
        filename=file.filename or "upload",
        collection_name=folder_name,
        doc_type=doc_type,
        title=title,
        description=description,
    )


@api.get("/documents", response_model=DocumentListResponse)
def list_documents(collection: str | None = None, service: DocumentService = Depends(get_document_service)):
    documents = service.list_documents(collection=collection)
    return DocumentListResponse(documents=documents, total=len(documents))


@api.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, service: DocumentService = Depends(get_document_service)):
    document = service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@api.put("/documents/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: str, update_data: DocumentUpdate, service: DocumentService = Depends(get_document_service)
):
    """Update metadata. Setting collection_name moves the file and re-indexes it."""
    document = service.update_document(document_id, update_data)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@api.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, service: DocumentService = Depends(get_document_service)):
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


# --- Q&A ----------------------------------------------------------------------


@api.post("/qa", response_model=QAResponse, status_code=201)
def create_qa(qa_data: QACreate, service: QAService = Depends(get_qa_service)):
    return service.create_qa(qa_data)


@api.get("/qa", response_model=QAListResponse)
def list_qa(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: QAService = Depends(get_qa_service),
):
    qa_pairs = service.list_qa(limit=limit, offset=offset)
    return QAListResponse(qa_pairs=qa_pairs, total=service.count())


@api.get("/qa/{qa_id}", response_model=QAResponse)
def get_qa(qa_id: str, service: QAService = Depends(get_qa_service)):
    qa = service.get_qa(qa_id)
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    return qa


@api.delete("/qa/{qa_id}", status_code=204)
def delete_qa(qa_id: str, service: QAService = Depends(get_qa_service)):
    if not service.delete_qa(qa_id):
        raise HTTPException(status_code=404, detail="Q&A pair not found")


# --- query --------------------------------------------------------------------


@api.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: QueryService = Depends(get_query_service)):
    """Semantic search across document collections and (optionally) Q&A pairs."""
    return service.query(request)


# --- collections --------------------------------------------------------------


def _collection_response(name: str, rag: MultiCollectionRAG) -> CollectionResponse:
    coll = rag.collections[name]
    sources = {m["source"] for m in coll.collection.get(include=["metadatas"])["metadatas"] if m.get("source")}
    return CollectionResponse(
        name=name,
        doc_type=coll.doc_type,
        document_count=len(sources),
        chunk_count=coll.count(),
        docs_path=str(coll.docs_path),
    )


@api.get("/collections", response_model=CollectionListResponse)
def list_collections(rag: MultiCollectionRAG = Depends(get_rag_instance)):
    collections = [_collection_response(name, rag) for name in rag.collections]
    return CollectionListResponse(collections=collections, total=len(collections))


@api.post("/collections", response_model=CollectionResponse, status_code=201)
def create_collection(collection_data: CollectionCreate, rag: MultiCollectionRAG = Depends(get_rag_instance)):
    name = collection_data.collection_name  # validated by CollectionName
    if name in rag.collections or name == settings.QA_COLLECTION_NAME:
        raise HTTPException(status_code=409, detail="Collection already exists")
    docs_path = Path(settings.DOCUMENTS_BASE_PATH) / name
    docs_path.mkdir(parents=True, exist_ok=True)
    rag.add_collection(
        collection_name=name,
        docs_path=str(docs_path),
        doc_type=collection_data.doc_type,
        chunk_size=collection_data.chunk_size,
        chunking_strategy=collection_data.chunking_strategy,
    )
    return _collection_response(name, rag)


@api.get("/collections/{collection_name}", response_model=CollectionResponse)
def get_collection(collection_name: str, rag: MultiCollectionRAG = Depends(get_rag_instance)):
    if collection_name not in rag.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    return _collection_response(collection_name, rag)


app.include_router(api)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
