"""Pydantic models for request/response validation."""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# Document Models
class DocumentCreate(BaseModel):
    """Model for document upload."""
    folder_name: str = Field(..., description="Collection name (folder name)")
    doc_type: Literal["manual", "policy", "contract", "general"] = Field(
        default="general",
        description="Document type"
    )
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")


class DocumentUpdate(BaseModel):
    """Model for document metadata update."""
    title: Optional[str] = None
    description: Optional[str] = None
    doc_type: Optional[Literal["manual", "policy", "contract", "general"]] = None
    collection_name: Optional[str] = None


class DocumentResponse(BaseModel):
    """Model for document response."""
    id: str
    filename: str
    collection: str
    doc_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    source_path: str
    chunk_count: int
    upload_date: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Model for document list response."""
    documents: List[DocumentResponse]
    total: int


# Q&A Models
class QACreate(BaseModel):
    """Model for Q&A pair creation."""
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags")
    category: Optional[str] = Field(None, description="Optional category")


class QAResponse(BaseModel):
    """Model for Q&A response."""
    id: str
    question: str
    answer: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAListResponse(BaseModel):
    """Model for Q&A list response."""
    qa_pairs: List[QAResponse]
    total: int


# Query Models
class QueryRequest(BaseModel):
    """Model for query request."""
    query_text: str = Field(..., description="Query text")
    collections: Optional[List[str]] = Field(
        None,
        description="Specific collections to search (optional)"
    )
    doc_types: Optional[List[str]] = Field(
        None,
        description="Specific document types to filter (optional)"
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    include_qa: bool = Field(
        default=True,
        description="Whether to include Q&A pairs in search"
    )
    return_all: bool = Field(
        default=False,
        description="If false, return only best result. If true, return all results up to n_results"
    )


class DocumentCitation(BaseModel):
    """Citation model for document results."""
    source: str
    filename: str
    collection: str
    doc_type: str
    page: Optional[int] = None
    distance: float
    id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QACitation(BaseModel):
    """Citation model for Q&A results."""
    question: str
    answer: str
    qa_id: str
    distance: float
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Model for a single query result."""
    text: str
    source_type: Literal["document", "qa"]
    citations: List[DocumentCitation | QACitation]


class QueryResponse(BaseModel):
    """Model for query response."""
    query: str
    return_all: bool
    results: List[QueryResult]


# Collection Models
class CollectionCreate(BaseModel):
    """Model for collection creation."""
    collection_name: str = Field(..., description="Collection name")
    doc_type: Literal["manual", "policy", "contract", "general"] = Field(
        default="general",
        description="Default document type for this collection"
    )
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=128, ge=0, le=500)
    chunking_strategy: Literal["hybrid", "hierarchical"] = Field(
        default="hybrid",
        description="Chunking strategy"
    )


class CollectionResponse(BaseModel):
    """Model for collection response."""
    name: str
    doc_type: str
    document_count: int
    chunk_count: int
    docs_path: str


class CollectionListResponse(BaseModel):
    """Model for collection list response."""
    collections: List[CollectionResponse]
    total: int

