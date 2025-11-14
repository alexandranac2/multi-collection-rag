"""Query service for searching both documents and Q&A pairs."""
from typing import List, Dict, Optional
from rag_package import MultiCollectionRAG

from app.config import settings
from app.models import QueryRequest, QueryResponse, QueryResult, DocumentCitation, QACitation
from app.services.qa_service import QAService
from app.services.langfuse_service import langfuse_service


class QueryService:
    """Service for querying RAG system."""
    
    def __init__(self, rag_instance: MultiCollectionRAG, qa_service: QAService):
        """Initialize query service."""
        self.rag = rag_instance
        self.qa_service = qa_service
    
    def query(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Execute query against both documents and Q&A."""
        trace = None
        if langfuse_service.enabled:
            trace = langfuse_service.create_trace(
                name="rag_query",
                metadata={
                    "query": request.query_text[:100],
                    "include_qa": request.include_qa,
                    "return_all": request.return_all
                }
            )
        
        try:
            all_results = []
            
            # Debug: Check if collections exist
            print(f"DEBUG: Query '{request.query_text}' - Collections available: {list(self.rag.collections.keys())}")
            
            # Step 1: Search document collections
            if langfuse_service.enabled and trace:
                doc_span = langfuse_service.create_span(
                    trace=trace,
                    name="document_search"
                )
            
            # Only query documents if RAG has collections
            doc_results = []
            if self.rag.collections:
                try:
                    print(f"DEBUG: Querying documents with n_results={request.n_results}")
                    doc_results = self.rag.query_combined(
                        query_text=request.query_text,
                        collections=request.collections,
                        doc_types=request.doc_types,
                        n_results=request.n_results
                    )
                    print(f"DEBUG: Found {len(doc_results)} document results")
                except Exception as e:
                    print(f"Warning: Error querying documents: {e}")
                    import traceback
                    traceback.print_exc()
                    doc_results = []
            else:
                print("DEBUG: No collections available for document search")
            
            # Format document results
            for result in doc_results:
                metadata = result.get('metadata', {})
                # Extract filename from source path if not in metadata
                source_path = metadata.get('source', '')
                filename = metadata.get('filename', '')
                if not filename and source_path:
                    from pathlib import Path
                    filename = Path(source_path).name
                
                citation = DocumentCitation(
                    source=source_path,
                    filename=filename,
                    collection=result.get('collection', metadata.get('collection', '')),
                    doc_type=metadata.get('doc_type', 'general'),
                    page=metadata.get('page'),
                    distance=result.get('distance', 0.0),
                    id=result.get('id', ''),
                    metadata=metadata
                )
                
                all_results.append(QueryResult(
                    text=result.get('text', ''),
                    source_type="document",
                    citations=[citation]
                ))
            
            if langfuse_service.enabled and trace:
                langfuse_service.track_retrieval(
                    trace=trace,
                    query=request.query_text,
                    results_count=len(doc_results),
                    collection="documents",
                    metadata={"collections": request.collections or "all"}
                )
            
            # Step 2: Search Q&A collection if enabled
            qa_results = []
            if request.include_qa:
                if langfuse_service.enabled and trace:
                    qa_span = langfuse_service.create_span(
                        trace=trace,
                        name="qa_search"
                    )
                
                try:
                    print(f"DEBUG: Querying Q&A with n_results={request.n_results}")
                    qa_results = self.qa_service.query_qa(
                        query_text=request.query_text,
                        n_results=request.n_results
                    )
                    print(f"DEBUG: Found {len(qa_results)} Q&A results")
                except Exception as e:
                    print(f"Warning: Error querying Q&A: {e}")
                    import traceback
                    traceback.print_exc()
                    qa_results = []
                
                # Format Q&A results
                for result in qa_results:
                    result_metadata = result.get('metadata', {})
                    # Handle tags - could be string or list
                    tags = result_metadata.get('tags', '')
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
                    elif not isinstance(tags, list):
                        tags = []
                    
                    citation = QACitation(
                        question=result.get('question', result_metadata.get('question', '')),
                        answer=result.get('answer', result_metadata.get('answer', '')),
                        qa_id=result.get('qa_id', result.get('id', result_metadata.get('qa_id', ''))),
                        distance=result.get('distance', 0.0),
                        tags=tags,
                        metadata=result_metadata
                    )
                    
                    all_results.append(QueryResult(
                        text=result.get('text', ''),
                        source_type="qa",
                        citations=[citation]
                    ))
            
            print(f"DEBUG: Total results before sorting: {len(all_results)}")
            
            # Step 3: Sort all results by distance (lower is better)
            # Only sort if we have results with citations
            if all_results:
                all_results.sort(key=lambda x: x.citations[0].distance if x.citations and len(x.citations) > 0 else float('inf'))
            
            # Step 4: Apply return_all logic
            if not request.return_all:
                # Return only the best result
                final_results = all_results[:1] if all_results else []
            else:
                # Return all results up to n_results
                final_results = all_results[:request.n_results] if all_results else []
            
            print(f"DEBUG: Final results count: {len(final_results)}")
            
            if langfuse_service.enabled and trace:
                langfuse_service.track_query(
                    trace=trace,
                    query_text=request.query_text,
                    results=[r.model_dump() if hasattr(r, 'model_dump') else r.dict() for r in final_results],
                    metadata={
                        "total_results": len(all_results),
                        "returned_results": len(final_results),
                        "return_all": request.return_all
                    }
                )
                trace.update(output={"results_count": len(final_results), "success": True})
            
            return QueryResponse(
                query=request.query_text,
                return_all=request.return_all,
                results=final_results
            )
        except Exception as e:
            if langfuse_service.enabled and trace:
                trace.update(output={"success": False, "error": str(e)})
            raise

