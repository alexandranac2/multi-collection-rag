"""Query service: one ranked answer list across document collections and Q&A pairs."""

import logging
from pathlib import Path

from app.models import DocumentCitation, QACitation, QueryRequest, QueryResponse, QueryResult
from app.services.langfuse_service import langfuse_service
from app.services.qa_service import QAService
from rag_package import MultiCollectionRAG

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, rag_instance: MultiCollectionRAG, qa_service: QAService):
        self.rag = rag_instance
        self.qa_service = qa_service

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Search documents and (optionally) curated Q&A pairs, merge them by cosine
        distance, and return the best hit or the top `n_results`.
        """
        trace = langfuse_service.create_trace(
            name="rag_query",
            metadata={
                "query": request.query_text[:100],
                "include_qa": request.include_qa,
                "return_all": request.return_all,
            },
        )
        try:
            results = self._search_documents(request)
            langfuse_service.track_retrieval(
                trace=trace,
                query=request.query_text,
                results_count=len(results),
                collection="documents",
                metadata={"collections": request.collections or "all"},
            )
            if request.include_qa:
                results += self._search_qa(request)

            # Both sources use cosine distance, so they rank on one scale (lower = closer)
            results.sort(key=lambda r: r.citations[0].distance)
            final = results[: request.n_results if request.return_all else 1]

            langfuse_service.track_query(
                trace=trace,
                query_text=request.query_text,
                results=[r.model_dump() for r in final],
                metadata={"total_results": len(results), "returned_results": len(final)},
            )
            trace.update(output={"results_count": len(final), "success": True})
            return QueryResponse(query=request.query_text, return_all=request.return_all, results=final)
        except Exception as exc:
            trace.update(output={"success": False, "error": type(exc).__name__})
            raise

    def _search_documents(self, request: QueryRequest) -> list[QueryResult]:
        if not self.rag.collections:
            return []

        unknown = set(request.collections or []) - set(self.rag.collections)
        if unknown:
            logger.info("Query named unknown collections: %s", sorted(unknown))

        hits = self.rag.query_combined(
            query_text=request.query_text,
            collections=request.collections,
            doc_types=request.doc_types,
            n_results=request.n_results,
        )
        results = []
        for hit in hits:
            metadata = hit.get("metadata", {})
            source = metadata.get("source", "")
            results.append(
                QueryResult(
                    text=hit.get("text", ""),
                    source_type="document",
                    citations=[
                        DocumentCitation(
                            source=source,
                            filename=metadata.get("filename") or Path(source).name,
                            collection=hit.get("collection", metadata.get("collection", "")),
                            doc_type=metadata.get("doc_type", "general"),
                            page=metadata.get("page"),
                            distance=hit.get("distance", 0.0),
                            id=hit.get("id", ""),
                            metadata=metadata,
                        )
                    ],
                )
            )
        return results

    def _search_qa(self, request: QueryRequest) -> list[QueryResult]:
        results = []
        for hit in self.qa_service.query_qa(query_text=request.query_text, n_results=request.n_results):
            metadata = hit.get("metadata", {})
            tags = metadata.get("tags", "")
            results.append(
                QueryResult(
                    text=hit.get("text", ""),
                    source_type="qa",
                    citations=[
                        QACitation(
                            question=hit["question"],
                            answer=hit["answer"],
                            qa_id=hit["qa_id"],
                            distance=hit.get("distance", 0.0),
                            tags=[t.strip() for t in tags.split(",") if t.strip()],
                            metadata=metadata,
                        )
                    ],
                )
            )
        return results
