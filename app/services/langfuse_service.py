"""Optional LangFuse tracing. Every method is a safe no-op when tracing is off."""

import logging
import uuid
from typing import Any

from app.config import settings
from app.context import get_user_id

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except ImportError:  # tracing is optional
    Langfuse = None


class MockTrace:
    """Stands in for a LangFuse trace so callers never need to branch."""

    def __init__(self, trace_id: str | None = None):
        self.id = trace_id or str(uuid.uuid4())

    def update(self, **kwargs):
        pass

    def span(self, **kwargs):
        return MockSpan()

    def generation(self, **kwargs):
        pass


class MockSpan:
    def update(self, **kwargs):
        pass


class LangFuseService:
    """Thin wrapper over the LangFuse v2 SDK (client.trace / trace.span / trace.generation)."""

    def __init__(self):
        self.client = None
        self.enabled = False

        if Langfuse is None:
            logger.info("LangFuse disabled: package not installed")
            return
        if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
            logger.info("LangFuse disabled: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
            return

        try:
            self.client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            self.enabled = True
        except Exception:
            logger.exception("LangFuse initialisation failed; tracing disabled")

    def create_trace(self, name: str, user_id: str | None = None, metadata: dict[str, Any] | None = None):
        if not self.enabled:
            return MockTrace()
        try:
            return self.client.trace(name=name, user_id=user_id or get_user_id(), metadata=metadata or {})
        except Exception:
            logger.warning("Could not create LangFuse trace %r", name, exc_info=True)
            return MockTrace()

    def create_span(self, trace, name: str, metadata: dict[str, Any] | None = None):
        if not self.enabled:
            return None
        try:
            return trace.span(name=name, metadata=metadata or {})
        except Exception:
            logger.warning("Could not create LangFuse span %r", name, exc_info=True)
            return None

    def track_embedding(self, trace, texts: list[str], model: str, usage: dict | None = None):
        if not self.enabled:
            return
        try:
            trace.generation(
                name="embedding_generation",
                model=model,
                input=texts,
                usage=usage,
                metadata={"model": model, "text_count": len(texts)},
            )
        except Exception:
            logger.warning("Could not track embedding in LangFuse", exc_info=True)

    def track_retrieval(self, trace, query: str, results_count: int, collection: str, metadata: dict | None = None):
        if not self.enabled:
            return None
        try:
            return trace.span(
                name="vector_retrieval",
                input={"query": query, "collection": collection},
                metadata={"results_count": results_count, "collection": collection, **(metadata or {})},
            )
        except Exception:
            logger.warning("Could not track retrieval in LangFuse", exc_info=True)
            return None

    def track_query(self, trace, query_text: str, results: list[dict], metadata: dict | None = None):
        if not self.enabled:
            return
        try:
            trace.generation(
                name="rag_query",
                input={"query": query_text},
                output={"results_count": len(results)},
                metadata={"results_count": len(results), **(metadata or {})},
            )
        except Exception:
            logger.warning("Could not track query in LangFuse", exc_info=True)

    def flush(self):
        if self.enabled:
            try:
                self.client.flush()
            except Exception:
                logger.warning("LangFuse flush failed", exc_info=True)

    def shutdown(self):
        if self.enabled:
            try:
                self.client.shutdown()
            except Exception:
                logger.warning("LangFuse shutdown failed", exc_info=True)


langfuse_service = LangFuseService()
