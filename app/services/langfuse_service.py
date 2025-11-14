"""LangFuse observability service for tracking RAG operations."""
from typing import Optional, Dict, Any, List
from functools import wraps
import uuid

from app.config import settings
from app.context import get_user_id

# Try to import LangFuse, but handle gracefully if not available
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


class MockTrace:
    """Mock trace object for when LangFuse is disabled or unavailable."""
    def __init__(self, trace_id: str = None):
        self.id = trace_id or str(uuid.uuid4())
    
    def update(self, **kwargs):
        """No-op update method."""
        pass
    
    def span(self, **kwargs):
        """Return a mock span."""
        return MockSpan()
    
    def generation(self, **kwargs):
        """No-op generation method."""
        pass


class MockSpan:
    """Mock span object."""
    def update(self, **kwargs):
        """No-op update method."""
        pass


class LangFuseService:
    """Service for LangFuse observability."""
    
    def __init__(self):
        """Initialize LangFuse client if keys are provided."""
        self.enabled = bool(
            LANGFUSE_AVAILABLE and 
            settings.LANGFUSE_PUBLIC_KEY and 
            settings.LANGFUSE_SECRET_KEY
        )
        
        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                # Verify credentials (only on startup, not in production)
                try:
                    self.client.auth_check()
                    print("✅ LangFuse initialized and credentials verified successfully")
                except Exception as auth_error:
                    print(f"⚠️  LangFuse initialized but auth check failed: {auth_error}")
                    print("   This might indicate incorrect API keys or network issues")
            except Exception as e:
                print(f"❌ Warning: Failed to initialize LangFuse: {e}")
                import traceback
                traceback.print_exc()
                self.enabled = False
                self.client = None
        else:
            if not LANGFUSE_AVAILABLE:
                print("⚠️  LangFuse disabled: langfuse package not installed")
            elif not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
                print("⚠️  LangFuse disabled: Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY in .env")
            self.client = None
    
    def create_trace(self, name: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Create a new trace using the correct LangFuse API."""
        if not self.enabled or not self.client:
            print(f"⚠️  LangFuse trace '{name}' skipped: LangFuse not enabled")
            return MockTrace()
        
        # Auto-get user_id from context if not explicitly provided
        if user_id is None:
            user_id = get_user_id()
        
        try:
            # Correct LangFuse API: langfuse.trace() returns a trace object
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                metadata=metadata or {}
            )
            print(f"✅ LangFuse trace created: '{name}' for user '{user_id}'")
            return trace
        except Exception as e:
            print(f"❌ Warning: Failed to create LangFuse trace '{name}': {e}")
            import traceback
            traceback.print_exc()
            return MockTrace()
    
    def create_span(self, trace, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a span within a trace using the correct LangFuse API."""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Correct LangFuse API: trace.span() creates a span within the trace
            span = trace.span(
                name=name,
                metadata=metadata or {}
            )
            return span
        except Exception as e:
            print(f"Warning: Failed to create LangFuse span: {e}")
            return None
    
    def track_embedding(self, trace, texts: List[str], model: str, usage: Optional[Dict] = None):
        """Track embedding generation using the correct LangFuse API."""
        if not self.enabled or not self.client:
            return
        
        try:
            # Correct LangFuse API: trace.generation() tracks LLM/embedding calls
            trace.generation(
                name="embedding_generation",
                model=model,
                input=texts,
                usage=usage,
                metadata={"model": model, "text_count": len(texts)}
            )
        except Exception as e:
            print(f"Warning: Failed to track embedding in LangFuse: {e}")
    
    def track_retrieval(self, trace, query: str, results_count: int, collection: str, metadata: Optional[Dict] = None):
        """Track vector retrieval operation using the correct LangFuse API."""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Correct LangFuse API: trace.span() for retrieval operations
            span = trace.span(
                name="vector_retrieval",
                input={"query": query, "collection": collection},
                metadata={
                    "results_count": results_count,
                    "collection": collection,
                    **(metadata or {})
                }
            )
            return span
        except Exception as e:
            print(f"Warning: Failed to track retrieval in LangFuse: {e}")
            return None
    
    def track_query(self, trace, query_text: str, results: List[Dict], metadata: Optional[Dict] = None):
        """Track query operation using the correct LangFuse API."""
        if not self.enabled or not self.client:
            return
        
        try:
            # Correct LangFuse API: trace.generation() for query operations
            trace.generation(
                name="rag_query",
                input={"query": query_text},
                output={"results_count": len(results)},
                metadata={
                    "results_count": len(results),
                    **(metadata or {})
                }
            )
        except Exception as e:
            print(f"Warning: Failed to track query in LangFuse: {e}")
    
    def flush(self):
        """Flush pending events to LangFuse."""
        if self.enabled and self.client:
            try:
                # Flush events to LangFuse
                self.client.flush()
                print("✅ LangFuse events flushed successfully")
            except Exception as e:
                print(f"❌ Warning: Failed to flush LangFuse events: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  LangFuse flush skipped: LangFuse not enabled")
    
    def shutdown(self):
        """Shutdown LangFuse client and ensure all events are sent."""
        if self.enabled and self.client:
            try:
                # Shutdown is blocking and ensures all events are sent
                self.client.shutdown()
                print("✅ LangFuse client shut down successfully")
            except Exception as e:
                print(f"⚠️  Warning: Failed to shutdown LangFuse client: {e}")


# Global instance
langfuse_service = LangFuseService()


def trace_operation(operation_name: str):
    """Decorator to trace an operation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if langfuse_service.enabled:
                trace = langfuse_service.create_trace(
                    name=operation_name,
                    metadata={"function": func.__name__}
                )
                try:
                    result = func(*args, **kwargs)
                    if trace:
                        trace.update(output={"success": True})
                    return result
                except Exception as e:
                    if trace:
                        trace.update(output={"success": False, "error": str(e)})
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

