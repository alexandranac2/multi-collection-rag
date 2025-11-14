"""LangFuse observability service for tracking RAG operations."""
from typing import Optional, Dict, Any, List
from functools import wraps
import time
import uuid

from app.config import settings

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
            except Exception as e:
                print(f"Warning: Failed to initialize LangFuse: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None
    
    def create_trace(self, name: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Create a new trace."""
        if not self.enabled or not self.client:
            return MockTrace()
        
        try:
            # Try to use the trace method if it exists
            if hasattr(self.client, 'trace'):
                trace = self.client.trace(
                    name=name,
                    user_id=user_id,
                    metadata=metadata or {}
                )
                # Ensure trace has an id attribute
                if not hasattr(trace, 'id'):
                    trace.id = str(uuid.uuid4())
                return trace
            else:
                # Fallback: create a mock trace
                return MockTrace()
        except Exception as e:
            print(f"Warning: Failed to create LangFuse trace: {e}")
            return MockTrace()
    
    def create_span(self, trace_id: str, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a span within a trace."""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Handle trace object or trace ID string
            actual_trace_id = trace_id.id if hasattr(trace_id, 'id') else str(trace_id)
            
            if hasattr(self.client, 'span'):
                span = self.client.span(
                    trace_id=actual_trace_id,
                    name=name,
                    metadata=metadata or {}
                )
                return span
        except Exception as e:
            print(f"Warning: Failed to create LangFuse span: {e}")
        
        return None
    
    def track_embedding(self, trace_id: str, texts: List[str], model: str, usage: Optional[Dict] = None):
        """Track embedding generation."""
        if not self.enabled or not self.client:
            return
        
        try:
            # Handle trace object or trace ID string
            actual_trace_id = trace_id.id if hasattr(trace_id, 'id') else str(trace_id)
            
            if hasattr(self.client, 'generation'):
                self.client.generation(
                    trace_id=actual_trace_id,
                    name="embedding_generation",
                    model=model,
                    input=texts,
                    usage=usage,
                    metadata={"model": model, "text_count": len(texts)}
                )
        except Exception as e:
            print(f"Warning: Failed to track embedding in LangFuse: {e}")
    
    def track_retrieval(self, trace_id: str, query: str, results_count: int, collection: str, metadata: Optional[Dict] = None):
        """Track vector retrieval operation."""
        if not self.enabled or not self.client:
            return
        
        try:
            # Handle trace object or trace ID string
            actual_trace_id = trace_id.id if hasattr(trace_id, 'id') else str(trace_id)
            
            if hasattr(self.client, 'span'):
                self.client.span(
                    trace_id=actual_trace_id,
                    name="vector_retrieval",
                    input={"query": query, "collection": collection},
                    metadata={
                        "results_count": results_count,
                        "collection": collection,
                        **(metadata or {})
                    }
                )
        except Exception as e:
            print(f"Warning: Failed to track retrieval in LangFuse: {e}")
    
    def track_query(self, trace_id: str, query_text: str, results: List[Dict], metadata: Optional[Dict] = None):
        """Track query operation."""
        if not self.enabled or not self.client:
            return
        
        try:
            # Handle trace object or trace ID string
            actual_trace_id = trace_id.id if hasattr(trace_id, 'id') else str(trace_id)
            
            if hasattr(self.client, 'generation'):
                self.client.generation(
                    trace_id=actual_trace_id,
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
                if hasattr(self.client, 'flush'):
                    self.client.flush()
            except Exception as e:
                print(f"Warning: Failed to flush LangFuse events: {e}")


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

