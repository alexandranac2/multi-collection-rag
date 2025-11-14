"""Context management for request-scoped variables like user_id."""
from contextvars import ContextVar
from typing import Optional

# Context variable to store user_id per request
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


def get_user_id() -> Optional[str]:
    """Get the current request's user_id from context."""
    return user_id_context.get()


def set_user_id(user_id: Optional[str]) -> None:
    """Set the user_id for the current request context."""
    user_id_context.set(user_id)

