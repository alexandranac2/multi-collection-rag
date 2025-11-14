"""
Utility functions for the RAG package.
"""


def format_distance(distance: float) -> str:
    """Format distance score for display."""
    return f"{distance:.4f}"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
