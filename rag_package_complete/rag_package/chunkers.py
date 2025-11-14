from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker


def get_chunker(
    strategy: str = "hybrid",
    chunk_size: int = 512,
    chunk_overlap: int = 128
):
    """Get appropriate chunker based on strategy."""
    
    if strategy == "hierarchical":
        return HierarchicalChunker(
            max_tokens=chunk_size,
            merge_list_items=True
        )
    
    elif strategy == "hybrid":
        return HybridChunker(
            tokenizer="gpt-4",
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            merge_peers=True
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
