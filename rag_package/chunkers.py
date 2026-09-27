from functools import lru_cache

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

# Same encoding as OpenAI's text-embedding-3 models, so chunk_size is measured
# in the tokens the embedding model actually sees.
EMBEDDING_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding(EMBEDDING_ENCODING)


def get_chunker(strategy: str = "hybrid", chunk_size: int = 512):
    """
    hierarchical: one chunk per structural element (section, paragraph, table, list);
                  no size limit, so chunk_size does not apply.
    hybrid:       structure-aware like hierarchical, then oversized chunks are split and
                  small neighbours under the same heading merged, up to chunk_size tokens.
    """
    if strategy == "hierarchical":
        return HierarchicalChunker(merge_list_items=True)
    if strategy == "hybrid":
        tokenizer = OpenAITokenizer(tokenizer=_encoding(), max_tokens=chunk_size)
        return HybridChunker(tokenizer=tokenizer, merge_peers=True)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
