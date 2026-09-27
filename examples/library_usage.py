"""
Using rag_package directly, without the API.

Expects folders of documents under ./documents/<collection>/ and OPENAI_API_KEY
in the environment. Run from the repo root: python -m examples.library_usage
"""
import logging

from rag_package import MultiCollectionRAG

logging.basicConfig(level=logging.INFO)


def main():
    rag = MultiCollectionRAG(chroma_persist_dir="./chroma_db", cache_dir="./.rag_state")

    # Different document types get different chunking
    rag.add_collection("user_manuals", "./documents/user_manuals", doc_type="manual", chunking_strategy="hierarchical")
    rag.add_collection("policies", "./documents/policies", doc_type="policy", chunk_size=800)
    rag.add_collection("general", "./documents/general", doc_type="general")

    for name in rag.collections:
        print(name, rag.ingest_collection(name))  # unchanged files are skipped via SHA-256 cache

    # All collections, merged by distance
    for hit in rag.query_combined("How do I reset my password?", n_results=3):
        print(f"{hit['distance']:.3f}  {hit['metadata']['filename']}  {hit['text'][:80]!r}")

    # One collection only
    print(rag.query("installation steps", collections=["user_manuals"], n_results=2))

    # Filter by document type across collections
    print(rag.query_combined("vacation policy", doc_types=["policy"], n_results=2))


if __name__ == "__main__":
    main()
