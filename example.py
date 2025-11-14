from rag_package import MultiCollectionRAG
from dotenv import load_dotenv

load_dotenv()

# Initialize
rag = MultiCollectionRAG(
    chroma_persist_dir="./chroma_db",
    embedding_model="text-embedding-3-small"
)

# Add a collection (folder of documents)
rag.add_collection(
    collection_name="my_docs",
    docs_path="./documents/pdf",  # Your folder path
    doc_type="manual",
    chunk_size=600,
    chunking_strategy="hierarchical",
    save_recognized=True 
)

# Ingest documents (one-time setup)
rag.ingest_all()

# Query
results = rag.query_combined("waht is Alexandra passionate about?", n_results=5)
print(results)