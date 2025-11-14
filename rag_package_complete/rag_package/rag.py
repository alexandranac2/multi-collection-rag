import os
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .collection import CollectionManager

load_dotenv()


class MultiCollectionRAG:
    """Multi-collection RAG system."""
    
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None
    ):
        # Setup OpenAI
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        
        # Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collections = {}
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]
    
    def add_collection(
        self,
        collection_name: str,
        docs_path: str,
        doc_type: Literal["manual", "policy", "contract", "general"] = "general",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        chunking_strategy: Literal["hybrid", "hierarchical"] = "hybrid",
        save_recognized: bool = False
    ):
        """Add a document collection."""
        # Create ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "doc_type": doc_type}
        )
        
        # Create collection manager
        self.collections[collection_name] = CollectionManager(
            collection=chroma_collection,
            collection_name=collection_name,
            docs_path=docs_path,
            doc_type=doc_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            embed_func=self._embed_texts,
            save_recognized=save_recognized
        )
        
        print(f"✅ Added: {collection_name} ({doc_type})")
        return self
    
    def ingest_collection(self, collection_name: str, force_reindex: bool = False):
        """Ingest a specific collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        self.collections[collection_name].ingest(force_reindex)
    
    def ingest_all(self, force_reindex: bool = False):
        """Ingest all collections."""
        for coll in self.collections.values():
            coll.ingest(force_reindex)
    
    def query(
        self,
        query_text: str,
        collections: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        n_results: int = 5
    ) -> Dict[str, List[Dict]]:
        """Query collections."""
        target_colls = collections or list(self.collections.keys())
        
        # Build filter
        where_clause = {}
        if doc_types:
            where_clause["doc_type"] = {"$in": doc_types}
        
        # Embed query
        query_embedding = self._embed_texts([query_text])[0]
        
        # Query each collection
        results = {}
        for coll_name in target_colls:
            if coll_name in self.collections:
                results[coll_name] = self.collections[coll_name].query(
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where_clause=where_clause if where_clause else None
                )
        
        return results
    
    def query_combined(
        self,
        query_text: str,
        collections: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """Query and combine results."""
        all_results = self.query(query_text, collections, doc_types, n_results)
        
        # Combine and sort
        combined = []
        for results in all_results.values():
            combined.extend(results)
        
        combined.sort(key=lambda x: x['distance'])
        return combined[:n_results]
    
    def list_collections(self):
        """List all collections."""
        print("\n📚 Collections:")
        for name, coll in self.collections.items():
            print(f"  • {name}: {coll.count()} chunks ({coll.doc_type})")
