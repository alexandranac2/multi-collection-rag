from pathlib import Path
from typing import List, Dict, Optional
from .cache import CacheManager
from .processor import DocumentProcessor


class CollectionManager:
    """Manages a single document collection."""
    
    def __init__(
        self,
        collection,
        collection_name: str,
        docs_path: str,
        doc_type: str,
        chunk_size: int,
        chunk_overlap: int,
        chunking_strategy: str,
        embed_func,
        save_recognized: bool = False
    ):
        self.collection = collection
        self.name = collection_name
        self.docs_path = Path(docs_path)
        self.doc_type = doc_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.embed_func = embed_func
        self.save_recognized = save_recognized
        
        # Initialize cache and processor
        self.cache = CacheManager(Path(f".{collection_name}_cache.json"))
        self.processor = DocumentProcessor()
    
    def ingest(self, force_reindex: bool = False):
        """Ingest all documents in this collection."""
        if not self.docs_path.exists():
            raise ValueError(f"Path {self.docs_path} does not exist")
        
        # Find all supported files (exclude recognize folder)
        files_to_process = []
        for ext in self.processor.get_supported_extensions():
            for file_path in self.docs_path.rglob(f"*{ext}"):
                # Skip files in recognize folder
                if "recognize" not in file_path.parts:
                    files_to_process.append(file_path)
        
        print(f"\n📁 Collection: {self.name}")
        print(f"Found {len(files_to_process)} documents")
        
        for file_path in files_to_process:
            self._process_file(file_path, force_reindex)
    
    def _process_file(self, file_path: Path, force_reindex: bool):
        """Process a single file."""
        # Check cache
        file_hash = self.cache.calculate_file_hash(file_path)
        file_key = str(file_path)
        
        if not force_reindex and self.cache.get(file_key) == file_hash:
            print(f"⏭️  Skipping {file_path.name}")
            return
        
        try:
            print(f"📄 Processing {file_path.name}...")
            
            # Process document
            chunks = self.processor.process(
                file_path=file_path,
                chunking_strategy=self.chunking_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                doc_type=self.doc_type,
                collection_name=self.name,
                save_recognized=self.save_recognized
            )
            
            if not chunks:
                print(f"⚠️  No content from {file_path.name}")
                return
            
            # Prepare for ChromaDB
            texts = [c["text"] for c in chunks]
            embeddings = self.embed_func(texts)
            metadatas = [c["metadata"] for c in chunks]
            ids = [f"{self.name}_{file_path.stem}_{i}" for i in range(len(chunks))]
            
            # Remove old chunks
            try:
                self.collection.delete(where={"source": str(file_path)})
            except:
                pass
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update cache
            self.cache.set(file_key, file_hash)
            self.cache.save()
            
            print(f"✅ Indexed {file_path.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            print(f"❌ Error: {file_path.name}: {e}")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_clause: Optional[Dict] = None
    ) -> List[Dict]:
        """Query this collection."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i],
                    'collection': self.name
                })
        
        return formatted
    
    def count(self) -> int:
        """Get number of chunks in collection."""
        return self.collection.count()
