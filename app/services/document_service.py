"""Document service for managing document uploads and ingestion."""
import os
import uuid
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import UploadFile

from rag_package import MultiCollectionRAG

from app.config import settings
from app.models import DocumentCreate, DocumentUpdate, DocumentResponse
from app.services.langfuse_service import langfuse_service


class DocumentService:
    """Service for managing documents."""
    
    def __init__(self, rag_instance: MultiCollectionRAG):
        """Initialize document service with RAG instance."""
        self.rag = rag_instance
        self.documents_base_path = Path(settings.DOCUMENTS_BASE_PATH)
        self.documents_base_path.mkdir(parents=True, exist_ok=True)
        
        # Document registry (JSON-based persistence)
        self.registry_path = Path(".document_registry.json")
        self.document_registry: Dict[str, Dict] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load document registry from JSON file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_registry(self):
        """Save document registry to JSON file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.document_registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save document registry: {e}")
    
    def upload_document(
        self,
        file: UploadFile,
        folder_name: str,
        doc_type: str = "general",
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> DocumentResponse:
        """Upload and ingest a document."""
        trace = None
        if langfuse_service.enabled:
            trace = langfuse_service.create_trace(
                name="document_upload",
                metadata={"filename": file.filename, "folder_name": folder_name}
            )
        
        try:
            # Create folder if it doesn't exist
            folder_path = self.documents_base_path / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename to avoid conflicts
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            file_path = folder_path / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Ensure collection exists
            if folder_name not in self.rag.collections:
                self.rag.add_collection(
                    collection_name=folder_name,
                    docs_path=str(folder_path),
                    doc_type=doc_type,
                    chunk_size=512,
                    chunk_overlap=128,
                    chunking_strategy="hierarchical"
                )
            
            # Process single file (force reindex to ensure it's added to vector DB)
            collection_manager = self.rag.collections[folder_name]
            print(f"📄 Processing and indexing file: {file_path.name}")
            
            try:
                collection_manager._process_file(file_path, force_reindex=True)
            except Exception as e:
                print(f"❌ Error processing file {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                raise Exception(f"Failed to process document: {e}")
            
            # Get chunk count for this specific document
            # Count chunks that match this source file
            chunk_count = 0
            try:
                results = collection_manager.collection.get(where={"source": str(file_path)})
                chunk_count = len(results['ids']) if results['ids'] else 0
                print(f"✅ Document indexed with {chunk_count} chunks")
            except Exception as e:
                print(f"⚠️  Warning: Could not count chunks for {file_path.name}: {e}")
                # Try alternative method - get all and filter
                try:
                    all_results = collection_manager.collection.get()
                    if all_results and all_results.get('ids'):
                        # Count IDs that match our file
                        source_str = str(file_path)
                        chunk_count = sum(1 for i, metadata in enumerate(all_results.get('metadatas', [])) 
                                        if metadata.get('source') == source_str)
                except:
                    # Final fallback
                    chunk_count = collection_manager.collection.count()
                print(f"✅ Document indexed (chunk count: {chunk_count})")
            
            # Create document ID
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Register document
            document_data = {
                "id": doc_id,
                "filename": file.filename,
                "unique_filename": unique_filename,
                "collection": folder_name,
                "doc_type": doc_type,
                "title": title or file.filename,
                "description": description,
                "source_path": str(file_path),
                "chunk_count": chunk_count,
                "upload_date": datetime.now().isoformat(),
                "metadata": {}
            }
            self.document_registry[doc_id] = document_data
            self._save_registry()
            
            if trace and langfuse_service.enabled:
                trace.update(output={"document_id": doc_id, "success": True})
            
            return DocumentResponse(**document_data)
        except Exception as e:
            if trace and langfuse_service.enabled:
                trace.update(output={"success": False, "error": str(e)})
            raise
    
    def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """Get document by ID."""
        doc_data = self.document_registry.get(document_id)
        if not doc_data:
            return None
        
        # Update chunk count
        if doc_data["collection"] in self.rag.collections:
            collection_manager = self.rag.collections[doc_data["collection"]]
            doc_data["chunk_count"] = collection_manager.collection.count()
        
        # Convert ISO string back to datetime
        if isinstance(doc_data.get("upload_date"), str):
            doc_data["upload_date"] = datetime.fromisoformat(doc_data["upload_date"])
        
        return DocumentResponse(**doc_data)
    
    def list_documents(self, collection: Optional[str] = None) -> List[DocumentResponse]:
        """List all documents, optionally filtered by collection."""
        documents = []
        for doc_data in self.document_registry.values():
            if collection and doc_data["collection"] != collection:
                continue
            
            # Update chunk count
            if doc_data["collection"] in self.rag.collections:
                collection_manager = self.rag.collections[doc_data["collection"]]
                doc_data["chunk_count"] = collection_manager.collection.count()
            
            # Convert ISO string back to datetime
            if isinstance(doc_data.get("upload_date"), str):
                doc_data["upload_date"] = datetime.fromisoformat(doc_data["upload_date"])
            
            documents.append(DocumentResponse(**doc_data))
        
        return documents
    
    def update_document(self, document_id: str, update_data: DocumentUpdate) -> Optional[DocumentResponse]:
        """Update document metadata."""
        doc_data = self.document_registry.get(document_id)
        if not doc_data:
            return None
        
        # Update fields
        if update_data.title is not None:
            doc_data["title"] = update_data.title
        if update_data.description is not None:
            doc_data["description"] = update_data.description
        if update_data.doc_type is not None:
            doc_data["doc_type"] = update_data.doc_type
        
        # Handle collection change
        if update_data.collection_name and update_data.collection_name != doc_data["collection"]:
            old_collection = doc_data["collection"]
            new_collection = update_data.collection_name
            
            # Move file
            old_path = Path(doc_data["source_path"])
            new_folder = self.documents_base_path / new_collection
            new_folder.mkdir(parents=True, exist_ok=True)
            new_path = new_folder / old_path.name
            
            shutil.move(str(old_path), str(new_path))
            
            # Remove from old collection
            if old_collection in self.rag.collections:
                collection_manager = self.rag.collections[old_collection]
                collection_manager.collection.delete(where={"source": str(old_path)})
            
            # Add to new collection
            if new_collection not in self.rag.collections:
                self.rag.add_collection(
                    collection_name=new_collection,
                    docs_path=str(new_folder),
                    doc_type=doc_data["doc_type"]
                )
            
            new_collection_manager = self.rag.collections[new_collection]
            print(f"📄 Reindexing moved document: {new_path.name}")
            new_collection_manager._process_file(new_path, force_reindex=True)
            
            doc_data["collection"] = new_collection
            doc_data["source_path"] = str(new_path)
        
        self._save_registry()
        
        # Convert ISO string back to datetime
        if isinstance(doc_data.get("upload_date"), str):
            doc_data["upload_date"] = datetime.fromisoformat(doc_data["upload_date"])
        
        return DocumentResponse(**doc_data)
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        doc_data = self.document_registry.get(document_id)
        if not doc_data:
            return False
        
        try:
            # Remove from filesystem
            file_path = Path(doc_data["source_path"])
            if file_path.exists():
                file_path.unlink()
            
            # Remove from ChromaDB
            if doc_data["collection"] in self.rag.collections:
                collection_manager = self.rag.collections[doc_data["collection"]]
                collection_manager.collection.delete(where={"source": str(file_path)})
            
            # Remove from registry
            del self.document_registry[document_id]
            self._save_registry()
            
            return True
        except Exception:
            return False

