from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter
from .chunkers import get_chunker


class DocumentProcessor:
    """Handles document conversion and chunking."""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def process(
        self,
        file_path: Path,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        doc_type: str,
        collection_name: str,
        save_recognized: bool = False
    ) -> List[Dict]:
        """
        Process document: convert and chunk.
        
        Returns list of chunks with metadata.
        """
        # Convert document
        result = self.converter.convert(str(file_path))
        
        # Save recognized text if requested
        # Note: This only runs when file is processed (not cached), so recognize files
        # are only saved/replaced when the source file is new or changed
        if save_recognized:
            try:
                # Extract all text from document texts
                text_parts = []
                if hasattr(result.document, 'texts') and result.document.texts:
                    for text_item in result.document.texts:
                        if hasattr(text_item, 'text') and text_item.text:
                            text_parts.append(text_item.text)
                
                # Join all text parts with double newlines for readability
                full_text = "\n\n".join(text_parts) if text_parts else ""
                
                # Create output path: same folder as source file / recognize / filename.txt
                output_path = file_path.parent / "recognize" / f"{file_path.stem}.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write mode 'w' will replace existing file if it exists
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                
                print(f"💾 Saved recognized text: {output_path}")
            except Exception as e:
                print(f"⚠️  Failed to save recognized text: {e}")
        
        # Get chunker
        chunker = get_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Chunk document
        chunks = []
        for chunk in chunker.chunk(result.document):
            # Build metadata dictionary
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "doc_type": doc_type,
                "collection": collection_name,
            }
            
            # Only add page if it exists and is not None
            page = getattr(chunk, 'page', None)
            if page is not None:
                metadata["page"] = page
            
            # Filter out any None values as safety measure (ChromaDB doesn't accept None)
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            chunks.append({
                "text": chunk.text,
                "metadata": metadata
            })
        
        return chunks
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return list of supported file extensions."""
        return ['.pdf', '.docx', '.pptx', '.html', '.md', '.txt']
