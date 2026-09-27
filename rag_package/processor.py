import logging
import tempfile
from pathlib import Path
from typing import Dict, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .chunkers import get_chunker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".html", ".md", ".txt"]


class DocumentProcessor:
    """Converts documents with Docling (OCR on for scanned PDFs) and chunks them."""

    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _convert(self, file_path: Path):
        """Docling has no plain-text reader, so .txt is converted as Markdown."""
        if file_path.suffix.lower() != ".txt":
            return self.converter.convert(str(file_path)).document

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / f"{file_path.stem}.md"
            md_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
            return self.converter.convert(str(md_path)).document

    def _save_recognized_text(self, document, file_path: Path) -> None:
        """Write the extracted text next to the source (recognize/<name>.txt) for OCR spot checks."""
        text = "\n\n".join(item.text for item in getattr(document, "texts", []) if getattr(item, "text", None))
        output_path = file_path.parent / "recognize" / f"{file_path.stem}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("Saved recognized text: %s", output_path)

    def process(
        self,
        file_path: Path,
        chunking_strategy: str,
        chunk_size: int,
        doc_type: str,
        collection_name: str,
        save_recognized: bool = False,
    ) -> List[Dict]:
        """Convert and chunk one document. Returns [{"text", "metadata"}, ...]."""
        document = self._convert(file_path)

        if save_recognized:
            try:
                self._save_recognized_text(document, file_path)
            except OSError:
                logger.warning("Could not save recognized text for %s", file_path, exc_info=True)

        chunker = get_chunker(strategy=chunking_strategy, chunk_size=chunk_size)

        chunks = []
        for chunk in chunker.chunk(document):
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "doc_type": doc_type,
                "collection": collection_name,
            }
            page = _first_page(chunk)
            if page is not None:  # Chroma rejects None metadata values
                metadata["page"] = page
            chunks.append({"text": chunk.text, "metadata": metadata})
        return chunks

    @staticmethod
    def get_supported_extensions() -> List[str]:
        return list(SUPPORTED_EXTENSIONS)


def _first_page(chunk):
    """Page number of the chunk's first source item, when the format has pages."""
    for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            if getattr(prov, "page_no", None) is not None:
                return prov.page_no
    return None
