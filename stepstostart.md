# Steps to Start - Installation Guide

## Complete Installation Steps

### 1. Create Project Folder
```bash
mkdir my_project
cd my_project
```

### 2. Copy `rag_package_complete` Folder
```bash
cp -r /path/to/rag_package_complete .
```

### 3. Create Main `requirements.txt` (Optional)
If you need the main project dependencies (FastAPI, LangChain, etc.), create a root `requirements.txt`:

```txt
# Main project dependencies (if needed)
langchain
langchain-community
langchain-openai
langgraph
fastapi
uvicorn[standard]
python-dotenv
docling
pymupdf
pdfminer.six
sentence-transformers
faiss-cpu
chromadb
numpy
tiktoken
requests
pydantic
pydantic-settings
```

### 4. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 5. Install the RAG Package (Editable Mode)
```bash
pip install -e rag_package_complete
```

This will install the `rag_package` and all its dependencies from `setup.py`.

### 6. Install Main Project Requirements (If Needed)
```bash
pip install -r requirements.txt
```

### 7. Create `.env` File
Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Quick Start Summary

```bash
# Complete sequence:
python -m venv venv
source venv/bin/activate
pip install -e rag_package_complete  # ← Specify the path!
pip install -r requirements.txt      # ← If you have main project deps
```

## Important Notes

1. ✅ `pip install -e rag_package_complete` installs the package in editable mode and automatically installs dependencies from `setup.py`.

2. ⚠️ The `rag_package_complete/requirements.txt` includes `watchdog>=3.0.0` which is not in `setup.py`. You may want to add it to `setup.py` or install it separately if needed.

3. 📝 If you only need the RAG package functionality, you can skip the main `requirements.txt` step.

## Testing the Installation

After installation, test it with:
```bash
python rag_package_complete/example.py
```

Or create your own script:
```python
from rag_package import MultiCollectionRAG
from dotenv import load_dotenv

load_dotenv()

rag = MultiCollectionRAG(
    chroma_persist_dir="./chroma_db",
    embedding_model="text-embedding-3-small"
)
```

## Troubleshooting

- **Module not found**: Make sure you activated the virtual environment and installed with `-e` flag
- **OpenAI API errors**: Check your `.env` file has the correct `OPENAI_API_KEY`
- **ChromaDB errors**: Ensure `chromadb>=0.4.0` is installed (should be automatic from setup.py)
