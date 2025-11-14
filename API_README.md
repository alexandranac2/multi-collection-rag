# RAG API Documentation

FastAPI-based REST API for managing documents, Q&A pairs, and querying the RAG system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with required variables:
```bash
OPENAI_API_KEY=your_openai_api_key

# Optional: LangFuse observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com  # optional, defaults to cloud
```

3. Run the API:
```bash
python -m app.main
# or
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Documents

- `POST /api/documents` - Upload and ingest a document
  - Requires: `file` (multipart), `folder_name` (form field)
  - Optional: `doc_type`, `title`, `description`
  
- `GET /api/documents` - List all documents
  - Query params: `collection` (optional)
  
- `GET /api/documents/{document_id}` - Get document details

- `PUT /api/documents/{document_id}` - Update document metadata

- `DELETE /api/documents/{document_id}` - Delete a document

### Q&A

- `POST /api/qa` - Create a Q&A pair
  - Body: `{"question": "...", "answer": "...", "tags": [], "category": "..."}`

- `GET /api/qa` - List all Q&A pairs
  - Query params: `limit`, `offset`

- `GET /api/qa/{qa_id}` - Get Q&A pair by ID

- `DELETE /api/qa/{qa_id}` - Delete a Q&A pair

### Query

- `POST /api/query` - Query the RAG system
  - Body: 
    ```json
    {
      "query_text": "your question",
      "collections": ["collection1"],  // optional
      "doc_types": ["manual"],  // optional
      "n_results": 5,  // optional, default 5
      "include_qa": true,  // optional, default true
      "return_all": false  // optional, default false (returns only best result)
    }
    ```

### Collections

- `GET /api/collections` - List all collections

- `POST /api/collections` - Create a new collection

- `GET /api/collections/{collection_name}` - Get collection details

## Features

- **Document Management**: Upload, list, update, and delete documents
- **Q&A Storage**: Store and search past question/answer pairs in vector DB
- **Unified Search**: Query both documents and Q&A pairs simultaneously
- **Citations**: All results include full citation metadata
- **LangFuse Integration**: Optional observability and tracing
- **Persistent Registry**: Document metadata is saved to `.document_registry.json`

## Response Format

Query responses include:
- `query`: Original query text
- `return_all`: Whether all results or just best result was returned
- `results`: Array of results, each with:
  - `text`: Excerpt from document or Q&A
  - `source_type`: "document" or "qa"
  - `citations`: Full citation metadata

## Swagger Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

