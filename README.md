# RAG Package 🚀

A production-ready, multi-collection RAG (Retrieval-Augmented Generation) system that you can reuse across all your projects. Built with Docling, ChromaDB, and OpenAI embeddings.

Perfect for handling multiple document types: user manuals, company policies, legal contracts, and more — all in separate, queryable collections.

## ✨ Features

- 🗂️ **Multi-Collection Support** - Separate collections for different document types
- 📄 **Smart Document Processing** - Powered by Docling (PDF, DOCX, PPTX, HTML, MD, TXT)
- 🧩 **Flexible Chunking** - Hybrid and Hierarchical strategies
- 💾 **Intelligent Caching** - SHA256-based caching to skip unchanged files
- 🔍 **Powerful Querying** - Query specific collections, document types, or all at once
- 📦 **Plug & Play** - Easy to integrate into any project

---

## 📦 Installation

### 1. Clone or Copy the Package

```bash
# Copy the rag_package folder to your project
cp -r rag_package /path/to/your/project/
```

### 2. Install Dependencies

```bash
pip install openai chromadb docling python-dotenv
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Install as Editable Package (Recommended)

```bash
# From the directory containing setup.py
pip install -e .
```

This allows you to import the package anywhere: `from rag_package import MultiCollectionRAG`

### 4. Set Up Environment Variables

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

---

## 🚀 Quick Start

```python
from rag_package import MultiCollectionRAG

# 1. Initialize the RAG system
rag = MultiCollectionRAG(
    chroma_persist_dir="./chroma_db",
    embedding_model="text-embedding-3-small"
)

# 2. Add your document collections
rag.add_collection(
    collection_name="user_manuals",
    docs_path="./documents/manuals",
    doc_type="manual",
    chunk_size=600,
    chunking_strategy="hierarchical"
)

rag.add_collection(
    collection_name="policies",
    docs_path="./documents/policies",
    doc_type="policy",
    chunk_size=800,
    chunking_strategy="hierarchical"
)

# 3. Ingest documents (one-time setup)
rag.ingest_all()

# 4. Query your documents
results = rag.query_combined(
    "How do I reset my password?",
    n_results=5
)

for result in results:
    print(f"📄 {result['metadata']['filename']}")
    print(f"   {result['text'][:200]}...")
    print(f"   Score: {result['distance']:.4f}\n")
```

---

## 📖 Detailed Usage

### Adding Collections

Each collection represents a distinct set of documents with its own configuration:

```python
rag.add_collection(
    collection_name="legal_contracts",      # Unique identifier
    docs_path="./docs/contracts",           # Folder with documents
    doc_type="contract",                    # Type tag for filtering
    chunk_size=800,                         # Tokens per chunk
    chunk_overlap=50,                       # Overlap between chunks
    chunking_strategy="hierarchical"        # "hybrid" or "hierarchical"
)
```

**Chunking Strategies:**

- **`hierarchical`** - Respects document structure (sections, lists). Best for legal docs, manuals, policies.
- **`hybrid`** - Token-aware + structure-aware. Good for general documents.

**Recommended Settings:**

| Document Type    | Chunk Size | Overlap | Strategy       |
|-----------------|------------|---------|----------------|
| Legal Contracts | 800        | 50      | hierarchical   |
| User Manuals    | 600        | 100     | hierarchical   |
| Company Policies| 800        | 100     | hierarchical   |
| General Docs    | 512        | 128     | hybrid         |

---

### Ingesting Documents

**Ingest all collections:**
```python
rag.ingest_all()
```

**Ingest a specific collection:**
```python
rag.ingest_collection("user_manuals")
```

**Force re-indexing (ignores cache):**
```python
rag.ingest_all(force_reindex=True)
```

**Caching:**
- SHA256 hashes track file changes
- Only modified files are re-processed
- Cache files: `.{collection_name}_cache.json`

---

### Querying Documents

#### 1. Query All Collections (Combined Results)

```python
results = rag.query_combined(
    query_text="What is the vacation policy?",
    n_results=5  # Top 5 results across all collections
)
```

#### 2. Query Specific Collections

```python
results = rag.query(
    query_text="How to install the software?",
    collections=["user_manuals", "installation_guides"],
    n_results=3
)

# Returns: {"user_manuals": [...], "installation_guides": [...]}
```

#### 3. Query by Document Type

```python
results = rag.query(
    query_text="What are the security requirements?",
    doc_types=["policy", "contract"],  # Filter by type
    n_results=5
)
```

#### 4. Advanced Filtering

```python
results = rag.query(
    query_text="password reset procedure",
    collections=["user_manuals"],
    n_results=10
)

for coll_name, coll_results in results.items():
    print(f"\n📁 Collection: {coll_name}")
    for r in coll_results:
        print(f"  • {r['metadata']['filename']}: {r['distance']:.4f}")
```

---

### Working with Results

Each result contains:

```python
{
    'text': 'The chunk text content...',
    'metadata': {
        'source': '/path/to/file.pdf',
        'filename': 'file.pdf',
        'doc_type': 'manual',
        'collection': 'user_manuals',
        'page': 5
    },
    'distance': 0.234,  # Lower = more similar
    'id': 'user_manuals_file_0',
    'collection': 'user_manuals'
}
```

**Example: Display results nicely**

```python
def display_results(results):
    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Result {i} | Score: {r['distance']:.4f}")
        print(f"Source: {r['metadata']['filename']}")
        print(f"Type: {r['metadata']['doc_type']}")
        if r['metadata']['page']:
            print(f"Page: {r['metadata']['page']}")
        print(f"\nContent:\n{r['text'][:300]}...")

results = rag.query_combined("expense reimbursement", n_results=3)
display_results(results)
```

---

### Managing Collections

**List all collections:**
```python
rag.list_collections()

# Output:
# 📚 Collections:
#   • user_manuals: 245 chunks (manual)
#   • policies: 89 chunks (policy)
#   • contracts: 156 chunks (contract)
```

**Check collection info:**
```python
# Get count for a specific collection
count = rag.collections["user_manuals"].count()
print(f"User manuals: {count} chunks")
```

---

## 🏗️ Project Structure

```
your_project/
├── rag_package/
│   ├── __init__.py
│   ├── rag.py              # Main RAG class
│   ├── collection.py       # Collection management
│   ├── processor.py        # Document processing
│   ├── cache.py            # Caching logic
│   ├── chunkers.py         # Chunking strategies
│   └── utils.py            # Helper functions
├── documents/
│   ├── manuals/
│   │   ├── user_guide.pdf
│   │   └── admin_manual.docx
│   ├── policies/
│   │   └── hr_policy.pdf
│   └── contracts/
│       └── vendor_agreement.pdf
├── chroma_db/              # Vector database (auto-created)
├── .user_manuals_cache.json  # Cache files (auto-created)
├── .policies_cache.json
├── .env                    # Environment variables
├── setup.py
├── requirements.txt
└── main.py                 # Your application
```

---

## 🔧 Configuration Options

### RAG Initialization

```python
rag = MultiCollectionRAG(
    chroma_persist_dir="./chroma_db",           # Where to store vector DB
    embedding_model="text-embedding-3-small",   # OpenAI embedding model
    openai_api_key="sk-..."                     # Optional, uses .env if not provided
)
```

### Collection Configuration

```python
rag.add_collection(
    collection_name="my_docs",          # Unique name (required)
    docs_path="./documents",            # Path to documents (required)
    doc_type="general",                 # "manual", "policy", "contract", "general"
    chunk_size=512,                     # Tokens per chunk (default: 512)
    chunk_overlap=128,                  # Overlap tokens (default: 128)
    chunking_strategy="hybrid"          # "hybrid" or "hierarchical"
)
```

---

## 💡 Use Cases

### Use Case 1: Customer Support Bot

```python
# Setup
rag = MultiCollectionRAG()
rag.add_collection("help_docs", "./help", doc_type="manual")
rag.add_collection("faqs", "./faqs", doc_type="general")
rag.ingest_all()

# Query
def answer_question(question):
    results = rag.query_combined(question, n_results=3)
    
    # Pass results to LLM for final answer
    context = "\n\n".join([r['text'] for r in results])
    return generate_answer(question, context)
```

### Use Case 2: Legal Document Search

```python
# Setup with large chunks for legal precision
rag = MultiCollectionRAG()
rag.add_collection(
    "contracts",
    "./legal/contracts",
    doc_type="contract",
    chunk_size=1000,
    chunk_overlap=50,
    chunking_strategy="hierarchical"
)
rag.ingest_all()

# Query specific contract clauses
results = rag.query(
    "limitation of liability clauses",
    collections=["contracts"],
    n_results=10
)
```

### Use Case 3: Multi-Department Knowledge Base

```python
# Setup
rag = MultiCollectionRAG()
rag.add_collection("hr_policies", "./hr", doc_type="policy")
rag.add_collection("it_manuals", "./it", doc_type="manual")
rag.add_collection("finance_docs", "./finance", doc_type="policy")
rag.ingest_all()

# Route query to correct department
def smart_search(query, department=None):
    if department:
        collection_map = {
            "hr": ["hr_policies"],
            "it": ["it_manuals"],
            "finance": ["finance_docs"]
        }
        collections = collection_map.get(department)
    else:
        collections = None  # Search all
    
    return rag.query_combined(query, collections=collections)
```

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"

**Solution:** Create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Issue: "Collection not found"

**Solution:** Make sure you call `add_collection()` before querying:
```python
rag.add_collection("my_docs", "./documents")
rag.ingest_collection("my_docs")
```

### Issue: No results returned

**Check:**
1. Documents were ingested: `rag.list_collections()`
2. Supported formats: PDF, DOCX, PPTX, HTML, MD, TXT
3. Try broader query or increase `n_results`

### Issue: Out of memory during ingestion

**Solutions:**
1. Process collections one at a time:
   ```python
   rag.ingest_collection("collection1")
   rag.ingest_collection("collection2")
   ```
2. Reduce chunk size to generate fewer embeddings per document

### Issue: Slow queries

**Solutions:**
1. Query specific collections instead of all
2. Reduce `n_results`
3. Consider using FAISS instead of ChromaDB for larger datasets

---

## 📚 API Reference

### `MultiCollectionRAG`

**Methods:**
- `add_collection(...)` - Add a document collection
- `ingest_collection(name, force_reindex=False)` - Ingest one collection
- `ingest_all(force_reindex=False)` - Ingest all collections
- `query(query_text, collections=None, doc_types=None, n_results=5)` - Query collections
- `query_combined(query_text, collections=None, doc_types=None, n_results=5)` - Query and combine results
- `list_collections()` - Display all collections

---

## 🔄 Updating Documents

The system automatically detects file changes via SHA256 hashing:

```python
# Add new documents to your folders
# documents/manuals/new_guide.pdf

# Run ingestion again - only new/changed files are processed
rag.ingest_all()
```

**To force complete re-indexing:**
```python
rag.ingest_all(force_reindex=True)
```

---

## 📝 Example: Complete Application

```python
from rag_package import MultiCollectionRAG
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize
    print("🚀 Initializing RAG system...")
    rag = MultiCollectionRAG()
    
    # Add collections
    rag.add_collection(
        "user_manuals",
        "./docs/manuals",
        doc_type="manual",
        chunk_size=600,
        chunking_strategy="hierarchical"
    )
    
    rag.add_collection(
        "company_policies",
        "./docs/policies",
        doc_type="policy",
        chunk_size=800,
        chunking_strategy="hierarchical"
    )
    
    # Ingest
    print("\n📥 Ingesting documents...")
    rag.ingest_all()
    
    # Display collections
    rag.list_collections()
    
    # Interactive query loop
    print("\n💬 Ask questions (type 'quit' to exit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        results = rag.query_combined(query, n_results=3)
        
        print(f"\n📊 Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['metadata']['filename']} (Score: {r['distance']:.4f})")
            print(f"   {r['text'][:200]}...")

if __name__ == "__main__":
    main()
```

---

## 🤝 Contributing

This is a reusable package for your projects. Feel free to modify and extend it!

---

## 📄 License

MIT License - Use freely in your projects!

---

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the examples
3. Check ChromaDB/Docling documentation

---

**Built with ❤️ using Docling, ChromaDB, and OpenAI**
# p1-ai-compliance-security
