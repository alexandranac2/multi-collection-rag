"""
Example usage of the RAG package.
"""

from rag_package import MultiCollectionRAG
from dotenv import load_dotenv

load_dotenv()


def main():
    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag = MultiCollectionRAG(
        chroma_persist_dir="./chroma_db",
        embedding_model="text-embedding-3-small"
    )
    
    # Add collections
    print("\n📁 Adding collections...")
    
    rag.add_collection(
        collection_name="user_manuals",
        docs_path="./documents/manuals",
        doc_type="manual",
        chunk_size=600,
        chunk_overlap=100,
        chunking_strategy="hierarchical"
    )
    
    rag.add_collection(
        collection_name="company_policies",
        docs_path="./documents/policies",
        doc_type="policy",
        chunk_size=800,
        chunk_overlap=100,
        chunking_strategy="hierarchical"
    )
    
    rag.add_collection(
        collection_name="general_docs",
        docs_path="./documents/general",
        doc_type="general",
        chunk_size=512,
        chunk_overlap=128,
        chunking_strategy="hybrid"
    )
    
    # Ingest documents
    print("\n📥 Ingesting documents...")
    rag.ingest_all()
    
    # Show collections
    rag.list_collections()
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    # Query 1: Search all collections
    print("\n1️⃣ Query all collections:")
    results = rag.query_combined(
        "How do I reset my password?",
        n_results=3
    )
    
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  📄 {r['metadata']['filename']}")
        print(f"  📊 Score: {r['distance']:.4f}")
        print(f"  📝 {r['text'][:150]}...")
    
    # Query 2: Search specific collection
    print("\n2️⃣ Query specific collection (user_manuals):")
    results = rag.query(
        "installation instructions",
        collections=["user_manuals"],
        n_results=2
    )
    
    for coll_name, coll_results in results.items():
        print(f"\n  Collection: {coll_name}")
        for r in coll_results:
            print(f"    • {r['metadata']['filename']}: {r['distance']:.4f}")
    
    # Query 3: Filter by document type
    print("\n3️⃣ Query by document type (policy):")
    results = rag.query_combined(
        "vacation policy",
        doc_types=["policy"],
        n_results=2
    )
    
    for r in results:
        print(f"  • {r['metadata']['filename']} ({r['metadata']['doc_type']})")
    
    # Interactive mode (optional)
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Type your questions (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\n❓ Your question: ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query.strip():
                continue
            
            results = rag.query_combined(query, n_results=3)
            
            if not results:
                print("   No results found.")
                continue
            
            print(f"\n📊 Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n  {i}. {r['metadata']['filename']}")
                print(f"     Score: {r['distance']:.4f}")
                print(f"     {r['text'][:200]}...")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
