# main.py
from pathlib import Path
from dotenv import load_dotenv
import os

from models import get_llm
from document_loader import DocumentProcessor
from vectorstore import VectorStoreManager
from rag_processor import EnhancedRAGProcessor

def setup_directories():
    """Setup necessary directories."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    db_dir = base_dir / "chroma_db"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    db_dir.mkdir(exist_ok=True)
    
    return str(data_dir), str(db_dir)

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup directories
    data_dir, db_dir = setup_directories()
    print(f"Data directory: {data_dir}")
    print(f"Database directory: {db_dir}")
    
    try:
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Load and process documents
        print("\nLoading documents...")
        documents = doc_processor.load_documents(data_dir)
        if not documents:
            print("No documents found. Please add documents to the data directory.")
            return
        print(f"Loaded {len(documents)} documents")
        
        # Process documents into chunks
        print("\nProcessing documents...")
        chunks = doc_processor.process_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Initialize and setup vector store
        print("\nInitializing vector store...")
        vector_manager = VectorStoreManager(db_dir)
        vectorstore = vector_manager.create_or_load_vectorstore(chunks)
        if not vectorstore:
            print("Failed to initialize vector store")
            return
        
        # Get vector store metadata
        metadata = vector_manager.get_metadata()
        print("\nVector Store Information:")
        print(f"Total Documents: {metadata.total_documents}")
        print(f"Document Types: {', '.join(metadata.document_types)}")
        print(f"Total Chunks: {metadata.total_chunks}")
        
        # Initialize LLM and RAG processor
        print("\nInitializing RAG processor...")
        model = get_llm(os.getenv("REASONING_MODEL_ID"))
        processor = EnhancedRAGProcessor(vectorstore, model)
        
        # Create processing chain
        chain = processor.create_chain()
        
        # Run test queries
        test_queries = [
            "What are Elevate Digital Solutions' main strengths and weaknesses?",
            "What is the financial status of Leadingbit Solutions LLC?",
            "Show me the trend in ETH balance growth",
            "Give me a summary of all available document types and their key information"
        ]
        
        print("\nRunning test queries:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = chain.invoke(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error processing query: {str(e)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()