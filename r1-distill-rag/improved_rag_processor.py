from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Import our improved modules
from improved_rag import process_documents, create_vector_store
from improved_rag_processor import EnhancedRAGProcessor

def get_model(model_id):
    """Get the appropriate model based on environment settings."""
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(
            model_id=model_id,
            token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize directories
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # Load and process documents
    print("Loading documents...")
    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} total documents")
    
    # Process documents with improved chunking
    print("Processing documents...")
    chunks = process_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    
    # Initialize the RAG processor
    model = get_model(os.getenv("REASONING_MODEL_ID"))
    processor = EnhancedRAGProcessor(vectordb, model)
    
    # Create the processing chain
    chain = create_rag_chain(processor)
    
    # Print available document types
    doc_info = processor.get_available_document_types()
    print("\nAvailable Document Types:")
    print(f"Total Documents: {doc_info['total_documents']}")
    print(f"Document Types: {', '.join(doc_info['document_types'])}")
    print(f"Total Chunks: {doc_info['total_chunks']}")
    
    # Example queries to test
    test_queries = [
        "What are Elevate Digital Solutions' main strengths and weaknesses?",
        "What is the financial status of Leadingbit Solutions LLC?",
        "Show me the trend in ETH balance growth",
        "Give me a summary of all available document types and their key information"
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chain.invoke(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()