# test_installation.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch
import os

def test_dependencies():
    print("Testing dependencies...")
    
    # Test CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test embeddings
    print("\nTesting embeddings model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embeddings model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading embeddings: {str(e)}")
    
    # Test Chroma
    print("\nTesting Chroma database...")
    try:
        # Create a test document
        test_doc = Document(page_content="This is a test document.", metadata={"source": "test"})
        
        # Create a temporary vector store
        db = Chroma.from_documents(
            documents=[test_doc],
            embedding=embeddings,
            persist_directory="./test_db"
        )
        
        # Test search
        results = db.similarity_search("test", k=1)
        print("✓ Chroma database working correctly")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_db"):
            shutil.rmtree("./test_db")
            
    except Exception as e:
        print(f"✗ Error with Chroma: {str(e)}")

if __name__ == "__main__":
    test_dependencies()
    print("\nAll tests completed!")