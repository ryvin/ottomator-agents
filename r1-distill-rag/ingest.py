# ingest.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from typing import List, Dict, Any
import pandas as pd
import os
import shutil

class DocumentIngester:
    def __init__(self, data_dir: str, db_dir: str):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Define loader mapping
        self.LOADER_MAPPING = {
            ".pdf": (PyPDFLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            ".md": (TextLoader, {"encoding": "utf8"}),
            ".docx": (Docx2txtLoader, {}),
            ".csv": (CSVLoader, {"encoding": "utf-8"})
        }
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory."""
        all_documents = []
        
        for ext, (loader_cls, loader_kwargs) in self.LOADER_MAPPING.items():
            loader = DirectoryLoader(
                self.data_dir,
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True
            )
            
            try:
                documents = loader.load()
                if documents:
                    print(f"Loaded {len(documents)} {ext} files")
                    all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {ext} files: {str(e)}")
        
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            print("No documents to process")
            return []
        
        # Use different chunk sizes based on document type
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document]):
        """Create and persist Chroma vector store."""
        if os.path.exists(self.db_dir):
            print(f"Clearing existing vector store at {self.db_dir}")
            shutil.rmtree(self.db_dir)
        
        print("Creating new vector store...")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        vectordb.persist()
        return vectordb
    
    def ingest(self):
        """Run the full ingestion process."""
        print("Starting document ingestion...")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load documents
        print("\nLoading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} total documents")
        
        # Process documents
        print("\nProcessing documents...")
        chunks = self.process_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        print("\nCreating vector store...")
        vectordb = self.create_vector_store(chunks)
        print(f"Vector store created and persisted at {self.db_dir}")
        
        return vectordb

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    db_dir = os.path.join(base_dir, "chroma_db")
    
    # Create and run ingester
    ingester = DocumentIngester(data_dir, db_dir)
    ingester.ingest()

if __name__ == "__main__":
    main()