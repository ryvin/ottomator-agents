# vectorstore.py
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dataclasses import dataclass

@dataclass
class VectorStoreMetadata:
    total_documents: int
    document_types: List[str]
    total_chunks: int

class VectorStoreManager:
    """Manages vector store operations with enhanced metadata handling."""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore: Optional[Chroma] = None

    def create_or_load_vectorstore(self, documents: Optional[List[Document]] = None) -> Optional[Chroma]:
        """Create new or load existing vector store."""
        persist_path = Path(self.persist_dir)
        
        try:
            if persist_path.exists() and any(persist_path.iterdir()):
                print("Loading existing vector store...")
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_path)
                )
                print(f"Loaded {len(self.vectorstore.get()['ids'])} documents")
            elif documents:
                print(f"Creating new vector store with {len(documents)} documents...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(persist_path)
                )
                print("Vector store created successfully")
            
            return self.vectorstore
        
        except Exception as e:
            print(f"Error with vector store: {e}")
            return None

    def get_metadata(self) -> VectorStoreMetadata:
        """Get vector store metadata."""
        if not self.vectorstore:
            return VectorStoreMetadata(0, [], 0)

        collection = self.vectorstore.get()
        metadatas = collection.get('metadatas', [])
        
        unique_sources = set()
        document_types = set()

        for metadata in metadatas:
            if metadata:
                source = metadata.get('source')
                doc_type = metadata.get('doc_type')
                if source:
                    unique_sources.add(source)
                if doc_type:
                    document_types.add(doc_type)

        return VectorStoreMetadata(
            total_documents=len(unique_sources),
            document_types=sorted(list(document_types)),
            total_chunks=len(metadatas)
        )