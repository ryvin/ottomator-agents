# vectorstore.py
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dataclasses import dataclass
import chromadb
import shutil
import os
import time

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
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self._client = None
        self.vectorstore: Optional[Chroma] = None
        
    def _get_client(self):
        """Get or create a chromadb client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client
        
    def _clear_vector_store(self):
        """Clear existing vector store data."""
        try:
            # First, close any existing connections
            if self.vectorstore is not None:
                if hasattr(self.vectorstore, '_client') and self.vectorstore._client is not None:
                    if hasattr(self.vectorstore._client, 'close'):
                        self.vectorstore._client.close()
                self.vectorstore = None
            
            if self._client is not None:
                if hasattr(self._client, 'close'):
                    self._client.close()
                self._client = None
            
            persist_path = Path(self.persist_dir)
            if persist_path.exists():
                # On Windows, we need to handle file locks
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Add a small delay before attempting to remove
                        time.sleep(1)
                        shutil.rmtree(str(persist_path))
                        break
                    except PermissionError:
                        if retry < max_retries - 1:
                            print(f"Retrying clear vector store (attempt {retry + 1})")
                            continue
                        raise
                
                persist_path.mkdir(parents=True, exist_ok=True)
                print("Cleared existing vector store")
        except Exception as e:
            print(f"Error clearing vector store: {e}")

    def create_or_load_vectorstore(self, documents: Optional[List[Document]] = None) -> Optional[Chroma]:
        """Create new or load existing vector store."""
        try:
            persist_path = Path(self.persist_dir)
            
            # Clear existing store if documents are provided
            if documents:
                print(f"Creating new vector store with {len(documents)} documents...")
                try:
                    # Close existing connections first
                    if self.vectorstore is not None:
                        try:
                            if hasattr(self.vectorstore, '_collection'):
                                self.vectorstore._collection = None
                            if hasattr(self.vectorstore, '_client'):
                                self.vectorstore._client = None
                            self.vectorstore = None
                        except:
                            pass
                    
                    # Instead of clearing, create a new timestamped directory
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    new_persist_dir = Path(self.persist_dir).parent / f"chroma_db_{timestamp}"
                    new_persist_dir.mkdir(parents=True, exist_ok=True)
                    self.persist_dir = str(new_persist_dir)
                    
                    # Create new store with Windows-compatible paths
                    client = chromadb.PersistentClient(path=self.persist_dir)
                    self.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=self.persist_dir,
                        collection_name="document_store",
                        client=client
                    )
                    print("Vector store created successfully")
                    
                    # Clean up old directories in the background
                    def cleanup_old_dirs():
                        try:
                            for old_dir in Path(self.persist_dir).parent.glob("chroma_db_*"):
                                if old_dir != new_persist_dir:
                                    try:
                                        shutil.rmtree(str(old_dir))
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Start cleanup in a separate thread
                    from threading import Thread
                    Thread(target=cleanup_old_dirs, daemon=True).start()
                    
                except Exception as e:
                    print(f"Error creating vector store: {e}")
                    return None
            else:
                # Try to load existing store
                print("Loading existing vector store...")
                try:
                    client = chromadb.PersistentClient(path=self.persist_dir)
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_dir,
                        embedding_function=self.embeddings,
                        collection_name="document_store",
                        client=client
                    )
                    collection_data = self.vectorstore.get()
                    count = len(collection_data.get('ids', []))
                    print(f"Loaded {count} documents")
                    
                    if count == 0:
                        print("Warning: Vector store is empty")
                        return None
                except Exception as load_error:
                    print(f"Error loading existing store: {load_error}")
                    return None
            
            return self.vectorstore
            
        except Exception as e:
            print(f"Error with vector store: {e}")
            return None

    def get_metadata(self) -> VectorStoreMetadata:
        """Get vector store metadata."""
        if not self.vectorstore:
            return VectorStoreMetadata(0, [], 0)

        try:
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
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return VectorStoreMetadata(0, [], 0)