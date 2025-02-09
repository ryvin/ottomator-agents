# document_loader.py
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

class DocumentProcessor:
    """Handles document loading and processing with enhanced metadata."""
    
    SUPPORTED_EXTENSIONS = {
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.md': UnstructuredMarkdownLoader,
    }

    CHUNK_SIZES = {
        '.csv': 500,
        '.xlsx': 500,
        '.md': 1000,
        '.py': 1000,
        '.pdf': 750,
        '.docx': 750,
        '.txt': 750,
    }

    def __init__(self):
        """Initialize the document processor."""
        # Initialize pandas display options for better CSV handling
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def load_documents(self, data_dir: str) -> List[Document]:
        """Load documents from the specified directory."""
        data_path = Path(data_dir)
        documents = []

        for file_path in data_path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    # Special handling for Excel files
                    if file_path.suffix.lower() == '.xlsx':
                        # Use pandas for Excel files
                        df = pd.read_excel(file_path)
                        # Convert DataFrame to string format
                        content = df.to_string(index=False)
                        documents.append(Document(
                            page_content=content,
                            metadata={'source': str(file_path)}
                        ))
                    else:
                        loader_class = self.SUPPORTED_EXTENSIONS[file_path.suffix.lower()]
                        loader = loader_class(str(file_path))
                        documents.extend(loader.load())
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with enhanced metadata and chunking."""
        processed_chunks = []

        for doc in documents:
            source_path = Path(doc.metadata['source'])
            file_extension = source_path.suffix.lower()

            # Create text splitter with adaptive chunk size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZES.get(file_extension, 750),
                chunk_overlap=50,
                length_function=len,
                add_start_index=True
            )

            # Split document and enhance metadata
            chunks = text_splitter.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    'source': str(source_path),
                    'doc_type': file_extension,
                    'file_name': source_path.name,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'start_index': sum(len(c) for c in chunks[:i])
                }
                processed_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))

        return processed_chunks