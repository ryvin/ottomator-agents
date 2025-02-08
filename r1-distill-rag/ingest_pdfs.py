from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import os
import shutil
import re
import json

load_dotenv()

def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata to ensure compatibility with Chroma."""
    cleaned = {}
    for key, value in metadata.items():
        # Convert lists to comma-separated strings
        if isinstance(value, list):
            cleaned[key] = ", ".join(str(v) for v in value)
        # Convert numbers to strings
        elif isinstance(value, (int, float)):
            cleaned[key] = str(value)
        # Keep strings and booleans as is
        elif isinstance(value, (str, bool)):
            cleaned[key] = value
        # Convert other types to string representation
        else:
            cleaned[key] = str(value)
    return cleaned

class SimpleExcelLoader:
    """Simple Excel loader that reads all sheets."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> List[Document]:
        """Implement lazy_load for DirectoryLoader compatibility."""
        return self.load()

    def load(self) -> List[Document]:
        """Load Excel file and return documents."""
        try:
            xlsx = pd.ExcelFile(self.file_path)
            documents = []

            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                
                # Convert DataFrame to string representation
                content = f"Sheet: {sheet_name}\n\n"
                content += df.to_string(index=False)
                
                # Create and clean metadata
                metadata = clean_metadata({
                    "source": self.file_path,
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns)
                })
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

            return documents
        except Exception as e:
            print(f"Error loading Excel file {self.file_path}: {str(e)}")
            return []

class EnhancedCSVLoader(CSVLoader):
    """Enhanced CSV loader with better error handling."""
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding='utf-8', csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': None
        })
        
    def load(self) -> List[Document]:
        """Override load to add better error handling."""
        try:
            documents = super().load()
            # Clean metadata for each document
            for doc in documents:
                doc.metadata = clean_metadata(doc.metadata)
            return documents
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    self.encoding = encoding
                    documents = super().load()
                    for doc in documents:
                        doc.metadata = clean_metadata(doc.metadata)
                    return documents
                except UnicodeDecodeError:
                    continue
            raise Exception(f"Could not read CSV with any supported encoding: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV file {self.file_path}: {str(e)}")

class EnhancedDocx2txtLoader(Docx2txtLoader):
    """Enhanced DOCX loader that better handles forms."""
    
    def load(self) -> List[Document]:
        """Load and process the document."""
        import docx2txt
        
        text = docx2txt.process(self.file_path)
        cleaned_text = self._clean_text(text)
        processed_text = self._process_form_sections(cleaned_text)
        
        metadata = clean_metadata({"source": self.file_path})
        return [Document(page_content=processed_text, metadata=metadata)]

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[_\-*]{2,}', '', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _process_form_sections(self, text: str) -> str:
        """Process form sections to maintain structure."""
        sections = re.split(r'(?:\n\s*){2,}(?=[A-Z][^a-z\n]*:)', text)
        
        processed_sections = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            fields = re.findall(r'([A-Z][^:\n]+):\s*([^\n]+)?', section)
            if fields:
                formatted_section = '\n'.join(
                    f"{field.strip()}: {value.strip() if value else 'Not Provided'}"
                    for field, value in fields
                )
                processed_sections.append(formatted_section)
            else:
                processed_sections.append(section)
        
        return '\n\n'.join(processed_sections)

# Define loader mapping
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".md": (TextLoader, {"encoding": "utf8"}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".docx": (EnhancedDocx2txtLoader, {}),
    ".csv": (EnhancedCSVLoader, {}),
    ".xlsx": (SimpleExcelLoader, {}),
    ".xls": (SimpleExcelLoader, {})
}

def load_documents(data_dir: str) -> List[Document]:
    """Load documents from directory with multiple file types."""
    all_documents = []
    
    for ext, (loader_cls, loader_kwargs) in LOADER_MAPPING.items():
        loader = DirectoryLoader(
            data_dir,
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

def process_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with improved handling of forms."""
    if not documents:
        print("No documents to process")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    # Clean metadata for all chunks
    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)
    return chunks

def create_vector_store(chunks, persist_directory: str):
    """Create and persist Chroma vector store."""
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    print("Loading documents...")
    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} total documents")
    
    print("Processing documents...")
    chunks = process_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()