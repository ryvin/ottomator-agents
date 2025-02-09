from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
import glob

def load_documents(data_dir: str) -> List[Document]:
    """Load documents from the specified directory, handling various file types."""
    all_files = []
    for file_type in ["*.csv", "*.docx", "*.pdf", "*.txt", "*.xlsx", "*.md"]:
        all_files.extend(glob.glob(os.path.join(data_dir, file_type)))

    if not all_files:
        print(f"No files found in {data_dir}")
        return []

    documents = []
    for file_path in all_files:
        try:
            loader = get_loader(file_path)
            if loader:
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return documents

def get_loader(file_path: str):
    """Get the appropriate loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        return CSVLoader(file_path)
    elif file_extension == ".docx":
        return Docx2txtLoader(file_path)
    elif file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path)
    elif file_extension == ".xlsx":
        return UnstructuredExcelLoader(file_path)
    elif file_extension == ".md":
        return UnstructuredMarkdownLoader(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return None

def process_documents(documents: List[Document]) -> List[Document]:
    """Process and chunk documents with improved metadata and chunking."""
    processed_chunks = []
    for doc in documents:
        # Extract file extension and name
        file_extension = os.path.splitext(doc.metadata['source'])[1]
        file_name = os.path.basename(doc.metadata['source'])

        # Create a text splitter with adaptive chunk sizes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_chunk_size(file_extension),
            chunk_overlap=50,  # Consistent overlap
            length_function=len,
            add_start_index=True
        )

        # Split the document into chunks
        chunks = text_splitter.split_text(doc.page_content)

        # Create new Document objects for each chunk with enhanced metadata
        for i, chunk_text in enumerate(chunks):
            metadata = {
                'source': doc.metadata['source'],
                'doc_type': file_extension,
                'file_name': file_name,
                'chunk_index': i,  # Index of the chunk within the document
                'start_index': doc.metadata.get('start_index', 0) + sum(len(c) for c in chunks[:i]) # Start index from original doc
            }
            processed_chunks.append(Document(page_content=chunk_text, metadata=metadata))
    return processed_chunks

def get_chunk_size(file_extension: str) -> int:
    """Determine chunk size based on file type for better context."""
    if file_extension in ['.csv', '.xlsx']:
        return 500
    elif file_extension in ['.md', '.py']:
        return 1000
    else:  # Default for .docx, .pdf, .txt
        return 750

def create_vector_store(chunks: List[Document], db_dir: str) -> Chroma:
    """Create or load a Chroma vector store, with added debugging prints."""
    embeddings = HuggingFaceEmbeddings()
    print(f"DB Directory: {db_dir}")  # Debug: Print the DB directory

    if os.path.exists(db_dir) and os.listdir(db_dir):
        print("Attempting to load existing vector store...")
        try:
            vectordb = Chroma(embedding_function=embeddings, persist_directory=db_dir)
            print("Loaded existing vector store.")
            # Debug: Check if the loaded database has any data
            print(f"Number of documents after loading: {len(vectordb.get().get('ids', []))}")
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Recreating...")
            vectordb = None
    else:
        print("No existing vector store found. Creating a new one.")
        vectordb = None

    if not vectordb:
        if not chunks:
            print("No chunks to index.")
            return None

        print(f"Creating vector store with {len(chunks)} chunks...")
        try:
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_dir
            )
            print("Created new vector store.")
            # Debug: Check if the created database has any data
            print(f"Number of documents after creation: {len(vectordb.get().get('ids', []))}")

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    collection_metadata = collect_vectorstore_metadata(vectordb)
    print(f"Collected Metadata: {collection_metadata}")  # Debug: Print collected metadata
    return vectordb



def collect_vectorstore_metadata(vectordb: Chroma) -> Dict[str, Any]:
    """Collect metadata with added debugging prints."""
    print("Collecting vector store metadata...")
    all_docs = vectordb.get()
    print(f"Raw data from vectordb.get(): {all_docs}")  # Debug: Print raw data

    total_documents = 0
    document_types = set()
    total_chunks = 0

    if all_docs:
        metadatas = all_docs.get('metadatas', [])
        total_chunks = len(all_docs.get('ids', []))

        unique_sources = set()

        for doc_metadata in metadatas:
            metadata = doc_metadata if isinstance(doc_metadata, dict) else {}
            source = metadata.get('source')
            doc_type = metadata.get('doc_type')

            if source:
                unique_sources.add(source)
            if doc_type:
                document_types.add(doc_type)

        total_documents = len(unique_sources)
        document_types = list(document_types)

    metadata = {
        'total_documents': total_documents,
        'document_types': document_types,
        'total_chunks': total_chunks
    }
    return metadata



class EnhancedRAGProcessor:
    def __init__(self, vectorstore: Chroma, llm_model):
        self.vectorstore = vectorstore
        self.llm = llm_model
        self.output_parser = StrOutputParser()

    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify the query to determine optimal search parameters."""
        # Define patterns for different query types
        patterns = {
            'overview': r'overview|summary|tell me about|what (information|data)|describe',
            'financial': r'financial|revenue|profit|loss|balance|cost|expense|money',
            'comparison': r'compare|versus|vs|difference|better|worse|than',
            'temporal': r'trend|over time|history|when|date|period|timeline',
            'technical': r'code|implementation|setup|configure|install|run'
        }

        # Check query against patterns
        intent_matches = {
            intent: bool(re.search(pattern, query.lower()))
            for intent, pattern in patterns.items()
        }

        # Determine search parameters based on intent
        search_params = {
            'k': 5,  # default number of results
            'min_doc_types': 2,  # minimum number of different document types
            'filter_metadata': {}
        }

        # Adjust parameters based on intent
        if intent_matches['overview']:
            search_params['k'] = 8
            search_params['min_doc_types'] = 3
        elif intent_matches['financial']:
            search_params['filter_metadata'] = {'doc_type': ['.csv', '.xlsx', '.docx']}
        elif intent_matches['technical']:
            search_params['filter_metadata'] = {'doc_type': ['.md', '.py']}

        return {
            'intents': intent_matches,
            'search_params': search_params
        }

    def _format_document_content(self, doc: Document) -> str:
        """Format document content based on its type."""
        doc_type = doc.metadata.get('doc_type', '')
        content = doc.page_content

        if doc_type in ['.csv', '.xlsx']:
            # Format tabular data more readably
            lines = content.split('\n')
            if len(lines) > 1:
                try:
                    # Attempt to align columns
                    rows = [line.split(',') for line in lines if line.strip()]
                    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
                    formatted_rows = [
                        ' | '.join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
                        for row in rows
                    ]
                    content = '\n'.join(formatted_rows)
                except:
                    pass  # Fall back to original content if formatting fails

        elif doc_type == '.md':
            # Preserve markdown formatting but clean up extra whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content).strip()

        return f"{content}\n[Source: {doc.metadata.get('source', 'Unknown')}]"

    def _create_context_prompt(self, documents: List[Document], query: str) -> str:
        """Create a well-structured context prompt from documents."""
        context_parts = []

        # Group documents by type
        docs_by_type = {}
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append(doc)

        # Format each document group
        for doc_type, docs in docs_by_type.items():
            context_parts.append(f"\n=== Content from {doc_type} files ===\n")
            for doc in docs:
                context_parts.append(self._format_document_content(doc))

        context = "\n".join(context_parts)

        # Create the full prompt
        prompt = f"""Based on the following information, answer the question: "{query}"

Context:
{context}

Instructions:
1. Use only information provided in the context
2. If the context doesn't contain enough information, say so
3. Cite the document sources when providing information
4. Format the response clearly with markdown headings and lists where appropriate

Question: {query}"""

        return prompt

    def process_query(self, query: str) -> str:
        """Process a query with improved context handling and response generation."""
        # Classify query and get search parameters
        query_analysis = self._classify_query_intent(query)
        search_params = query_analysis['search_params']

        # Get relevant documents
        print(f"Query: {query}")  # Debug: Print the query
        print(f"Search Params: {search_params}")  # Debug: Print search params

        docs = self.vectorstore.similarity_search(
            query,
            k=search_params['k'],
            filter=search_params['filter_metadata']
        )
        print(f"Documents retrieved: {docs}")  # Debug: Print the retrieved documents

        # Create prompt with context
        prompt = self._create_context_prompt(docs, query)

        # Generate response
        response = self.llm.invoke(prompt)

        # Add metadata about sources used
        doc_sources = set(doc.metadata.get('source') for doc in docs)
        source_summary = "\n\nSources consulted:\n" + "\n".join(f"- {src}" for src in doc_sources)

        return response + source_summary

    def get_available_document_types(self) -> Dict[str, int]:
        """Get summary of available document types and counts."""
        collection_metadata = self.vectorstore.get()
        if not collection_metadata:
            return {}

        return {
            'total_documents': collection_metadata.get('total_documents', 0),
            'document_types': collection_metadata.get('document_types', []),
            'total_chunks': collection_metadata.get('total_chunks', 0)
        }

# Example usage
def create_rag_chain(processor: EnhancedRAGProcessor):
    """Create a chain for the RAG process."""

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on the provided context:

    Context: {context}
    Question: {question}

    Instructions:
    - Be specific and cite sources where possible
    - If information is missing, say so
    - Format the response clearly
    """)

    # Create the chain
    chain = (
        {'context': processor.process_query, 'question': RunnablePassthrough()}
        | prompt
        | processor.llm
        | processor.output_parser
    )

    return chain