# main.py docstring
"""
Smart Document Assistant - Main Application

This application provides an intelligent document processing and question-answering system
using RAG (Retrieval Augmented Generation) technology. It processes various document types,
creates a searchable knowledge base, and provides natural language answers to user queries.

Key Components:
    - Document Processing: Handles multiple file formats with adaptive chunking
    - Vector Store: Creates and manages document embeddings for efficient retrieval
    - RAG Processing: Combines document retrieval with language model generation
    - Error Handling: Comprehensive error catching and reporting
    - Logging: Detailed activity and error logging

Usage:
    1. Place documents in the 'data' directory
    2. Run the application: python main.py
    3. Ask questions about document content

Author: [Your Name]
Version: 1.0.0
Date: February 2024
"""

# DocumentProcessor class docstring
"""
Handles document loading and processing with enhanced metadata.

This class provides comprehensive document processing capabilities including:
    - Multi-format document loading (PDF, DOCX, CSV, etc.)
    - Adaptive text chunking based on document type
    - Enhanced metadata generation
    - Error handling and validation

Attributes:
    SUPPORTED_EXTENSIONS (dict): Mapping of file extensions to loader classes
    CHUNK_SIZES (dict): Optimal chunk sizes for different document types

Methods:
    load_documents: Load and validate documents from a directory
    process_documents: Process documents into chunks with metadata
"""

# VectorStoreManager class docstring
"""
Manages vector store operations for document embeddings.

This class handles the creation, loading, and management of document embeddings,
providing efficient similarity search capabilities for document retrieval.

Key Features:
    - Automatic handling of new vs. existing vector stores
    - Metadata tracking and management
    - Efficient document embedding and retrieval
    - Persistence management

Attributes:
    persist_dir (str): Directory for vector store persistence
    embeddings (HuggingFaceEmbeddings): Document embedding model
    vectorstore (Optional[Chroma]): Active vector store instance

Methods:
    create_or_load_vectorstore: Initialize or load existing vector store
    get_metadata: Retrieve vector store statistics and information
"""

# EnhancedRAGProcessor class docstring
"""
Enhanced RAG processor for intelligent document query handling.

This class combines document retrieval with language model generation to provide
accurate and context-aware answers to user queries about document content.

Features:
    - Query intent classification
    - Context-aware document retrieval
    - Smart answer generation
    - Source citation and tracking

Attributes:
    vectorstore (Chroma): Vector store for document retrieval
    llm: Language model for answer generation
    output_parser (StrOutputParser): Response formatting

Methods:
    create_chain: Create processing chain for query handling
    _classify_query_intent: Analyze and optimize query handling
    _format_document_content: Format retrieved content for clarity
"""