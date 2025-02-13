from pathlib import Path
from dotenv import load_dotenv
import os
import shutil
from typing import Tuple
from thought_logger import ThoughtLogger

from models import get_llm
from document_loader import DocumentProcessor
from vectorstore import VectorStoreManager
from rag_processor import EnhancedRAGProcessor
from gui import EnhancedRAGUI  # Using the enhanced version
from logger_config import setup_logging
from system_monitor import SystemMonitor

def setup_directories() -> Tuple[str, str]:
    """Setup necessary directories and return their paths."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    db_dir = base_dir / "chroma_db"
    logs_dir = base_dir / "logs"
    
    # Create all directories
    for dir_path in [data_dir, db_dir, logs_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Clean up vector store if needed
    if db_dir.exists() and any(db_dir.iterdir()):
        try:
            shutil.rmtree(str(db_dir))
            db_dir.mkdir()
        except PermissionError:
            print("Warning: Could not clear existing vector store.")
    
    return str(data_dir), str(db_dir)

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Smart Document Assistant")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Setup directories
        data_dir, db_dir = setup_directories()
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Database directory: {db_dir}")
        
        # Set data directory in environment for GUI
        os.environ["DATA_DIR"] = data_dir
        
        # Initialize system monitor
        monitor = SystemMonitor(data_dir, db_dir, logger)
        
        # Check system health
        status = monitor.get_system_status()
        if not status.is_healthy:
            logger.warning("System health check failed!")
        
        # Check LLM availability
        llm_available, llm_message = monitor.check_llm_availability()
        if not llm_available:
            logger.warning(f"LLM Service Issue: {llm_message}")
            logger.warning("\nPlease ensure:")
            logger.warning("1. Ollama is installed (https://ollama.com/download)")
            logger.warning("2. Ollama service is running ('ollama serve')")
            logger.warning("3. The model is pulled ('ollama pull deepseek-r1:32b')")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Initialize components
        doc_processor = DocumentProcessor()
        documents = doc_processor.load_documents(data_dir)
        
        if not documents:
            logger.warning("No documents found. Please add documents to the data directory.")
            return
            
        logger.info(f"Loaded {len(documents)} documents")
        chunks = doc_processor.process_documents(documents)
        
        # Initialize vector store
        vector_manager = VectorStoreManager(db_dir)
        vectorstore = vector_manager.create_or_load_vectorstore(chunks)
        
        if not vectorstore:
            logger.error("Failed to initialize vector store")
            return
        
        # Initialize LLM and RAG processor
        model = get_llm(os.getenv("REASONING_MODEL_ID"))
        processor = EnhancedRAGProcessor(vectorstore, model)
        
        # Create and launch GUI with all required components
        logger.info("Starting GUI...")
        ui = EnhancedRAGUI(
            processor=processor,
            doc_processor=doc_processor,
            vector_manager=vector_manager,
            logger=logger
        )
        ui.launch(share=True)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()