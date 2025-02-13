# logger_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_dir (str): Directory where log files will be stored. Defaults to "logs".
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"rag_processor_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("RAGProcessor")
    logger.setLevel(logging.DEBUG)
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger