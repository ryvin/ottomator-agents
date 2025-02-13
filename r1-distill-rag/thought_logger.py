# thought_logger.py
import logging
from pathlib import Path
from datetime import datetime
import json
import os
from typing import Dict, Any

class ThoughtLogger:
    """Logger for RAG thinking process and debugging."""
    
    def __init__(self, log_dir: str = "thoughts"):
        # Ensure log directory exists
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"thought_log_{self.session_timestamp}.jsonl"
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up the logging configuration."""
        self.logger = logging.getLogger(f"ThoughtLogger_{self.session_timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for JSON lines
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
    def log_step(self, step_name: str, details: Dict[str, Any], query: str = None):
        """Log a processing step with details."""
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "step": step_name,
            "query": query,
            "details": details
        }
        
        # Log to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Print to console
        print(f"\n=== {step_name.upper()} ===")
        self._print_dict(details)
        
    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        """Helper to pretty print nested dictionaries."""
        for key, value in d.items():
            indent_str = "  " * indent
            if isinstance(value, dict):
                print(f"{indent_str}{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{indent_str}{key}:")
                for item in value:
                    print(f"{indent_str}  - {item}")
            else:
                print(f"{indent_str}{key}: {value}")
    
    def log_error(self, error_type: str, error_msg: str, details: Dict[str, Any] = None):
        """Log an error with context."""
        error_details = {
            "error_type": error_type,
            "error_message": error_msg,
            **(details or {})
        }
        self.log_step("ERROR", error_details)
    
    def log_query_start(self, query: str):
        """Log the start of a new query."""
        self.log_step("QUERY_START", {"query": query}, query)
    
    def log_query_end(self, query: str, success: bool, duration: float):
        """Log the completion of a query."""
        self.log_step("QUERY_END", {
            "query": query,
            "success": success,
            "duration_seconds": duration
        }, query)