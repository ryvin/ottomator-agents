# error_handling.py
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessingError:
    """
    Container for processing errors that occur during document handling.
    
    Attributes:
        error_type (str): Type of error that occurred
        message (str): Detailed error message
        source (Optional[str]): Source of the error (e.g., file path)
        details (Optional[Any]): Additional error details
    """
    error_type: str
    message: str
    source: Optional[str] = None
    details: Optional[Any] = None

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    def __init__(self, error: ProcessingError):
        self.error = error
        super().__init__(f"{error.error_type}: {error.message}")

def validate_file_access(file_path: Path) -> None:
    """
    Validate file accessibility and permissions.
    
    Args:
        file_path (Path): Path to the file to validate
        
    Raises:
        DocumentProcessingError: If file validation fails
    """
    if not file_path.exists():
        raise DocumentProcessingError(ProcessingError(
            "FileNotFound",
            f"File does not exist: {file_path}"
        ))
    
    if not file_path.is_file():
        raise DocumentProcessingError(ProcessingError(
            "InvalidFile",
            f"Path is not a file: {file_path}"
        ))
    
    try:
        with open(file_path, 'rb'):
            pass
    except PermissionError:
        raise DocumentProcessingError(ProcessingError(
            "PermissionError",
            f"No permission to access file: {file_path}"
        ))
    except Exception as e:
        raise DocumentProcessingError(ProcessingError(
            "FileAccessError",
            f"Cannot access file: {file_path}",
            details=str(e)
        ))

def validate_directory_access(dir_path: Path) -> None:
    """
    Validate directory accessibility and permissions.
    
    Args:
        dir_path (Path): Path to the directory to validate
        
    Raises:
        DocumentProcessingError: If directory validation fails
    """
    if not dir_path.exists():
        raise DocumentProcessingError(ProcessingError(
            "DirectoryNotFound",
            f"Directory does not exist: {dir_path}"
        ))
    
    if not dir_path.is_dir():
        raise DocumentProcessingError(ProcessingError(
            "InvalidDirectory",
            f"Path is not a directory: {dir_path}"
        ))
    
    try:
        # Test directory write permissions
        test_file = dir_path / ".test_write_permission"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise DocumentProcessingError(ProcessingError(
            "PermissionError",
            f"No write permission in directory: {dir_path}"
        ))
    except Exception as e:
        raise DocumentProcessingError(ProcessingError(
            "DirectoryAccessError",
            f"Cannot access directory: {dir_path}",
            details=str(e)
        ))