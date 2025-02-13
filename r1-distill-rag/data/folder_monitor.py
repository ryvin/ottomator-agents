# folder_monitor.py
from typing import Callable, Set
from pathlib import Path
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

class DocumentFolderHandler(FileSystemEventHandler):
    """Handles document folder change events."""
    
    def __init__(self, callback: Callable, logger: logging.Logger):
        self.callback = callback
        self.logger = logger
        self.last_processed: Set[str] = set()
        self._processing = False
        
    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_change("created", event.src_path)
        
    def on_deleted(self, event):
        if event.is_directory:
            return
        self._handle_change("deleted", event.src_path)
        
    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_change("modified", event.src_path)
        
    def _handle_change(self, event_type: str, file_path: str):
        """Handle file change events with debouncing."""
        if self._processing:
            return
            
        try:
            self._processing = True
            
            # Check if file type is supported
            file_path = Path(file_path)
            if not any(file_path.name.endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.csv', '.xlsx', '.md']):
                return
                
            # Avoid processing the same event multiple times
            event_key = f"{event_type}_{file_path}"
            if event_key in self.last_processed:
                return
                
            self.last_processed.add(event_key)
            if len(self.last_processed) > 100:  # Prevent unlimited growth
                self.last_processed.clear()
                
            self.logger.info(f"Document {event_type}: {file_path}")
            self.callback(event_type, file_path)
            
        finally:
            self._processing = False

class FolderMonitor:
    """Monitors a folder for document changes."""
    
    def __init__(self, callback: Callable, logger: logging.Logger):
        self.callback = callback
        self.logger = logger
        self.observer = None
        self.current_path = None
        
    def start_monitoring(self, folder_path: str):
        """Start monitoring a folder for changes."""
        try:
            if self.observer:
                self.stop_monitoring()
                
            folder_path = Path(folder_path)
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
                
            self.current_path = folder_path
            self.observer = Observer()
            handler = DocumentFolderHandler(self.callback, self.logger)
            self.observer.schedule(handler, str(folder_path), recursive=False)
            self.observer.start()
            self.logger.info(f"Started monitoring folder: {folder_path}")
            
        except Exception as e:
            self.logger.error(f"Error starting folder monitor: {e}")
            raise
            
    def stop_monitoring(self):
        """Stop monitoring the current folder."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.logger.info("Stopped folder monitoring")
            
    def get_current_path(self) -> str:
        """Get the currently monitored folder path."""
        return str(self.current_path) if self.current_path else ""