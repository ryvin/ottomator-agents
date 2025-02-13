import psutil
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class SystemStatus:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    vectorstore_size: int
    document_count: int
    is_healthy: bool
    
class SystemMonitor:
    def __init__(self, data_dir: str, db_dir: str, logger: logging.Logger):
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.logger = logger
        
    def get_system_status(self) -> SystemStatus:
        """Get current system status and resources."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Application metrics
            vectorstore_size = sum(f.stat().st_size for f in self.db_dir.rglob('*') if f.is_file())
            document_count = len(list(self.data_dir.rglob('*')))
            
            # Health check
            is_healthy = all([
                cpu_usage < 95,
                memory_usage < 95,
                disk_usage < 95
                # Removed vectorstore size check as it might be empty initially
            ])
            
            return SystemStatus(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                vectorstore_size=vectorstore_size,
                document_count=document_count,
                is_healthy=is_healthy
            )
            
        except Exception as e:
            self.logger.error(f"Error in system monitoring: {e}")
            return SystemStatus(0, 0, 0, 0, 0, False)
    
    def check_llm_availability(self) -> tuple[bool, str]:
        """Check if the LLM service is available and return status with message."""
        try:
            import requests
            # Use list endpoint instead of health endpoint
            ollama_url = "http://localhost:11434/api/tags"
            self.logger.info(f"Checking Ollama availability at {ollama_url}")
            
            response = requests.get(ollama_url, timeout=5)
            self.logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Log available models
                available_models = [model.get("name") for model in models]
                self.logger.info(f"Available models: {available_models}")
                
                if any("deepseek-r1:32b" in model for model in available_models):
                    return True, "Ollama service and model are available"
                else:
                    return False, "deepseek-r1:32b model is not found in available models. Please run: ollama pull deepseek-r1:32b"
            else:
                return False, f"Ollama service returned unexpected status code: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to Ollama service. Is it running on port 11434?"
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama service timed out"
        except Exception as e:
            return False, f"Error checking Ollama service: {str(e)}"
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}