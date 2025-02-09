# models.py
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaLLM

def get_llm(model_id: str):
    """Get the appropriate LLM based on environment settings."""
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    
    if using_huggingface:
        return HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        )
    else:
        return OllamaLLM(
            model=model_id,
            base_url="http://localhost:11434",
        )