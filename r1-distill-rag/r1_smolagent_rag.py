from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    """Get the appropriate model based on environment settings."""
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(
            model_id=model_id, 
            token=huggingface_api_token
        )
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

class RetryableRagReasoner:
    def __init__(self, reasoning_model, vectordb):
        self.reasoner = CodeAgent(
            tools=[], 
            model=reasoning_model, 
            add_base_tools=False, 
            max_steps=2,
            system_prompt="""You are an AI assistant that helps analyze documents.

{{managed_agents_descriptions}}

{{authorized_imports}}
import sys
import os
{{/authorized_imports}}

Follow this EXACT format:

Thoughts: [Your analysis]
Code:
```python
print('''
[Your response]
''')
"""
        )
        self.vectordb = vectordb
        self.max_retries = 3
    
    def format_response(self, response: str) -> str:
        """Ensure proper response format."""
        if not response:
            return self._get_error_response("Empty response received")
        
        try:
            # Extract thoughts
            thoughts = "Analyzing available information"
            thoughts_match = re.search(r"Thoughts:(.*?)(?=Code:|$)", response, re.DOTALL)
            if thoughts_match:
                thoughts = thoughts_match.group(1).strip()
            
            # Get content from print statement if present
            content = response
            print_match = re.search(r"print\([\"']{{3}}(.*?)[\"']{{3}}\)", response, re.DOTALL)
            if print_match:
                content = print_match.group(1).strip()
            else:
                # Clean up content
                content = re.sub(r"Thoughts:.*?(?=Code:|$)", "", response, flags=re.DOTALL)
                content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
                content = content.strip()
            
            # Format response
            return f"""Thoughts: {thoughts}
Code:
```python
print('''{content}''')
"""
        except Exception as e:
            return self._get_error_response(f"Error formatting response: {str(e)}")
    
    def _get_error_response(self, error_msg: str) -> str:
        """Generate a properly formatted error response."""
        return f"""Thoughts: Error occurred while processing
Code:
```python
print('''Error: {error_msg}
Please try again or rephrase your question.''')
"""
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a formatted response."""
        if not query:
            return self._get_error_response("Empty query received")
            
        for attempt in range(self.max_retries):
            try:
                # Get relevant documents
                docs = self.vectordb.similarity_search(query, k=3)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                # Create prompt
                prompt = f"""Based on this context, answer the question.

Context:
{context}

Question: {query}

Follow this EXACT format:

Thoughts: [Your analysis]
Code:
```python
print('''
[Your response]
''')
"""
                
                # Get and format response
                response = self.reasoner.run(prompt, reset=True)
                if response:
                    return self.format_response(response)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return self._get_error_response(f"Failed after {self.max_retries} attempts. Error: {str(e)}")
                continue
        
        return self._get_error_response("Failed to generate a valid response")

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Create reasoning model
reasoning_model = get_model(reasoning_model_id)
rag_reasoner = RetryableRagReasoner(reasoning_model, vectordb)

@tool
def process_query(user_query: str) -> str:
    """Process a user query using RAG and reasoning.
    
    Args:
        user_query: The question to answer.
    
    Returns:
        str: The reasoned response based on available documents.
    """
    if not user_query:
        return "Error: Empty query received"
    try:
        response = rag_reasoner.process_query(user_query)
        return response if response else "Error: No response generated"
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Create the primary agent
tool_model = get_model(tool_model_id)
system_prompt = """You are an AI assistant that helps answer questions using RAG.

{{managed_agents_descriptions}}

Role:
1. Use process_query tool
2. Return exact response
3. No modifications
4. Handle errors

Always:
- Use tool
- Return output
- Keep format"""

# Create the agent with proper configuration
primary_agent = ToolCallingAgent(
    tools=[process_query], 
    model=tool_model, 
    add_base_tools=False,
    max_steps=4,
    system_prompt=system_prompt
)

def main():
    ui = GradioUI(primary_agent)
    ui.launch()

if __name__ == "__main__":
    main()