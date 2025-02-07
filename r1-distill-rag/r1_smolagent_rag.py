from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

class RagReasoner:
    def __init__(self, reasoning_model, vectordb):
        self.reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)
        self.vectordb = vectordb
    
    def process_query(self, query: str) -> str:
        try:
            # Search for relevant documents
            docs = self.vectordb.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Create the reasoning prompt
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Remember to structure your response exactly as:

Thoughts: Brief analysis of the context and question
Code:
```py
print('''
Your detailed answer here.
Start with a clear summary,
then provide supporting details,
and end with key takeaways.
''')
```
"""
            # Get response from reasoning model
            response = self.reasoner.run(prompt, reset=False)
            
            # Ensure proper formatting
            if "```py" in response and "Thoughts:" in response:
                return response
                
            return f"""Thoughts: Analyzing the provided information
Code:
```py
print('''{response}''')
```"""
                
        except Exception as e:
            return f"""Thoughts: Error occurred while processing
Code:
```py
print('''Unable to process query: {str(e)}
Please try rephrasing your question.''')
```"""

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Create the reasoner
reasoning_model = get_model(reasoning_model_id)
rag_reasoner = RagReasoner(reasoning_model, vectordb)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """Search and reason over documents to answer questions.
    
    Args:
        user_query: The question to search for and answer.
    
    Returns:
        A formatted response with reasoning based on found documents.
    """
    return rag_reasoner.process_query(user_query)

# Create the primary agent
tool_model = get_model(tool_model_id)
system_prompt = """You are an AI assistant that helps answer questions using RAG.
Your role is to:
1. Use the rag_with_reasoner tool to search for and analyze information
2. Review the reasoning model's output
3. Present the findings clearly

{{managed_agents_descriptions}}

The rag_with_reasoner tool will provide responses in this format:
Thoughts: Brief analysis
Code:
```py
print('''Your detailed answer here''')
```

Your job is to use this tool and relay its responses accurately."""

primary_agent = ToolCallingAgent(
    tools=[rag_with_reasoner], 
    model=tool_model, 
    add_base_tools=False, 
    max_steps=3,
    system_prompt=system_prompt
)

def main():
    ui = GradioUI(primary_agent)
    ui.launch()

if __name__ == "__main__":
    main()