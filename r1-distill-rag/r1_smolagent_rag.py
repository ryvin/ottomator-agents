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
            prompt = f"""Based on the following context, answer the question. 
If the context doesn't contain relevant information, clearly state that.

Context:
{context}

Question: {query}

Structure your response as follows:
1. Start with "Thoughts:" followed by a brief analysis
2. Then use this exact code block format:

Thoughts: Your brief analysis here
Code:
```py
print('''
Your detailed response here.
Make it clear and structured.
''')
```"""
            
            # Get response from reasoning model
            response = self.reasoner.run(prompt, reset=False)
            
            # If response is missing proper format, add it
            if not ("Thoughts:" in response and "```py" in response):
                response = f"""Thoughts: Analyzing available information
Code:
```py
print('''{response}''')
```"""
            
            return response
                
        except Exception as e:
            return f"""Thoughts: Error processing query
Code:
```py
print('''Error: {str(e)}
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
def process_query(user_query: str) -> str:
    """Process a user query using RAG and reasoning.
    
    Args:
        user_query: The question to answer.
    
    Returns:
        str: The reasoned response based on available documents.
    """
    return rag_reasoner.process_query(user_query)

# Create the primary agent
tool_model = get_model(tool_model_id)
system_prompt = """You are an AI assistant that helps answer questions using RAG.

{{managed_agents_descriptions}}

Your role is to:
1. Use the process_query tool to get information
2. Return the tool's response exactly as provided
3. Do not modify or reformat the response
4. Do not add calculations or extra processing

Important:
- Always use the process_query tool
- Return its response without modification
- Do not perform calculations
- Maintain exact formatting"""

primary_agent = ToolCallingAgent(
    tools=[process_query], 
    model=tool_model, 
    add_base_tools=False,
    max_steps=2,
    system_prompt=system_prompt
)

def main():
    ui = GradioUI(primary_agent)
    ui.launch()

if __name__ == "__main__":
    main()