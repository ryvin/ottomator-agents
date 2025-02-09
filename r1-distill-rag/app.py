# app.py
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from improved_rag_processor import EnhancedRAGProcessor
import os

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processor" not in st.session_state:
        st.session_state.processor = None

def initialize_rag():
    """Initialize the RAG system."""
    try:
        # Setup embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        if not os.path.exists(db_dir):
            st.error("Vector store not found. Please run ingest.py first.")
            return None
            
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings
        )
        
        # Initialize processor
        from r1_smolagent_rag import get_model
        model = get_model(os.getenv("REASONING_MODEL_ID", "deepseek-r1:7b-8k"))
        processor = EnhancedRAGProcessor(vectordb, model)
        
        return processor
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("Document Q&A System")
        st.markdown("""
        This system uses RAG (Retrieval Augmented Generation) to answer questions about your documents.
        
        ### Features
        - Context-aware responses
        - Document source tracking
        - Support for multiple document types
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("Ask Questions About Your Documents")
    
    # Initialize RAG processor if needed
    if st.session_state.processor is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.processor = initialize_rag()
            if st.session_state.processor is None:
                st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about your documents?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.processor.process_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()