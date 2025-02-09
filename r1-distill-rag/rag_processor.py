# rag_processor.py
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

@dataclass
class QueryIntent:
    intents: Dict[str, bool]
    search_params: Dict[str, Any]

class EnhancedRAGProcessor:
    """Enhanced RAG processor with advanced query handling and response generation."""

    def __init__(self, vectorstore: Chroma, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.output_parser = StrOutputParser()

    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Analyze query intent for optimized retrieval."""
        patterns = {
            'overview': r'overview|summary|tell me about|what (information|data)|describe',
            'financial': r'financial|revenue|profit|loss|balance|cost|expense|money',
            'comparison': r'compare|versus|vs|difference|better|worse|than',
            'temporal': r'trend|over time|history|when|date|period|timeline',
            'technical': r'code|implementation|setup|configure|install|run'
        }

        intents = {
            intent: bool(re.search(pattern, query.lower()))
            for intent, pattern in patterns.items()
        }

        # Default search parameters
        search_params = {
            'k': 5,
            'filter': None  # Initialize with no filter
        }

        # Adjust parameters based on intent
        if intents['overview']:
            search_params['k'] = 8
        elif intents['financial']:
            search_params['filter'] = {"doc_type": {"$in": [".csv", ".xlsx", ".docx"]}}
        elif intents['technical']:
            search_params['filter'] = {"doc_type": {"$in": [".md", ".py"]}}

        return QueryIntent(intents=intents, search_params=search_params)

    def _format_document_content(self, doc: Document) -> str:
        """Format document content based on type."""
        doc_type = doc.metadata.get('doc_type', '')
        content = doc.page_content

        if doc_type in ['.csv', '.xlsx']:
            lines = content.split('\n')
            if len(lines) > 1:
                try:
                    rows = [line.split(',') for line in lines if line.strip()]
                    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
                    formatted_rows = [
                        ' | '.join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
                        for row in rows
                    ]
                    content = '\n'.join(formatted_rows)
                except:
                    pass

        elif doc_type == '.md':
            content = re.sub(r'\n\s*\n', '\n\n', content).strip()

        return f"{content}\n[Source: {doc.metadata.get('source', 'Unknown')}]"

    def create_chain(self):
        """Create the RAG processing chain."""
        def retrieve_and_format(query: str) -> str:
            # Analyze query and get search parameters
            intent = self._classify_query_intent(query)
            
            try:
                # Retrieve relevant documents
                docs = self.vectorstore.similarity_search(
                    query,
                    k=intent.search_params['k'],
                    filter=intent.search_params['filter']
                )

                # Group and format documents
                docs_by_type = {}
                for doc in docs:
                    doc_type = doc.metadata.get('doc_type', 'unknown')
                    if doc_type not in docs_by_type:
                        docs_by_type[doc_type] = []
                    docs_by_type[doc_type].append(doc)

                # Create formatted context
                context_parts = []
                for doc_type, type_docs in docs_by_type.items():
                    context_parts.append(f"\n=== Content from {doc_type} files ===\n")
                    for doc in type_docs:
                        context_parts.append(self._format_document_content(doc))

                # Add source summary
                sources = set(doc.metadata.get('source') for doc in docs)
                source_list = "\n\nSources consulted:\n" + "\n".join(f"- {src}" for src in sources)
                
                return "\n".join(context_parts) + source_list
            
            except Exception as e:
                print(f"Error in document retrieval: {str(e)}")
                return f"An error occurred while retrieving documents: {str(e)}"

        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context:

        Context: {context}
        Question: {question}

        Instructions:
        - Use only information from the context
        - If the context doesn't contain enough information, say so
        - Cite sources when providing information
        - Format the response clearly

        Answer:
        """)

        # Create and return the chain
        chain = (
            {"context": retrieve_and_format, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | self.output_parser
        )

        return chain