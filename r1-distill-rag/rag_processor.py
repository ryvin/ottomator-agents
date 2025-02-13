# rag_processor.py
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from thought_logger import ThoughtLogger

@dataclass
class QueryIntent:
    intents: Dict[str, bool]
    search_params: Dict[str, Any]

class EnhancedRAGProcessor:
    """Enhanced RAG processor with improved document retrieval and response generation."""

    def __init__(self, vectorstore: Chroma, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.thought_logger = ThoughtLogger()

    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Analyze query intent for optimized retrieval."""
        patterns = {
            'overview': r'overview|summary|tell me about|what (information|data)|describe',
            'financial': r'financial|revenue|profit|loss|balance|cost|expense|money',
            'comparison': r'compare|versus|vs|difference|better|worse|than|good fit|candidate|qualified|suitable',
            'temporal': r'trend|over time|history|when|date|period|timeline',
            'technical': r'code|implementation|setup|configure|install|run',
            'person': r'who is|about|background|experience|skills|qualifications'
        }

        intents = {
            intent: bool(re.search(pattern, query.lower()))
            for intent, pattern in patterns.items()
        }

        # Adjust retrieval parameters based on intent
        k = 5  # Default number of documents
        if intents['overview'] or intents['person']:
            k = 8
        elif intents['comparison']:
            k = 10  # More documents for comparison queries
        elif any(intents.values()):
            k = 6

        search_params = {
            'k': k,
            'filter': None,
            'fetch_k': k * 2,
            'lambda_mult': 0.7
        }

        # Adjust filters based on intent
        if intents['person']:
            # Prioritize documents about people
            search_params['filter'] = {
                "doc_type": {"$in": [".pdf", ".docx", ".txt"]}  # Documents likely to contain personal info
            }

        # Log the intent analysis
        self.thought_logger.log_step("INTENT_ANALYSIS", {
            "detected_intents": intents,
            "search_parameters": search_params
        })

        return QueryIntent(intents=intents, search_params=search_params)

    def _verify_names(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Verify names found in query against documents."""
        # Extract names from query
        names_in_query = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', query)
        
        # Extract names from documents
        names_in_docs = set()
        for doc in docs:
            # Check filenames for names (often resumes have names in the filename)
            filename = doc.metadata.get('source', '').split('\\')[-1]
            file_names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', filename)
            names_in_docs.update(file_names)
            
            # Check document content for names
            content_names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', doc.page_content)
            names_in_docs.update(content_names)
        
        # Create verification report
        verification = {
            "names_in_query": names_in_query,
            "names_in_documents": list(names_in_docs),
            "verified_names": [name for name in names_in_query if any(name.lower() in doc_name.lower() for doc_name in names_in_docs)],
            "unverified_names": [name for name in names_in_query if not any(name.lower() in doc_name.lower() for doc_name in names_in_docs)]
        }
        
        self.thought_logger.log_step("NAME_VERIFICATION", verification)
        return verification

    def _get_person_specific_context(self, name: str, docs: List[Document]) -> List[Document]:
        """Get context specifically about a person from documents."""
        person_docs = []
        
        for doc in docs:
            filename = doc.metadata.get('source', '').split('\\')[-1]
            
            # Check if document is about the person (either in filename or content)
            if (name.lower() in filename.lower() or 
                name.lower() in doc.page_content.lower()):
                person_docs.append(doc)
                
        return person_docs

    def _format_document_content(self, doc: Document) -> str:
        """Format document content with enhanced metadata."""
        try:
            doc_type = doc.metadata.get('doc_type', '')
            source = doc.metadata.get('source', 'Unknown')
            chunk_index = doc.metadata.get('chunk_index', 0)
            total_chunks = doc.metadata.get('total_chunks', 1)
            
            # Clean up source path for display
            source = source.split('\\')[-1] if '\\' in source else source
            
            header = f"\n=== From {source} (Chunk {chunk_index + 1}/{total_chunks}) ===\n"
            
            # Format content based on document type
            if doc_type in ['.csv', '.xlsx']:
                try:
                    # Format tabular data more clearly
                    lines = doc.page_content.split('\n')
                    if len(lines) > 1:
                        max_lens = [max(len(str(cell)) for cell in col) for col in zip(*[line.split(',') for line in lines])]
                        formatted_lines = []
                        for line in lines:
                            cells = line.split(',')
                            formatted_cells = [str(cell).ljust(length) for cell, length in zip(cells, max_lens)]
                            formatted_lines.append(' | '.join(formatted_cells))
                        content = '\n'.join(formatted_lines)
                    else:
                        content = doc.page_content
                except:
                    content = doc.page_content
            else:
                content = doc.page_content

            return f"{header}{content}"
            
        except Exception as e:
            self.thought_logger.log_error("FORMATTING_ERROR", str(e))
            return str(doc.page_content)

    def _retrieve_relevant_documents(self, query: str, intent: QueryIntent) -> List[Document]:
        """Retrieve and process relevant documents with improved person focus."""
        try:
            # First extract names from query
            names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', query)
            
            all_docs = []
            if names:
                # If we found names, first get documents specific to those people
                for name in names:
                    name_docs = self.vectorstore.similarity_search(
                        name,
                        k=5,
                        filter={"doc_type": {"$in": [".pdf", ".docx", ".txt"]}}
                    )
                    all_docs.extend(name_docs)
            
            # Then get additional context based on the query
            query_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=intent.search_params['k'],
                fetch_k=intent.search_params['fetch_k'],
                lambda_mult=intent.search_params['lambda_mult'],
                filter=intent.search_params['filter']
            )
            
            # Combine and deduplicate documents
            all_docs.extend(query_docs)
            
            # Remove duplicates while preserving order
            seen = set()
            deduplicated_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    deduplicated_docs.append(doc)
            
            # Log retrieval results
            self.thought_logger.log_step("DOCUMENT_RETRIEVAL", {
                "person_documents_found": len(all_docs) - len(query_docs),
                "job_documents_found": len(query_docs),
                "total_unique_documents": len(deduplicated_docs),
                "sources": [doc.metadata.get('source', 'Unknown').split('\\')[-1] 
                          for doc in deduplicated_docs]
            })
            
            return deduplicated_docs
            
        except Exception as e:
            self.thought_logger.log_error("RETRIEVAL_ERROR", str(e))
            return []

def create_chain(self):
        """Create the RAG processing chain with improved person focus."""
        def retrieve_and_format(query: str) -> str:
            try:
                # Analyze query intent
                intent = self._classify_query_intent(query)
                
                # Extract names from query for consistency
                names_in_query = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', query)
                self.thought_logger.log_step("NAME_DETECTION", {
                    "found_names": names_in_query
                })
                
                # Retrieve relevant documents
                docs = self._retrieve_relevant_documents(query, intent)
                
                if not docs:
                    return "I could not find any relevant documents to answer your question. Please try rephrasing your query or ensure that relevant documents are loaded into the system."
                
                # Verify names against documents
                name_verification = self._verify_names(query, docs)
                if names_in_query and not name_verification["verified_names"]:
                    return f"I found no documents containing information about {', '.join(names_in_query)}. Please ensure the relevant documents are loaded into the system."
                
                # Format documents with emphasis on person-specific information
                context_parts = []
                sources_used = set()
                
                # Start with person-specific documents if we have names
                if names_in_query:
                    for name in names_in_query:
                        person_docs = self._get_person_specific_context(name, docs)
                        if person_docs:
                            context_parts.append(f"\n=== Information about {name} ===\n")
                            for doc in person_docs:
                                context_parts.append(self._format_document_content(doc))
                                sources_used.add(doc.metadata.get('source', 'Unknown').split('\\')[-1])
                
                # Add remaining documents
                for doc in docs:
                    if doc not in (person_docs if 'person_docs' in locals() else []):
                        context_parts.append(self._format_document_content(doc))
                        sources_used.add(doc.metadata.get('source', 'Unknown').split('\\')[-1])
                
                # Add source summary
                source_list = "\n\nSources referenced:\n" + "\n".join(f"- {src}" for src in sorted(sources_used))
                
                return "\n".join(context_parts) + source_list
            
            except Exception as e:
                self.thought_logger.log_error("PROCESSING_ERROR", str(e))
                return f"An error occurred while processing: {str(e)}"

        # Create prompt template with improved instructions
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context.

        Context: {context}
        Question: {question}

        CRITICAL INSTRUCTIONS:
        YOU MUST NEVER USE PLACEHOLDER NAMES (like John Doe, John Smith, etc.). 
        Only use names that are explicitly found in the context, exactly as they appear.

        For job fit evaluation:
        1. CANDIDATE VERIFICATION:
           - First check if you have documents containing the candidate's information
           - Verify the candidate's name appears in these documents
           - Extract qualifications ONLY from documents about this specific person
        
        2. TWO-PART ANALYSIS:
           A. Candidate's Documented Qualifications:
              - List qualifications found in their documents
              - Include specific evidence with document citations
              - Note the dates/timeline of their experience
           
           B. Job Requirements Analysis:
              - Break down the job's key requirements
              - Compare each requirement to the candidate's documented qualifications
              - Explicitly state when you can't find information about a requirement
        
        3. Evidence-Based Conclusion:
           - Compare only verified qualifications against requirements
           - Highlight clear matches and gaps
           - State confidence level based on available information
        
        4. Missing Information:
           - List any critical requirements that couldn't be verified
           - Suggest what additional information would help
        
        Remember: 
        - Use ONLY information from the documents
        - Never make assumptions about qualifications
        - Always cite the specific documents for claims
        - If you can't find the person's information, clearly state this
        
        Format your response in clear sections:
        1. Candidate Verification
        2. Documented Qualifications
        3. Job Requirements Analysis
        4. Gaps and Missing Information
        5. Evidence-Based Conclusion

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