# prompts.py
from langchain_core.prompts import ChatPromptTemplate

RAG_ANALYSIS_PROMPT = """
Answer the following question based on the provided context. The context includes relevant documents and source information.

Context: {context}
Question: {question}

CRITICAL INSTRUCTIONS:

FOR NAME VERIFICATION:
- Only refer to people using names that appear in the provided documents
- If a name isn't found in the documents, explicitly state this
- Never use generic names (like John Doe) or make assumptions
- Always cite the specific document when mentioning qualifications

FOR JOB FIT ANALYSIS:
1. First verify the candidate:
   - Confirm if their resume/CV is in the provided documents
   - Only use information from their actual documents
   - State which documents contain their information

2. Document Review:
   - List ONLY qualifications found in their documents
   - Include dates and timeline information
   - Quote specific relevant experience
   - Cite the source document for each qualification

3. Requirements Analysis:
   - Compare documented qualifications to job requirements
   - Mark requirements as:
     ✓ Confirmed (with evidence from documents)
     ? Unclear (insufficient information)
     ⚠ Not Found (no evidence in documents)

4. Gaps Analysis:
   - List requirements without clear evidence
   - Note which qualifications need verification
   - Suggest what additional information is needed

5. Overall Assessment:
   - Only based on verified information
   - Note confidence level in assessment
   - List key strengths and potential gaps

Format the response clearly with these sections. Only state what is explicitly supported by the documents.

Answer:
"""

QUERY_CLASSIFICATION_PATTERNS = {
    'overview': r'overview|summary|tell me about|what (information|data)|describe',
    'financial': r'financial|revenue|profit|loss|balance|cost|expense|money',
    'comparison': r'compare|versus|vs|difference|better|worse|than|good fit|candidate|qualified|suitable',
    'temporal': r'trend|over time|history|when|date|period|timeline',
    'technical': r'code|implementation|setup|configure|install|run',
    'person': r'who is|about|background|experience|skills|qualifications'
}

JOB_DETECTION_PATTERNS = {
    'job_section': r'About the job|About the Role|Job Description|Position Description',
    'requirements_section': r'Requirements|What you will need|Qualifications|Skills|Experience Required'
}

def get_rag_prompt() -> ChatPromptTemplate:
    """Get the RAG analysis prompt template."""
    return ChatPromptTemplate.from_template(RAG_ANALYSIS_PROMPT)

def get_job_patterns() -> dict:
    """Get patterns for job detection."""
    return JOB_DETECTION_PATTERNS

def get_query_patterns() -> dict:
    """Get patterns for query classification."""
    return QUERY_CLASSIFICATION_PATTERNS