# state.py
from typing import TypedDict, List, Optional, Dict, Any

class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    pdf_url: str
    summary: str  # arXiv abstract

class AgentState(TypedDict):
    # User inputs
    user_input: str
    query_type: str                   # "paper_id" | "topic"
    arxiv_id: Optional[str]
    
    # Retrieved paper and extracted text
    paper_metadata: Optional[PaperMetadata]
    candidate_papers: List[PaperMetadata]
    pdf_path: Optional[str]
    extracted_text: str
    parse_fallback: bool             # True if PDF failed & used abstract fallback
    
    # Vector store instance for QA
    vector_store: Optional[Any]      
    
    # Generation outputs
    briefing: Optional[Dict[str, Any]]
    qa_history: List[Dict[str, str]]
    error: Optional[str]