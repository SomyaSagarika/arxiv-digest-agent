# graph.py
import os
from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState
from ingestion import (
    detect_query_type,
    fetch_paper_by_id,
    search_papers_by_topic,
    download_and_extract_pdf,
)
from rag import build_vector_store
from models import ExecutiveBriefing

load_dotenv()


# Initialize Groq LLM using confirmed accessible model
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ----------------- GRAPH NODES -----------------

def understand_query_node(state: AgentState) -> dict:
    """Classify user query into direct arXiv ID lookup or natural topic search."""
    user_input = state["user_input"]
    q_type, arxiv_id = detect_query_type(user_input)
    return {
        "query_type": q_type,
        "arxiv_id": arxiv_id,
        "error": None
    }

def route_query(state: AgentState) -> Literal["fetch_direct_paper", "search_and_rank"]:
    """Conditional edge routing based on parsed query type."""
    if state["query_type"] == "paper_id":
        return "fetch_direct_paper"
    return "search_and_rank"

def fetch_direct_paper_node(state: AgentState) -> dict:
    """Fetch metadata for a known arXiv ID."""
    arxiv_id = state["arxiv_id"]
    paper = fetch_paper_by_id(arxiv_id)
    if not paper:
        return {"error": f"Could not find arXiv paper with ID: {arxiv_id}"}
    return {"paper_metadata": paper}

def search_and_rank_node(state: AgentState) -> dict:
    """Search candidates for a topic and use LLM to select the most relevant one."""
    topic = state["user_input"]
    candidates = search_papers_by_topic(topic, max_results=5)
    
    if not candidates:
        return {"error": f"No arXiv papers found for query: '{topic}'"}
    
    # Prompt LLM to choose the best candidate
    candidate_summaries = "\n\n".join([
        f"[{i+1}] ID: {c['arxiv_id']}\nTitle: {c['title']}\nAbstract: {c['summary'][:300]}..."
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""You are a research paper curator. A user searched for: "{topic}".
Here are the top candidates retrieved from arXiv:
{candidate_summaries}

Select the single most relevant paper to the user's intent.
Respond with ONLY the exact arXiv ID of that paper (e.g. 2305.18290) and nothing else."""
    
    response = llm.invoke(prompt)
    chosen_id = response.content.strip().split()[0]
    
    # Match chosen ID against candidates, fallback to top candidate if not matched
    selected = next((c for c in candidates if c['arxiv_id'] in chosen_id), candidates[0])
    return {
        "candidate_papers": candidates,
        "arxiv_id": selected["arxiv_id"],
        "paper_metadata": selected
    }

def parse_and_chunk_node(state: AgentState) -> dict:
    """Download PDF, extract text with fallback, and build in-memory vector store."""
    if state.get("error"):
        return {}
        
    paper = state["paper_metadata"]
    pdf_path, text, is_fallback = download_and_extract_pdf(paper["pdf_url"])
    
    # Build vector store
    collection = build_vector_store(
        paper_id=paper["arxiv_id"],
        text=text if not is_fallback else paper["summary"],
        abstract=paper["summary"]
    )
    
    return {
        "pdf_path": pdf_path,
        "extracted_text": text,
        "parse_fallback": is_fallback,
        "vector_store": collection
    }

def summarize_node(state: AgentState) -> dict:
    """Generate structured executive briefing artifact."""
    if state.get("error"):
        return {}
        
    paper = state["paper_metadata"]
    text_content = state["extracted_text"]
    
    # Provide representative context (first ~12,000 chars covers abstract, intro, and early methods)
    context = text_content[:12000] if not state["parse_fallback"] else f"Abstract: {paper['summary']}"
    
    structured_llm = llm.with_structured_output(ExecutiveBriefing)
    
    prompt = f"""You are an expert AI research scientist. Produce a structured executive briefing for this paper.
Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
arXiv ID: {paper['arxiv_id']}
Published: {paper['published']}
Link: {paper['pdf_url']}

Paper Content (Excerpts):
\"\"\"
{context}
\"\"\"

Be precise, objective, and ensure the 'limitations' field highlights genuine engineering or methodological tradeoffs."""

    briefing = structured_llm.invoke(prompt)
    return {"briefing": briefing.model_dump()}

# ----------------- BUILD GRAPH -----------------
# ----------------- BUILD GRAPH -----------------

def create_agent_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("understand_query", understand_query_node)
    graph.add_node("fetch_direct_paper", fetch_direct_paper_node)
    graph.add_node("search_and_rank", search_and_rank_node)
    graph.add_node("parse_and_chunk", parse_and_chunk_node)
    graph.add_node("summarize", summarize_node)
    
    # Set entry point
    graph.set_entry_point("understand_query")
    
    # Conditional branching (MAPPED TO EXACT NODE NAMES)
    graph.add_conditional_edges(
        "understand_query",
        route_query,
        {
            "fetch_direct_paper": "fetch_direct_paper",
            "search_and_rank": "search_and_rank"
        }
    )
    
    # Linear edges
    graph.add_edge("fetch_direct_paper", "parse_and_chunk")
    graph.add_edge("search_and_rank", "parse_and_chunk")
    graph.add_edge("parse_and_chunk", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()
