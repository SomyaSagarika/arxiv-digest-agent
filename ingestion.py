# ingestion.py
import re
import os
import urllib.request
import pymupdf as fitz # PyMuPDF
import arxiv
from typing import Optional, List, Tuple
from state import PaperMetadata

# Matches: 2401.12345, 1706.03762, https://arxiv.org/abs/2401.12345, etc.
ARXIV_ID_REGEX = r"(?:arxiv\.org\/(?:abs|pdf)\/|arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)"

def detect_query_type(query: str) -> Tuple[str, Optional[str]]:
    """Determine whether the input is a specific arXiv ID/URL or a topic search."""
    match = re.search(ARXIV_ID_REGEX, query.strip(), re.IGNORECASE)
    if match:
        return "paper_id", match.group(1)
    return "topic", None

def fetch_paper_by_id(arxiv_id: str) -> Optional[PaperMetadata]:
    """Retrieve metadata for a specific arXiv ID using arXiv Atom API."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        return None
    
    paper = results[0]
    return PaperMetadata(
        arxiv_id=paper.get_short_id(),
        title=paper.title,
        authors=[author.name for author in paper.authors],
        published=paper.published.strftime("%Y-%m-%d"),
        pdf_url=paper.pdf_url,
        summary=paper.summary
    )

def search_papers_by_topic(topic: str, max_results: int = 5) -> List[PaperMetadata]:
    """Search candidate papers for a natural-language topic."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    candidates = []
    for paper in client.results(search):
        candidates.append(PaperMetadata(
            arxiv_id=paper.get_short_id(),
            title=paper.title,
            authors=[author.name for author in paper.authors],
            published=paper.published.strftime("%Y-%m-%d"),
            pdf_url=paper.pdf_url,
            summary=paper.summary
        ))
    return candidates

def download_and_extract_pdf(pdf_url: str, output_dir: str = "./downloads") -> Tuple[str, str, bool]:
    """
    Downloads PDF directly from arXiv URL and extracts text using PyMuPDF.
    Returns: (file_path, extracted_text, is_fallback)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract clean file ID from URL (e.g. '1706.03762v7' -> '1706.03762v7.pdf')
    raw_name = pdf_url.split("/")[-1].replace(".pdf", "")
    file_name = f"{raw_name}.pdf"
    pdf_path = os.path.join(output_dir, file_name)

    # Direct download using standard urllib with custom user-agent
    if not os.path.exists(pdf_path):
        req = urllib.request.Request(
            pdf_url, 
            headers={'User-Agent': 'ArxivDigestAgent/1.0 (academic-researcher)'}
        )
        with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
            out_file.write(response.read())

    # Extract text using PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    full_text = [page.get_text() for page in doc]
    extracted = "\n".join(full_text).strip()
    
    # Fallback requirement: If extraction has < 500 characters (scanned/broken PDF)
    if len(extracted) < 500:
        return pdf_path, "", True
        
    return pdf_path, extracted, False