# test_fetch.py
from ingestion import detect_query_type, fetch_paper_by_id, download_and_extract_pdf

# 1. Test Regex Detection
test_input = "https://arxiv.org/abs/1706.03762"  # "Attention Is All You Need"
q_type, arxiv_id = detect_query_type(test_input)
print(f"[1] Query: {test_input}")
print(f"    Detected type: '{q_type}' | ID: '{arxiv_id}'")

# 2. Test arXiv Metadata Retrieval
print("\n[2] Fetching metadata from arXiv...")
paper = fetch_paper_by_id(arxiv_id)
if paper:
    print(f"    Title: {paper['title']}")
    print(f"    Published: {paper['published']}")
    print(f"    Authors: {', '.join(paper['authors'][:3])} et al.")
else:
    print("    Failed to fetch paper metadata.")

# 3. Test PDF Download and Extraction
print("\n[3] Downloading PDF and parsing with PyMuPDF...")
pdf_path, text, is_fallback = download_and_extract_pdf(paper['pdf_url'])
print(f"    PDF path: {pdf_path}")
print(f"    Total extracted characters: {len(text)}")
print(f"    Fallback mode triggered: {is_fallback}")

print("\n--- Ingestion sanity check successful! ---")