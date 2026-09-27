# Autonomous arXiv Paper Digest & QA Agent

An autonomous research paper analysis agent built with **LangGraph**, **PyMuPDF**, **ChromaDB**, and **Groq (openai/gpt-oss-120b)**. The agent accepts either a specific arXiv ID/URL or a natural language research topic, retrieves and parses the paper, synthesizes a structured executive briefing, and enables grounded follow-up QA through an in-memory RAG pipeline.

---

## 1. System Architecture & State Graph

The system is designed as an explicit, stateful graph using **LangGraph**, carrying an immutable state dictionary across nodes with conditional routing.

              [User Input]
                    │
                    ▼
         [understand_query_node]
                    │
     ┌──────────────┴──────────────┐
(is paper_id)                  (is topic)
│                             │
▼                             ▼
[fetch_direct_paper]          [search_and_rank]
(arXiv Atom API)             (arXiv API + LLM)
│                             │
└──────────────┬──────────────┘
               │
               ▼
      [parse_and_chunk_node]
    (PyMuPDF / Abstract Fallback / Chroma)
                │
                ▼
         [summarize_node]
    (Structured Pydantic Briefing)
                │
                ▼
        [Interactive QA Loop]
 (Grounded RAG with Anti-Hallucination)


 ### Shared State Schema (`state.py`)
```python
class AgentState(TypedDict):
    user_input: str
    query_type: str                   # "paper_id" | "topic"
    arxiv_id: Optional[str]
    paper_metadata: Optional[PaperMetadata]
    candidate_papers: List[PaperMetadata]
    pdf_path: Optional[str]
    extracted_text: str
    parse_fallback: bool             # Flagged True if PDF unparseable
    vector_store: Optional[Any]      # In-memory ChromaDB instance
    briefing: Optional[Dict[str, Any]]
    qa_history: List[Dict[str, str]]
    error: Optional[str]

2. Setup & Run Instructions
This project runs completely with free-tier and local open-source tooling. No paid APIs are required.

Prerequisites
Python 3.10+

A free Groq API key (available at https://console.groq.com)

Installation
Clone the repository and navigate into the directory:

Bash
git clone [https://github.com/SomyaSagarika/arxiv-digest-agent.git](https://github.com/SomyaSagarika/arxiv-digest-agent.git)
cd arxiv-digest-agent
Create and activate a virtual environment:

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install required dependencies:

Bash
pip install -r requirements.txt
Configure environment variables:
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=my_groq_api_key
Running the Agent
Run the interactive CLI:

Bash
python main.py
You can pass either:

A specific arXiv ID or URL: 1706.03762 or https://arxiv.org/abs/1706.03762

A natural language topic: recent work on KV-cache compression for LLMs

3. Example Execution & Sample QA Exchanges
Run: arXiv ID Lookup (1706.03762 - "Attention Is All You Need")
Plaintext
================================================================================
📑 EXECUTIVE BRIEFING: Attention Is All You Need
================================================================================
• Authors:     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.
• arXiv ID:    1706.03762v7 | Date: 2017-06-12
• Link:        [http://arxiv.org/pdf/1706.03762v7](http://arxiv.org/pdf/1706.03762v7)

💡 WHY THIS MATTERS (Summary):
Proposes the Transformer architecture, dispensing with recurrence and convolutions entirely in favor of self-attention mechanisms. This foundational architecture underpins modern large language models.

🎯 PROBLEM STATEMENT:
Sequential computation in recurrent neural networks (RNNs, LSTMs) creates fundamental training bottlenecks by preventing parallelization across sequence lengths.

🛠️ METHOD / APPROACH:
  - Multi-Head Attention mechanisms replacing recurrent units
  - Sinusoidal Positional Encoding for sequence awareness
  - Scaled Dot-Product Attention layers with residual connections

📊 KEY RESULTS / CLAIMS:
  - Achieved 28.4 BLEU on WMT 2014 English-to-German translation task
  - Trained in 3.5 days on 8 P100 GPUs, significantly faster than existing architectures

⚠️ EXPLICIT LIMITATIONS:
  - Quadratic computational complexity O(n^2) with respect to input sequence length
  - Inherent lack of inductive bias regarding sequence locality compared to CNNs

❓ SUGGESTED FOLLOW-UP QUESTIONS:
  ? How does multi-head attention scale compute vs. single-head attention?
  ? How does inference caching compare to recurrent hidden state updates?
================================================================================

💬 Grounded QA Interaction:
User  >> What warm-up steps parameter was used in the Adam optimizer formula?
Agent >> The warm-up steps parameter was set to **4000**.

User  >> How many heads were used in the multi-head attention mechanism for the base model?
Agent >> The paper states that the model uses **8 parallel attention layers (heads)** in its multi-head attention mechanism.

User  >> Does the paper mention using FlashAttention or KV-cache optimization?
Agent >> The paper does not provide enough evidence or mention this detail.

4. Design Decisions & Tradeoffs
1. Orchestration: State Graph over Monolithic Prompt Chains
Decision: Implemented LangGraph with an explicit AgentState schema rather than a standard linear chain.

Tradeoff: While a single prompt chain requires less setup, an explicit graph separates query classification from retrieval, enables conditional edges (direct ID lookup vs. topic re-ranking), and facilitates independent state persistence into the QA loop without re-running data extraction.

2. Ingestion & Graceful Failure Handling
Decision: Utilized PyMuPDF (fitz) for PDF parsing and built an automated abstract-fallback mechanism.

Tradeoff: While PyMuPDF does not run OCR on complex figures or scanned images, it parses clean digital research PDFs in milliseconds. If extracted characters fall below 500 (indicating a scanned, corrupt, or layout-broken PDF), the agent automatically falls back to the arXiv metadata abstract and notifies the user, ensuring system resilience.

3. Vector Database & Embedding Selection
Decision: Used an ephemeral in-memory ChromaDB instance (chromadb.EphemeralClient()) paired with local sentence-transformers (all-MiniLM-L6-v2).

Tradeoff: Storing vectors in ephemeral memory avoids managing persistent disk storage or external Docker services, keeping local reproduction frictionless. Embeddings execute locally on CPU with zero API costs and no external rate-limit bottlenecks.

4. Hallucination Mitigation in QA
Decision: Strict grounding instructions with negative constraints ("If the detail is not verifiable from the context, state that the paper does not provide enough evidence").

Tradeoff: The model avoids speculative extrapolation. While this prevents speculative creative answers, it ensures research-grade factual accuracy.

Known Limitations & Future Improvements
Table & Formula Parsing: Mathematical formulas and tabular data can lose structural integrity in raw text extraction; integrating vision-language document models (e.g., Nougat or MinerU) would improve equation extraction.

Multi-Paper Synthesis: The current scope ranks and focuses on the single most relevant paper for a topic query. Supporting multi-document comparative synthesis across 3–5 papers would be a natural next extension.

### 🛡️ Edge Cases & System Robustness (§5 Considerations)

Zero or Many arXiv Candidates: If a broad topic returns multiple candidates, rank_and_select_paper scores abstract relevancy to identify and select the top paper. If zero papers match, conditional graph edges trigger a clean recovery node prompting query refinement rather than failing.

PDF Parsing Failures: Text is extracted via PyMuPDF. If extracted text length falls below minimal thresholds (e.g., image-only scanned PDFs), error flags route execution to a fallback parser node rather than throwing unhandled runtime exceptions.

Grounding & Anti-Hallucination: QA answers require top-k vector context retrieval from Chroma DB. System prompts explicitly restrict LLM generation strictly to retrieved context.

State Passing & Persistence: Active session state is maintained in-memory using LangGraph's explicit state schema, with embeddings stored locally in Chroma DB.

---

### Conclusion

What Worked Well: LangGraph made node debugging and conditional routing transparent and modular. Local Chroma DB storage ensured zero network overhead and completely free execution without relying on cloud vector databases.

Known Limitations: Standard PyMuPDF layout parsing can occasionally lose section header hierarchies in complex or multi-column PDFs.

Future Work:
1. Integrate OCR-based parsers (Unstructured or Marker) for multi-column and image-heavy layouts.
2. Implement multi-document comparative synthesis across topic search results.
3. Add persistent SQLite state check-pointing across terminal sessions.
4. Build an interactive Streamlit web interface.