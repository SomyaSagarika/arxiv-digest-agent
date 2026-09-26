# models.py
from pydantic import BaseModel, Field
from typing import List

class ExecutiveBriefing(BaseModel):
    title: str = Field(description="Exact title of the research paper")
    authors: List[str] = Field(description="List of primary authors")
    arxiv_id: str = Field(description="arXiv ID of the paper")
    publish_date: str = Field(description="Publication or submission date")
    paper_link: str = Field(description="Direct URL to the paper or arXiv abstract")
    one_paragraph_summary: str = Field(
        description="A plain-English summary explaining why this paper matters"
    )
    problem_statement: str = Field(
        description="The core bottleneck, challenge, or question the authors are addressing"
    )
    method_approach: List[str] = Field(
        description="Key methodology, architecture, or algorithmic contributions in bullet points"
    )
    key_results: List[str] = Field(
        description="Main empirical findings, performance benchmarks, or validated claims"
    )
    limitations: List[str] = Field(
        description="Explicit limitations, trade-offs, compute constraints, or unaddressed failure modes"
    )
    suggested_followups: List[str] = Field(
        description="3-4 insightful questions a reader or engineer might ask next"
    )