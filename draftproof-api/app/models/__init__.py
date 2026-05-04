from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class DocumentOut(BaseModel):
    id: str
    filename: str
    created_at: datetime


class ScanRequest(BaseModel):
    document_id: str
    text: Optional[str] = None


class ScanOut(BaseModel):
    id: str
    document_id: str
    status: str
    report_id: Optional[str] = None
    tier: Optional[str] = None
    finding_count: Optional[int] = None
    progress_percent: int = 0
    progress_message: Optional[str] = None


class IssueOut(BaseModel):
    id: str
    severity: str
    title: Optional[str] = None
    description: str
    location: Optional[str] = None
    sentence_text: Optional[str] = None
    scanner: Optional[str] = None
    category: Optional[str] = None
    signal_category: Optional[str] = None
    score: Optional[float] = None
    top10_ratio: Optional[float] = None
    raw_risk: Optional[str] = None
    adjusted_risk: Optional[str] = None
    actionability: Optional[str] = None
    evidence: Optional[Any] = None
    recommendation: Optional[str] = None
    adjustment: Optional[Any] = None


class ReportOut(BaseModel):
    id: str
    document_name: str
    issues: list[IssueOut]
    created_at: datetime
    tier: Optional[str] = None
    ai_score: Optional[float] = None
    writing_score: Optional[float] = None
    ai_risk_badge: Optional[Any] = None
    report_md_url: Optional[str] = None
    report_pdf_url: Optional[str] = None
    results_json: Optional[Any] = None
    rewrite: Optional[Any] = None


class SuggestionOut(BaseModel):
    id: str
    text: str


class ApplySuggestionRequest(BaseModel):
    suggestion_id: str


class RewriteCreateRequest(BaseModel):
    scan_id: str


class RewriteOut(BaseModel):
    id: str
    scan_id: str
    status: str
    error: Optional[str] = None
    progress_percent: int = 0
    progress_message: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RewriteReportOut(BaseModel):
    summary: Optional[Any] = None
    sentence_comparison: Optional[list] = None
    ai_findings: Optional[list] = None
