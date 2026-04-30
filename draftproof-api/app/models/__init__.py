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


class IssueOut(BaseModel):
    id: str
    severity: str
    title: Optional[str] = None
    description: str
    location: Optional[str] = None
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
    report_md_url: Optional[str] = None
    report_pdf_url: Optional[str] = None
    results_json: Optional[Any] = None


class SuggestionOut(BaseModel):
    id: str
    text: str


class ApplySuggestionRequest(BaseModel):
    suggestion_id: str
