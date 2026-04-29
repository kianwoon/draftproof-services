from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentOut(BaseModel):
    id: str
    filename: str
    created_at: datetime


class ScanRequest(BaseModel):
    document_id: str


class ScanOut(BaseModel):
    id: str
    document_id: str
    status: str
    report_id: Optional[str] = None


class IssueOut(BaseModel):
    id: str
    severity: str
    description: str
    location: Optional[str] = None


class ReportOut(BaseModel):
    id: str
    document_name: str
    issues: list[IssueOut]
    created_at: datetime


class SuggestionOut(BaseModel):
    id: str
    text: str


class ApplySuggestionRequest(BaseModel):
    suggestion_id: str
