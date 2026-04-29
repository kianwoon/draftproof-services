from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from app.services.storage_service import save_upload, save_text
from app.models import DocumentOut

router = APIRouter()


class TextDocumentIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


@router.post("/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)):
    doc = await save_upload(file)
    return doc


@router.post("/text", response_model=DocumentOut)
async def upload_text(body: TextDocumentIn):
    doc = await save_text(body.text)
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str):
    raise HTTPException(status_code=404, detail="Document not found")
