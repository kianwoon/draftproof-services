from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.storage_service import save_upload
from app.models import DocumentOut

router = APIRouter()


@router.post("/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)):
    doc = await save_upload(file)
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str):
    raise HTTPException(status_code=404, detail="Document not found")
