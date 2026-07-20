from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field
from app.services.storage_service import save_upload, save_text
from app.models import DocumentOut
from app.routes.auth import get_current_user

router = APIRouter()


class TextDocumentIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


@router.post("/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    # save_upload raises ValueError for every validation reject (extension not allowed,
    # content/extension mismatch, size limit). Map to 400 so the caller sees the reason
    # instead of the generic 500 these used to fall through to.
    try:
        doc = await save_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return doc


@router.post("/text", response_model=DocumentOut)
async def upload_text(body: TextDocumentIn, user: dict = Depends(get_current_user)):
    doc = await save_text(body.text)
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, user: dict = Depends(get_current_user)):
    raise HTTPException(status_code=404, detail="Document not found")
