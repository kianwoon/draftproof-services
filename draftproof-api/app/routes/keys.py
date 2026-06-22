"""API-key management — web-facing (cookie/JWT auth) CRUD for personal keys.

Mounted at /api/keys. The clear-text key is returned exactly once, on create.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.routes.auth import get_current_user
from app.services import api_key_service

router = APIRouter()


class CreateKeyRequest(BaseModel):
    name: str | None = None


@router.post("/", status_code=201)
async def create_key_route(req: CreateKeyRequest, user: dict = Depends(get_current_user)):
    """Mint a key. Response includes the plaintext `key` — shown only here."""
    try:
        return await api_key_service.create_key(user["id"], name=req.name)
    except ValueError as exc:
        # e.g. per-user active-key cap reached.
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/")
async def list_keys_route(user: dict = Depends(get_current_user)):
    return await api_key_service.list_keys(user["id"])


@router.delete("/{key_id}")
async def revoke_key_route(key_id: str, user: dict = Depends(get_current_user)):
    revoked = await api_key_service.revoke_key(user["id"], key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"detail": "Revoked"}
