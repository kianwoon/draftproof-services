import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from authlib.integrations.starlette_client import OAuth
from jose import jwt
import boto3
from botocore.config import Config as BotoConfig

from app.config import (
    SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS,
    ALLOWED_EMAIL_DOMAINS, FRONTEND_URL,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT,
    R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME,
)
from app.models.db import get_db, User, UserIdentity, CreditAccount

router = APIRouter()

oauth = OAuth()

# Google OAuth
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Microsoft OAuth
oauth.register(
    name="microsoft",
    client_id=MICROSOFT_CLIENT_ID,
    client_secret=MICROSOFT_CLIENT_SECRET,
    server_metadata_url=f"https://login.microsoftonline.com/{MICROSOFT_TENANT}/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile User.Read"},
)


def _build_callback_url(request: Request, callback_name: str) -> str:
    url = str(request.url_for(callback_name))
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


def _validate_email_domain(email: str) -> str:
    domain = email.split("@")[-1].lower()
    if domain not in ALLOWED_EMAIL_DOMAINS:
        raise HTTPException(
            status_code=403,
            detail=f"Email domain '{domain}' is not allowed. Please use Gmail or Hotmail.",
        )
    return email


def _create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


async def _fetch_microsoft_avatar(access_token: str, user_id: str) -> str | None:
    """Fetch profile photo from Microsoft Graph, upload to R2, return URL."""
    import httpx
    log = logging.getLogger("auth.microsoft.avatar")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://graph.microsoft.com/v1.0/me/photo/$value",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                log.info("No photo available (status %d)", resp.status_code)
                return None
            photo_bytes = resp.content
            content_type = resp.headers.get("content-type", "image/jpeg")

        s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=BotoConfig(signature_version="s3v4"),
        )
        key = f"avatars/{user_id}.jpg"
        s3.put_object(
            Bucket=R2_BUCKET_NAME, Key=key, Body=photo_bytes, ContentType=content_type
        )
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": R2_BUCKET_NAME, "Key": key}, ExpiresIn=86400 * 30
        )
        log.info("Avatar uploaded for user %s", user_id)
        return url
    except Exception as e:
        log.warning("Failed to fetch Microsoft avatar: %s", e)
        return None


async def _upsert_user(db: AsyncSession, provider: str, user_info: dict) -> User:
    email = user_info["email"].lower().strip()
    email_normalized = email
    provider_user_id = str(user_info.get("sub", user_info.get("id", "")))
    provider_email_verified = user_info.get("email_verified", True)

    # Find existing identity by provider + provider_user_id
    result = await db.execute(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == provider_user_id,
        )
    )
    identity = result.scalar_one_or_none()

    if identity:
        # Existing user — update
        identity.last_login_at = datetime.utcnow()
        identity.provider_email = email
        identity.provider_email_verified = provider_email_verified

        result = await db.execute(select(User).where(User.id == identity.user_id))
        user = result.scalar_one()
        user.display_name = user_info.get("name", user.display_name)
        user.avatar_url = user_info.get("picture", user.avatar_url)
        user.updated_at = datetime.utcnow()
    else:
        # Check if user exists by email (case-insensitive to handle prior casing differences)
        result = await db.execute(
            select(User).where(
                func.lower(User.email_normalized) == email_normalized.lower()
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            # Create new user (handle race: another request may have inserted)
            user = User(
                email=email,
                email_normalized=email_normalized,
                display_name=user_info.get("name", ""),
                avatar_url=user_info.get("picture"),
                status="active",
            )
            db.add(user)
            try:
                await db.flush()
                # Only create credit account for truly new users
                account = CreditAccount(user_id=user.id)
                db.add(account)
            except IntegrityError:
                # Race condition: user was created by a concurrent request or prior sign-in
                await db.rollback()
                result = await db.execute(
                    select(User).where(
                        func.lower(User.email_normalized) == email_normalized.lower()
                    )
                )
                user = result.scalar_one()

        # Create identity
        identity = UserIdentity(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=email,
            provider_email_verified=provider_email_verified,
            last_login_at=datetime.utcnow(),
        )
        db.add(identity)

    await db.commit()
    await db.refresh(user)
    return user


@router.get("/google")
async def auth_google(request: Request):
    redirect_uri = _build_callback_url(request, "auth_google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def auth_google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    import logging
    log = logging.getLogger("auth.google")
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        if not user_info:
            user_info = await oauth.google.userinfo(token=token)

        email = user_info.get("email", "MISSING")
        log.info("Google callback success — email=%s sub=%s", email, user_info.get("sub"))

        _validate_email_domain(user_info["email"])
        user = await _upsert_user(db, "google", user_info)
        jwt_token = _create_jwt(user.id, user.email)

        log.info("User upserted — user_id=%s, redirecting to %s/auth/callback", user.id, FRONTEND_URL)

        response = RedirectResponse(url=f"{FRONTEND_URL}/auth/callback")
        response.set_cookie(
            "token", jwt_token,
            httponly=True, secure=True, samesite="lax",
            max_age=JWT_EXPIRATION_HOURS * 3600,
            path="/",
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        log.error("Google OAuth callback failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Google sign-in failed")


@router.get("/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = _build_callback_url(request, "auth_microsoft_callback")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri)


@router.get("/microsoft/callback")
async def auth_microsoft_callback(request: Request, db: AsyncSession = Depends(get_db)):
    import logging
    logger = logging.getLogger("auth.microsoft")
    try:
        token = await oauth.microsoft.authorize_access_token(
            request, claims_options={"iss": {"essential": False}}
        )
        logger.info("Microsoft token keys: %s", list(token.keys()))
        user_info = token.get("userinfo")
        if not user_info:
            # Microsoft returns user info in the ID token claims, not as a top-level key
            id_token = token.get("id_token")
            if id_token and isinstance(id_token, dict):
                user_info = id_token
            else:
                user_info = await oauth.microsoft.userinfo(token=token)
            logger.info("Userinfo keys: %s", list(user_info.keys()) if user_info else "None")

        email = user_info.get("email") or user_info.get("preferred_username") or user_info.get("upn")
        if not email:
            raise HTTPException(status_code=400, detail="Could not retrieve email from Microsoft account")

        user_info["email"] = email
        _validate_email_domain(email)

        # Microsoft doesn't include photo in userinfo — fetch from Graph API
        ms_access_token = token.get("access_token")
        if ms_access_token and not user_info.get("picture"):
            avatar_url = await _fetch_microsoft_avatar(ms_access_token, user_info.get("sub", ""))
            if avatar_url:
                user_info["picture"] = avatar_url

        user = await _upsert_user(db, "microsoft", user_info)
        jwt_token = _create_jwt(user.id, user.email)

        response = RedirectResponse(url=f"{FRONTEND_URL}/auth/callback")
        response.set_cookie(
            "token", jwt_token,
            httponly=True, secure=True, samesite="lax",
            max_age=JWT_EXPIRATION_HOURS * 3600,
            path="/",
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("auth.microsoft").error("Microsoft OAuth callback failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Microsoft sign-in failed")


async def get_current_user(request: Request) -> dict:
    """Dependency: extracts user_id from JWT cookie."""
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"id": payload["sub"], "email": payload["email"]}
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/me")
async def get_me(request: Request, db: AsyncSession = Depends(get_db)):
    import logging
    log = logging.getLogger("auth.me")
    token = request.cookies.get("token")
    if not token:
        log.info("/me — no token cookie found. Cookies: %s", list(request.cookies.keys()))
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload["sub"]
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        avatar_url = user.avatar_url if user else None
        log.info("/me — resolved user_id=%s email=%s avatar=%s", user_id, payload["email"], bool(avatar_url))
        return {"id": user_id, "email": payload["email"], "avatar_url": avatar_url}
    except jwt.JWTError:
        log.warning("/me — JWT decode failed")
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/logout")
async def logout():
    from starlette.responses import JSONResponse
    response = JSONResponse({"ok": True})
    # Explicitly expire the cookie by setting max_age=0 with empty value
    response.set_cookie(
        "token", "",
        httponly=True, secure=True, samesite="lax",
        max_age=0, path="/",
    )
    # Also try delete_cookie for good measure
    response.delete_cookie("token", path="/")
    return response
