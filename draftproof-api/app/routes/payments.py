import asyncio
import logging

import stripe
from fastapi import APIRouter, Request, HTTPException, Depends

log = logging.getLogger(__name__)
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import (
    STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET,
    TOKEN_CURRENCY_CODE, TOKEN_STRIPE_CURRENCY, TOKEN_PACKS, get_token_pack_price_sgd,
    FRONTEND_URL, SECRET_KEY, JWT_ALGORITHM,
)
from app.models.db import get_db, User, CreditAccount, CreditLedger, Payment
from jose import jwt, JWTError

router = APIRouter()
stripe.api_key = STRIPE_SECRET_KEY
stripe.api_version = "2024-12-18.acacia"
stripe.max_network_retries = 2


class CheckoutRequest(BaseModel):
    pack_id: str


def _get_user_id(request: Request) -> str:
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/packs")
async def get_packs():
    return [
        {
            "id": key,
            "name": pack["name"],
            "tokens": pack["tokens"],
            "price_sgd": get_token_pack_price_sgd(pack),
        }
        for key, pack in TOKEN_PACKS.items()
    ]


@router.post("/checkout")
async def create_checkout(body: CheckoutRequest, request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    pack_id = body.pack_id

    if pack_id not in TOKEN_PACKS:
        raise HTTPException(status_code=400, detail="Invalid pack")

    pack = TOKEN_PACKS[pack_id]
    price_sgd = get_token_pack_price_sgd(pack)
    price_cents = int(price_sgd * 100)
    log.info("Checkout: pack=%s, price=%d cents, currency=%s", pack_id, price_cents, TOKEN_STRIPE_CURRENCY)

    try:
        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": TOKEN_STRIPE_CURRENCY,
                    "product_data": {"name": f"DraftProof — {pack['name']} ({pack['tokens']} tokens)"},
                    "unit_amount": price_cents,
                },
                "quantity": 1,
            }],
            metadata={"user_id": user_id, "pack_id": pack_id, "tokens": str(pack["tokens"])},
            success_url=f"{FRONTEND_URL}/buy?success=1",
            cancel_url=f"{FRONTEND_URL}/buy?canceled=1",
        )
    except stripe.error.StripeError as e:
        log.error("Stripe checkout failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Stripe error: {e.user_message or str(e)}")

    return {"url": session.url}


@router.get("/balance")
async def get_balance(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    result = await db.execute(
        select(CreditAccount).where(CreditAccount.user_id == user_id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="No credit account")
    return {"balance": account.balance_tokens, "reserved": account.reserved_tokens}


@router.get("/history")
async def get_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    per_page: int = 10,
):
    user_id = _get_user_id(request)
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    offset = (page - 1) * per_page

    from sqlalchemy import func
    count_result = await db.execute(
        select(func.count()).select_from(Payment)
        .where(Payment.user_id == user_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Payment)
        .where(Payment.user_id == user_id)
        .order_by(Payment.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    payments = result.scalars().all()
    return {
        "items": [
            {
                "id": str(p.id),
                "tokens": p.tokens_purchased,
                "amount_cents": p.amount_cents,
                "currency": p.currency,
                "status": p.status,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in payments
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = await asyncio.to_thread(
            stripe.Webhook.construct_event, payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session_data = event["data"]["object"]
        user_id = session_data["metadata"]["user_id"]
        tokens = int(session_data["metadata"]["tokens"])
        stripe_session_id = session_data["id"]
        amount_cents = session_data["amount_total"]
        currency = str(session_data.get("currency") or TOKEN_CURRENCY_CODE).upper()

        # Check idempotency — skip if already processed
        result = await db.execute(
            select(Payment).where(Payment.provider_payment_id == stripe_session_id)
        )
        if result.scalar_one_or_none():
            return {"ok": True, "skipped": True}

        # Credit tokens with row-level lock to prevent race on concurrent webhooks
        result = await db.execute(
            select(CreditAccount).where(CreditAccount.user_id == user_id).with_for_update()
        )
        account = result.scalar_one_or_none()
        if not account:
            raise HTTPException(status_code=404, detail="Credit account not found")

        account.balance_tokens += tokens
        balance_after = account.balance_tokens

        # Record payment
        payment = Payment(
            user_id=user_id,
            provider="stripe",
            provider_payment_id=stripe_session_id,
            amount_cents=amount_cents,
            currency=currency,
            tokens_purchased=tokens,
            status="paid",
            idempotency_key=stripe_session_id,
        )
        db.add(payment)
        await db.flush()  # populate payment.id

        # Record ledger entry
        ledger = CreditLedger(
            credit_account_id=account.id,
            user_id=user_id,
            entry_type="purchase",
            token_delta=tokens,
            balance_after=balance_after,
            reference_type="payment",
            reference_id=payment.id,
            idempotency_key=f"ledger_{stripe_session_id}",
            note=f"Purchased {tokens} tokens via Stripe",
        )
        db.add(ledger)
        await db.commit()

    return {"ok": True}
