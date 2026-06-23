# Replace "5 free scans (≤800 words)" with "5 free credits at signup"

**Date:** 2026-06-23
**Status:** Approved (design)

## Goal

Retire the per-scan free exemption and instead grant every **new** user a one-time
balance of **5 credits** at account creation. Billing becomes uniform: every scan
costs credits at the normal rate; the 5 welcome credits are fungible (scans *or*
rewrites).

## Decisions (locked)

1. **Who gets the grant:** new signups only. Existing users get no retroactive top-up.
2. **Credit scope:** fungible — spendable on scans (1 credit / 1,000 words) or
   rewrites (5 credits / 1,000 words).
3. **Existing users — hard cutover:** the free-scan billing path is removed for
   everyone. An existing user keeps their purchased balance but no longer gets free
   short scans. Accepted downside: a 0-balance existing user can no longer scan free.

## Current state (traced)

- New `CreditAccount` is created with `balance_tokens=0` at
  `draftproof-api/app/routes/auth.py:156-158` (no welcome grant today).
- Free model: first 5 scans ≤800 words are free, tracked by the durable
  `users.free_scans_used` counter via an atomic CAS in
  `draftproof-api/app/services/scan_service.py:230-247`.
- `FREE_SCAN_WORD_LIMIT = 800`, `FREE_SCAN_LIMIT = 5` (scan_service.py:17-18).
- Paid rate: `_paid_scan_cost` = 1 credit per started 1,000 words (min 1).
- Rewrite rate: `_rewrite_cost` = 5 credits per started 1,000 words (min 5).
- Ledger convention: `CreditLedger(entry_type, token_delta, balance_after,
  reference_type, reference_id, idempotency_key, note)` — see payments.py:202.

## Changes

### Backend (no schema migration)

1. **`config.py`** — add `WELCOME_CREDITS = 5` (named constant).
2. **`auth.py`** — on truly-new account creation, set `balance_tokens =
   WELCOME_CREDITS`, `await db.flush()` to populate `account.id` (the PK default
   fires at flush, not instantiation), then write a `CreditLedger` row:
   `entry_type="admin_adjustment"`, `token_delta=WELCOME_CREDITS`,
   `balance_after=WELCOME_CREDITS`, `reference_type="user"`,
   `reference_id=user.id`, `idempotency_key=f"welcome_{user.id}"`,
   `note="Welcome bonus: 5 free credits"`. The idempotency key guarantees a single
   grant per user.
   - **entry_type choice:** the `credit_ledger` CHECK (migration 001) only permits
     `purchase, scan_debit, rewrite_debit, reservation, reservation_release,
     refund, admin_adjustment, expiry`. `welcome_grant` is not allowed, and adding
     it needs a manual SQL migration that must land *before* the auto-deploy or
     every new signup 500s. We reuse `admin_adjustment` (the non-purchase
     system-credit bucket); the `welcome_` idempotency prefix + note keep grants
     filterable. The `/history` view reads the `payments` table, not the ledger,
     so this entry_type is internal accounting and never renders in the UI.
3. **`scan_service.py`** — bill every scan at `_paid_scan_cost`. Remove the
   `free_scans_used` CAS consumption block, the `_scan_cost` free branch, and the
   special `always_paid` divergence (every scan is now paid, so the parameter is a
   no-op kept only for call-site compatibility). `get_free_scan_usage` is removed /
   the route stops exposing it. DB columns `free_scans_used` and `free_scan_counted`
   are left **dormant** (no destructive migration). `_refund_free_scan` is retained
   as a safe no-op so any free job in flight across the deploy still settles.

### Frontend (copy + scan-page widget)

4. Replace the "free scan" affordance with **credit balance + per-scan cost**.
   Files: `utils/scanBilling.js`, `i18n/{en,zh}/scan.js`
   (freeThrough, pricingNote, freeUsage, freeRemaining, freeLimitTitle,
   freeLimitMessage, freeScan), and the 800-word/free-scan mentions in
   `i18n/{en,zh}/pricing.js`, `i18n/{en,zh}/dashboard.js`,
   `i18n/{en,zh}/faqPage.js`, `pages/Landing.jsx`, `pages/Report.jsx`,
   `pages/Rewrite.jsx`. en + zh both.

## Testing

- `_paid_scan_cost` unchanged; assert short scan = 1 credit.
- New-account creation grants 5 credits + one ledger row; idempotent on re-run.
- Update/remove existing tests that assert the free-scan-counter behavior.
- Run `cd draftproof-api && pytest`.

## Out of scope

- No schema migration / column drops.
- No retroactive top-up for existing users.
- Pricing-pack amounts and rewrite rate unchanged.
