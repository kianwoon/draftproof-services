-- Backstop for the attempt cap enforced in app/routes/defence.py (independent-review
-- finding, cost-cap follow-up to 014): count_attempts() + a non-atomic check-then-insert
-- has a narrow TOCTOU window where two near-simultaneous submissions for the same
-- (scan_id, question_index) could both pass the application-level cap check before either
-- commits. This unique constraint makes that impossible at the DB level — the second
-- concurrent insert fails with a unique-violation instead of silently exceeding the cap;
-- app/services/defence_service.py::create_response() catches that violation and
-- app/routes/defence.py translates it into the same 409 the pre-check path already returns.

CREATE UNIQUE INDEX IF NOT EXISTS uq_defence_responses_scan_question_attempt
    ON defence_responses(scan_id, question_index, attempt);
