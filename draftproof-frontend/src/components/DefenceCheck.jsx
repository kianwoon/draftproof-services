// Defence-readiness check (Task 8) — student answers a flagged reflective question in
// their own words; an LLM judge (poc/detect/defence_judge.py, via the worker's
// judge_defence_answer Celery task) reads the answer against the anchor quote and returns
// a per-axis High/Medium/Low read + one-sentence rationale. Advisory/practice only — never
// changes the scan's score or tier.
//
// Data flow: `questions` is the SAME array Report.jsx already derives at
// `badge.critical_thinking_control.questions` for the read-only CriticalThinkingControl
// panel (see pages/report/CriticalThinkingControl.jsx) — passed down as a prop rather than
// re-read here, so there is one source of truth for "what questions exist on this report".
//
// Renders null in two cases (no broken/empty shell either way):
//   1. GET /api/scans/{scanId}/defence 404s — DRAFTPROOF_DEFENCE_CHECK is off server-side.
//   2. `questions` is empty/absent — nothing to answer.
//
// API contract (confirmed from draftproof-api/app/routes/defence.py, current as of Task 6's
// review — the plan's original description drifted slightly during implementation):
//   POST /api/scans/{scanId}/defence/answers  body {question_index, answer_text}
//     -> 202 {response_id, status: "pending"}
//     -> 404 flag off / scan not found or not owned
//     -> 409 attempt-cap exceeded (NOT 429 — corrected in Task 6 review; this is a
//        permanent per-question count cap, not a time-windowed rate limit)
//     -> 400 Pydantic validation failure (answer-length max_length, question_index shape, ...)
//        OR a plain "Invalid question_index" business-rule 400 — draftproof-api/app/main.py's
//        RequestValidationError handler returns 400 (not FastAPI's default 422) with
//        {detail: "Invalid request.", errors: [{loc, msg, type}, ...]} for every Pydantic
//        failure, so 400 is NOT specific to "answer too long" — see extractErrorMessage below,
//        which only shows a length-specific message when the `errors` array actually pins the
//        failure to `answer_text` with type "string_too_long"; every other 400 shows the
//        server's own detail/msg text rather than a guessed label (final-review Finding 1: an
//        earlier version of this file special-cased status 422, which the backend never sends).
//   GET /api/scans/{scanId}/defence
//     -> 200 {responses: [...], readiness: {dimension: {level, score, question_index, attempt}}}
//     -> 404 flag off
// Row shape (defence_service.list_responses): {id, question_index, dimension, question,
// anchor_quote, answer_text, attempt, status, judgment, judge_model, created_at, judged_at}.
// status in {"pending","judged","failed"}. judgment (only when judged):
// {schema_version, model, generated_at, axes: {answer_understanding, semantic_alignment,
// reasoning_depth, source_awareness}: {level, score, rationale}, overall: {level, score,
// derived}, flags: [...]}. axis/overall `level` in {"high","medium","low"} — HIGH here means
// strong demonstrated understanding (a good outcome), the opposite direction from the
// document's AI-risk severity scale, so this module deliberately does not reuse the
// --sev-* (risk) CSS tokens — see the readiness-toned classes in 06-report-overview.css.
//
// allow-hardcode: student-facing copy strings (labels, button text, status messages) below
// are presentation text, not a scoring/matching oracle — mirrors ConsistencyRisk.jsx's
// precedent for a new advisory panel added without full i18n coverage. The one exception is
// `dimensionLabel()`, which reuses the EXISTING `report.criticalThinking.dimensions.*.label`
// i18n keys (already covering all 5 valid dimension codes) so dimension names stay
// consistent with the read-only panel above this one.
import { useCallback, useEffect, useRef, useState } from 'react';
import { getDefence, submitDefenceAnswer } from '../api/draftproofApi';

// Named constants (no magic numbers): how often / how long we poll GET /defence after a
// submit while the worker judges the answer.
const DEFENCE_POLL_INTERVAL_MS = 4000;
const DEFENCE_POLL_MAX_ATTEMPTS = 30; // ~2 minutes before we stop and tell the user to check back later

// MUST match draftproof-api/app/config.py's DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS default (2000).
// That env var is server-only (not exposed to the frontend build), so this is a mirrored literal,
// not a shared source of truth — it only bounds the <textarea> client-side to cut down on
// over-length submits before they round-trip to the server; the server's own Pydantic
// max_length is still the authoritative cap enforced in submitDefenceAnswer's 400 response
// (final-review Finding 1). If the backend value changes, update this constant to match.
const DEFENCE_MAX_ANSWER_CHARS = 2000;

// MUST match draftproof-api/app/config.py's DRAFTPROOF_DEFENCE_MAX_QUESTIONS default (3) —
// same mirrored-literal caveat as DEFENCE_MAX_ANSWER_CHARS above. `questions` (the prop) is the
// FULL critical_thinking_control.questions array shared with the read-only panel above this
// one, which has no cap of its own; `_load_questions` in routes/defence.py truncates to the
// same first-N slice server-side, so submitting for an index beyond this cap would 400 with
// "Invalid question_index" anyway — slicing here keeps the UI from ever showing a card that
// can't actually be submitted, and bounds worst-case LLM-judge cost per scan to
// DEFENCE_MAX_QUESTIONS * (DRAFTPROOF_DEFENCE_MAX_ATTEMPTS, currently 1) judge calls.
const DEFENCE_MAX_QUESTIONS = 3;

const LEVEL_LABELS = { high: 'High', medium: 'Medium', low: 'Low' };
// Readiness-direction tones (high = good/green), distinct from the document's risk-severity
// --sev-* tokens (high = bad there). Mirrors the FUSED_TIER_TONES pattern already used
// inline in pages/Report.jsx for a similar "good outcome = green" scale.
const LEVEL_TONE = {
  high: { color: '#15803d', bg: '#dcfce7' },
  medium: { color: '#b45309', bg: '#fef3c7' },
  low: { color: '#b91c1c', bg: '#fee2e2' },
};
const AXIS_LABELS = {
  answer_understanding: 'Understanding',
  semantic_alignment: 'Alignment with your draft',
  reasoning_depth: 'Reasoning depth',
  source_awareness: 'Source awareness',
};

// True when the RequestValidationError `errors` array (draftproof-api/app/main.py's
// validation_exception_handler shape: [{loc, msg, type}, ...]) pins the failure specifically to
// `answer_text` being too long — i.e. Pydantic's Field(max_length=...) on DefenceAnswerIn, NOT
// the sibling `question_index` constraint or any other 400. Checking BOTH `type` and `loc` (not
// status code alone) is what makes this reliable: status 400 is shared by every validation
// failure on this route (see the module docstring above), so status alone can't disambiguate.
function isAnswerTooLongError(err) {
  const errors = err?.response?.data?.errors;
  if (!Array.isArray(errors)) return false;
  return errors.some((row) => {
    if (!row || typeof row !== 'object') return false;
    const loc = Array.isArray(row.loc) ? row.loc : [];
    return row.type === 'string_too_long' && loc.includes('answer_text');
  });
}

function extractErrorMessage(err, fallback) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    // FastAPI/Pydantic validation-error shape: [{loc, msg, type}, ...]
    const first = detail.find((row) => row && typeof row.msg === 'string');
    if (first) return first.msg;
  }
  const errors = err?.response?.data?.errors;
  if (Array.isArray(errors) && errors.length) {
    const first = errors.find((row) => row && typeof row.msg === 'string');
    if (first) return first.msg;
  }
  return fallback;
}

// Latest attempt per question_index. list_responses orders by (question_index, attempt)
// ascending, but we don't rely on that ordering here — compare attempt numbers explicitly.
function latestRowsByQuestion(responses) {
  const map = {};
  (Array.isArray(responses) ? responses : []).forEach((row) => {
    if (!row || typeof row.question_index !== 'number') return;
    const existing = map[row.question_index];
    if (!existing || (row.attempt || 0) >= (existing.attempt || 0)) {
      map[row.question_index] = row;
    }
  });
  return map;
}

function dimensionLabel(dimension, t) {
  if (!dimension) return null;
  if (typeof t !== 'function') return dimension;
  const key = `report.criticalThinking.dimensions.${dimension}.label`;
  const translated = t(key);
  return translated === key ? dimension : translated;
}

export default function DefenceCheck({ scanId, questions, t }) {
  // 'checking' (initial GET in flight) | 'enabled' | 'disabled' (404 / no questions / any load error)
  const [flagState, setFlagState] = useState('checking');
  const [responsesByQuestion, setResponsesByQuestion] = useState({});
  const [drafts, setDrafts] = useState({});
  const [submitState, setSubmitState] = useState({});
  const pollTimersRef = useRef({});

  // Cap to the same first-N slice the backend enforces (_load_questions in routes/defence.py)
  // — see the DEFENCE_MAX_QUESTIONS constant above for why. Every index used below (drafts,
  // submitState, responsesByQuestion, handleSubmit's questionIndex) refers to this capped
  // array's own index, which matches the backend's question_index for entries 0..N-1 since
  // both slice the same source array the same way (Array.prototype.slice / Python list[:N]
  // both keep original order, and neither this component nor the backend ever reorders).
  const cappedQuestions = Array.isArray(questions) ? questions.slice(0, DEFENCE_MAX_QUESTIONS) : [];
  const hasQuestions = cappedQuestions.length > 0;

  const stopPolling = useCallback((questionIndex) => {
    const timer = pollTimersRef.current[questionIndex];
    if (timer) clearInterval(timer);
    delete pollTimersRef.current[questionIndex];
  }, []);

  const pollForResult = useCallback((questionIndex, responseId) => {
    stopPolling(questionIndex);
    let attempts = 0;
    const timer = setInterval(async () => {
      attempts += 1;
      try {
        const { data } = await getDefence(scanId);
        const rows = Array.isArray(data?.responses) ? data.responses : [];
        setResponsesByQuestion(latestRowsByQuestion(rows));
        const row = rows.find((r) => r.id === responseId);
        if (row && row.status !== 'pending') {
          stopPolling(questionIndex);
          setSubmitState((s) => ({ ...s, [questionIndex]: { status: 'idle' } }));
          return;
        }
      } catch {
        // Transient poll failure — keep trying until the attempt cap below.
      }
      if (attempts >= DEFENCE_POLL_MAX_ATTEMPTS) {
        stopPolling(questionIndex);
        setSubmitState((s) => ({ ...s, [questionIndex]: { status: 'idle', timedOut: true } }));
      }
    }, DEFENCE_POLL_INTERVAL_MS);
    pollTimersRef.current[questionIndex] = timer;
  }, [scanId, stopPolling]);

  useEffect(() => {
    // Report.jsx keeps this component mounted across a scanId prop change (SPA navigation
    // between reports re-uses the same route element, only `id` changes — see Report.jsx's
    // own `useEffect(() => { ...reset... }, [id, ...])`). Reset everything scoped to the
    // PREVIOUS scan: stale drafts/results from scan A must never bleed into scan B's
    // question cards just because they share the same question_index. This effect's own
    // cleanup (below) already clears any poll timers left over from the previous scanId.
    setResponsesByQuestion({});
    setDrafts({});
    setSubmitState({});

    if (!scanId || !hasQuestions) {
      setFlagState('disabled');
      return undefined;
    }
    setFlagState('checking');
    let cancelled = false;
    getDefence(scanId)
      .then(({ data }) => {
        if (cancelled) return;
        setFlagState('enabled');
        const rows = Array.isArray(data?.responses) ? data.responses : [];
        const latest = latestRowsByQuestion(rows);
        setResponsesByQuestion(latest);
        // Resume polling for any question left `pending` from a prior visit (e.g. the
        // worker hadn't finished judging when the user last left the page).
        Object.values(latest).forEach((row) => {
          if (row?.status === 'pending' && row.id) {
            pollForResult(row.question_index, row.id);
          }
        });
      })
      .catch(() => {
        // 404 (flag off), ownership failure, or any other load error — render nothing
        // rather than a broken/empty shell.
        if (!cancelled) setFlagState('disabled');
      });
    return () => {
      cancelled = true;
      Object.values(pollTimersRef.current).forEach((timer) => clearInterval(timer));
      pollTimersRef.current = {};
    };
  }, [scanId, hasQuestions, pollForResult]);

  const handleSubmit = async (questionIndex) => {
    const answerText = (drafts[questionIndex] || '').trim();
    if (!answerText) return;
    setSubmitState((s) => ({ ...s, [questionIndex]: { status: 'submitting' } }));
    try {
      const { data } = await submitDefenceAnswer(scanId, { question_index: questionIndex, answer_text: answerText });
      setSubmitState((s) => ({ ...s, [questionIndex]: { status: 'pending' } }));
      pollForResult(questionIndex, data.response_id);
    } catch (err) {
      const status = err?.response?.status;
      const capReached = status === 409;
      // 400 is shared by several distinct validation failures on this route (see the module
      // docstring + isAnswerTooLongError above) — only claim "too long" when the server's own
      // `errors` array actually pins it to answer_text's max_length. This MUST be checked, and
      // its message used, BEFORE calling extractErrorMessage: the server's 400 response always
      // includes a generic `detail: "Invalid request."` string (module docstring lines 24-26),
      // which extractErrorMessage returns unconditionally as soon as it sees a non-empty
      // `detail` string — so passing the length-specific text in as extractErrorMessage's
      // `fallback` param (as this used to do) made it dead code; the fallback is only ever
      // reached when `detail` is absent, which never happens on this route's 400s. Any other
      // 400/error still falls through to the server's actual `detail`/`errors[].msg` text via
      // extractErrorMessage, instead of a guessed label that could be wrong for a different
      // validation failure.
      const lengthInvalid = status === 400 && isAnswerTooLongError(err);
      const message = lengthInvalid
        ? `Your answer is too long (max ${DEFENCE_MAX_ANSWER_CHARS.toLocaleString()} characters). Please shorten it and try again.`
        : extractErrorMessage(
            err,
            capReached
              ? 'You have reached the maximum number of attempts for this question.'
              : 'Could not submit your answer. Please try again.'
          );
      setSubmitState((s) => ({
        ...s,
        [questionIndex]: {
          status: 'error',
          error: message,
        },
      }));
    }
  };

  if (flagState !== 'enabled' || !hasQuestions) return null;

  return (
    <section className="defence-check-section" aria-label="Defence check">
      <div className="defence-check-head">
        <span className="defence-check-kicker">Defence check</span>
        <h2>Show your understanding</h2>
        <p>
          Answer a flagged question in your own words. An AI judge reads your answer against
          the passage and gives you an honest read on how it holds up — this is for your own
          practice, and does not change your score.
        </p>
      </div>
      <div className="defence-check-body">
        {cappedQuestions.map((q, i) => {
          const row = responsesByQuestion[i];
          const state = submitState[i] || {};
          const isPending = state.status === 'pending' || state.status === 'submitting' || row?.status === 'pending';
          const isJudged = row?.status === 'judged' && row.judgment && typeof row.judgment === 'object';
          const isFailed = row?.status === 'failed';
          const axes = isJudged && row.judgment.axes && typeof row.judgment.axes === 'object' ? row.judgment.axes : {};

          return (
            <div className="defence-check-card" key={`${q?.anchor_quote || ''}-${i}`}>
              <div className="defence-check-card-head">
                {q?.dimension && <span className="defence-check-dimension">{dimensionLabel(q.dimension, t)}</span>}
                {q?.anchor_quote && <span className="defence-check-quote">&ldquo;{q.anchor_quote}&rdquo;</span>}
              </div>
              <p className="defence-check-question">{q?.question}</p>

              <textarea
                className="defence-check-textarea"
                rows={4}
                placeholder="Answer in your own words…"
                value={drafts[i] ?? ''}
                onChange={(e) => setDrafts((d) => ({ ...d, [i]: e.target.value }))}
                disabled={isPending}
                maxLength={DEFENCE_MAX_ANSWER_CHARS}
                aria-label="Your answer"
              />

              <div className="defence-check-actions">
                <button
                  type="button"
                  className="btn btn-small btn-primary"
                  onClick={() => handleSubmit(i)}
                  disabled={isPending || !((drafts[i] || '').trim())}
                >
                  {isPending ? 'Judging…' : 'Submit answer'}
                </button>
                {typeof row?.attempt === 'number' && (
                  <span className="defence-check-attempt">Attempt {row.attempt}</span>
                )}
              </div>

              {state.status === 'error' && state.error && (
                <p className="defence-check-error" role="alert">{state.error}</p>
              )}

              {state.timedOut && (
                <p className="defence-check-note">
                  Still judging your answer — this is taking longer than usual. Your answer
                  was saved; check back in a bit.
                </p>
              )}

              {isFailed && (
                <p className="defence-check-note">
                  Judging is unavailable right now — your answer was saved. Please try again later.
                </p>
              )}

              {isJudged && (
                <div className="defence-check-result">
                  <div className="defence-check-axes">
                    {Object.entries(axes).map(([axis, val]) => {
                      const level = val && typeof val === 'object' ? val.level : null;
                      const tone = LEVEL_TONE[level] || LEVEL_TONE.medium;
                      return (
                        <div
                          className={`defence-check-axis-chip is-${level || 'medium'}`}
                          key={axis}
                          style={{ color: tone.color, background: tone.bg }}
                        >
                          <strong>{AXIS_LABELS[axis] || axis}</strong>: {LEVEL_LABELS[level] || level || '—'}
                          {val?.rationale && <span className="defence-check-axis-rationale"> — {val.rationale}</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
