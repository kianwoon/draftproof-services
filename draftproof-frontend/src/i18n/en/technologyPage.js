// allow-hardcode: i18n marketing copy for the /technology page (human-authored UI text,
// not a detection/scoring allow-list or matching oracle) — same pattern as en/whyPage.js.
export const technologyPage = {
  "eyebrow": "The Technology Behind DraftProof",
  "title": "We didn't build another AI detector. We built the engineering discipline a new era of education actually needs.",
  "lead": "Every AI writing checker makes a promise. Very few show their work. Here's what actually runs behind every DraftProof report — the practices we test, the guardrails we enforce, and the fairness bar we refuse to drop below.",
  "pillarsLabel": "The engineering practices behind every DraftProof report",
  "pillars": [
    {
      "title": "No single black-box score decides anything.",
      "body": "A single AI-detector call is noisy and easy to game in either direction. DraftProof combines multiple independent detection signals — pattern-based analysis and a separate deep-reading model — before any tier or score is shown, so no single miscalibrated signal can swing a verdict on its own.",
      "whyItMatters": "For students: one flaky detector can't wrongly flag your work. For educators: a verdict backed by agreement across independent signals is more defensible than a single tool's number."
    },
    {
      "title": "Validated against real multilingual student writing — not just native-English samples.",
      "body": "AI-detection tools are well known to misfire more often on non-native English writers, flagging normal ESL phrasing as \"AI-generated.\" Before any detection change ships, DraftProof runs it against a dedicated corpus of real student essays across proficiency levels, and blocks the release if it would raise false-positive rates for ESL writers or widen the gap between proficiency groups.",
      "whyItMatters": "For students: your writing style shouldn't be penalized for being a second language. For educators: this is the single most common fairness complaint against AI detectors — DraftProof tests for it explicitly, every time, not just once."
    },
    {
      "title": "When a signal is missing or uncertain, the system never guesses against you.",
      "body": "Every additional detection layer is designed to fail safe: if an experimental or unavailable signal can't be computed, DraftProof falls back to its established, calibrated score rather than inventing a harsher one. Guardrails are built to flag content for review, not to silently discard a student's real work.",
      "whyItMatters": "For students: a technical hiccup can't manufacture a false accusation. For educators: the system is built to avoid false positives by design, not just by promise."
    },
    {
      "title": "Re-tested against new AI writing tools as they appear.",
      "body": "AI writing tools keep changing, and a detector tuned once against yesterday's tools quietly goes stale. DraftProof re-validates its detection signals against outputs from multiple, independently-developed AI writing systems on an ongoing basis, so accuracy doesn't quietly decay as the underlying AI landscape shifts.",
      "whyItMatters": "For students: the bar doesn't drift unfairly over time. For educators: you're not relying on a snapshot from whenever the tool launched."
    },
    {
      "title": "You see the calibration, not just a percentage.",
      "body": "Every score DraftProof shows is tied to a labeled band and an explanation of what's driving it — never a bare, unexplained number. The calibration behind each tier is documented and versioned, so a score means the same thing today as it will next term.",
      "whyItMatters": "For students: you get something to act on, not just a grade. For educators: a transparent scale is one you can actually stand behind in a conversation with a student or a parent."
    }
  ],
  "ctaTitle": "This is what \"best stack\" means to us: proof, not adjectives.",
  "ctaBody": "No tool can promise a perfect verdict on every draft. What DraftProof can promise is a system that is tested, fails toward fairness, and keeps being re-checked as AI keeps changing.",
  "ctaRun": "Try it yourself",
  "ctaHow": "See a sample report"
};
