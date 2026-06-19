// allow-hardcode: i18n UI strings for pricing feature lists — human-crafted marketing copy, not a scoring/matching oracle
export const pricing = {
  "eyebrow": "Pricing",
  "title": "Pay for the review you need.",
  "lead": "No subscriptions. Buy credits once and use them for grounded review or focused revision.",
  "baseRate": "Base rate",
  "perWords": "per 1,000 words",
  "scanTitle": "Scan",
  "scanUnit": "/ 1,000 words",
  // allow-hardcode: i18n marketing copy for scan feature list — UI strings, not a detection/scoring oracle
  "scanFeatures": [
    "1 credit per 1,000 words (first 800 words free)",
    "Paragraph-level findings — not just a single score",
    "Citation gap detection — claims that lack source support",
    "Source integrity check — whether cited sources actually support the claim",
    "AI-like writing signals — generic phrasing and predictable structure",
    "Policy risk scoring — separate scores for AI-allowed vs AI-restricted policies",
    "Submission risk framing — text pattern, ownership, citation, and defence readiness",
    "Critical thinking assessment — 5 dimensions of author control",
    "Authorship signals — human-written markers already in your draft",
    "PDF report + email delivery"
  ],
  "rewriteTitle": "Rewrite",
  "rewriteUnit": "/ 1,000 words",
  // allow-hardcode: i18n marketing copy for rewrite feature list — UI strings, not a detection/scoring oracle
  "rewriteFeatures": [
    "5 credits per rewrite (completed scan required)",
    "Paragraph-level rewrite — only flagged sections are changed",
    "Grounding-based coaching — adds concrete anchors, examples, and evidence",
    "Explains what changed and why in each paragraph",
    "Before/after diff view — see exactly what DraftProof changed",
    "Preserves your original voice, argument, and structure",
    "Author checks highlighted — human-written markers surface for review",
    "Reviewable draft — a teaching example to learn from, not a final submission"
  ],
  "startScan": "Start a scan",
  "startWithScan": "Start with a scan",
  "signInStart": "Sign in to get started",
  "faqTitle": "Frequently asked questions",
  "faqs": [
    {
      "q": "Can DraftProof help me pass Turnitin?",
      "a": "DraftProof helps you prepare before a Turnitin-style review by finding citation gaps, weak source grounding, similarity risk, generic phrasing, and AI-like writing signals. It does not guarantee a Turnitin result, bypass Turnitin, or prove authorship. It gives you guidance so you can revise the draft responsibly before submission."
    },
    {
      "q": "How should I use DraftProof as guidance?",
      "a": "Start with a scan, review the report, then fix the highest-risk issues first: unsupported claims, missing citations, vague phrasing, overly uniform sections, and AI-style wording. If you use rewrite, treat the rewritten text as a teaching draft to review, personalize, and verify against your sources."
    },
    {
      "q": "What if my AI signal is above 20%?",
      "a": "Turnitin treats low AI ranges differently because lower scores are less reliable, and scores above that range usually deserve closer review. DraftProof highlights the sections and patterns that may need attention, but the right fix is stronger evidence, clearer source use, and more authentic explanation, not random wording changes."
    },
    {
      "q": "Is DraftProof a Turnitin checker?",
      "a": "No. DraftProof is not Turnitin and does not access Turnitin’s database or guarantee the same score. It is a pre-submission review tool that gives Turnitin-aware guidance using DraftProof’s own integrity, citation, writing, and authorship signals."
    },
    {
      "q": "What counts as a scan?",
      "a": "Scans with 800 words or fewer are free. After that, 1 credit covers each started 1,000-word block. For example, a 2,500-word document costs 3 credits."
    },
    {
      "q": "Will the rewrite make my work safe to submit?",
      "a": "No rewrite can honestly guarantee that. Rewrite shows how to reduce risky patterns and improve clarity, grounding, and academic tone, but you remain responsible for checking the meaning, citations, course requirements, and academic integrity rules before submission."
    },
    {
      "q": "Do I need a scan before a rewrite?",
      "a": "Yes. A completed scan is required before rewrite. DraftProof uses the scan report to identify which sections are safe and useful to revise."
    },
    {
      "q": "When are rewrite credits deducted?",
      "a": "Rewrite credits are deducted only when DraftProof delivers rewritten content. If no useful revision candidate is found, the original text is preserved, the rewrite fails, or no rewritten content is produced, reserved credits are released and not deducted."
    },
    {
      "q": "Do credits expire?",
      "a": "No. Your credits stay in your account until you use them."
    },
    {
      "q": "How do I submit my text?",
      "a": "Paste your text directly into the scan page. File upload coming soon."
    }
  ]
};
