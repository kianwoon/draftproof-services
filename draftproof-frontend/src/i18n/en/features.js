// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "Why DraftProof",
  title: "Prepare. Improve. Prove.",
  lead: "AI detectors flag your writing after you submit — DraftProof works before, on an engine we built and tuned ourselves: a fine-tuned detection model, sentence-level deep scan, and grounding checks that tell you why — not just a number.",
  techLabel: "Under the hood",
  // allow-hardcode: user-facing tech-pillar copy for /features — marketing prose grounded in
  // the real detector (fine-tune + ESL calibration + deep-scan + claim-graph). Not a scoring
  // oracle. Numbers mirror the vetted public claims (0.8% ESL FPR / 272-essay corpus).
  techPillars: [
    {
      metric: "GPT-5.x · Gemini · Qwen",
      title: "A detector we fine-tuned ourselves",
      body: "Not an off-the-shelf API. DraftProof runs its own model, fine-tuned on frontier-AI output and thousands of real essays — and re-tuned as new frontier models ship, so detection keeps pace with the tools students actually use.",
    },
    {
      metric: "0.8% FPR · lower-proficiency ESL",
      title: "Calibrated for non-native English writers",
      body: "Detectors are notorious for flagging fluent ESL writing as \"AI.\" Every scoring change is gated against a 272-essay corpus of real non-native English writing and ships only if false positives stay low — 0.8% on lower-proficiency writers at our standard threshold.",
    },
    {
      metric: "Sentence by sentence",
      title: "Deep scan reads at sentence level",
      body: "A separate deep-reading model scores your draft sentence by sentence, then fuses with the pattern signal — so no single noisy number can swing a verdict, and you see exactly which lines drive the score.",
    },
    {
      metric: "Entailment-checked",
      title: "Claim-graph grounding",
      body: "DraftProof maps the claims in your argument and checks whether each is actually supported by its cited source — surfacing the ungrounded, generic assertions that are the real reason writing reads as AI-generated.",
    },
  ],
  tableLabel: "How DraftProof is different",
  competitors: ["DraftProof", "GPTZero", "Turnitin", "Originality.ai", "Winston AI"],
  rows: [
    { label: "Paragraph-level output", values: ["yes", "yes", "yes", "yes", "no"] },
    { label: "Explains why content is flagged", values: ["yes", "no", "no", "no", "no"] },
    { label: "Integrated rewrite / coaching", values: ["yes", "no", "no", "no", "no"] },
    { label: "Before/after diff view", values: ["yes", "no", "no", "no", "no"] },
    { label: "Policy-aware scoring", values: ["yes", "no", "no", "no", "no"] },
    { label: "Submission risk framing", values: ["yes", "no", "no", "no", "no"] },
    { label: "Critical thinking assessment", values: ["yes", "no", "no", "no", "no"] },
    { label: "Honest about detector limits", values: ["yes", "no", "no", "no", "no"] },
    { label: "Individual access (no institution needed)", values: ["yes", "yes", "no", "yes", "yes"] },
  ],
  cardsLabel: "DraftProof-only features",
  // allow-hardcode: user-facing feature-card copy for /features — marketing prose, not a detector list or logic gate
  tabs: {
    scan: { label: "Scan", desc: "Diagnose what's flagged and why — before you submit." },
    rewrite: { label: "Rewrite", desc: "A worked before/after solution you revise in your own words." },
  },
  scanCards: [
    {
      icon: "ti-shield-check",
      title: "Policy risk (dual-mode)",
      body: "Two scores from one engine — AI-Allowed and AI-Restricted — matching your institution's actual policy, not a single generic verdict.",
    },
    {
      icon: "ti-clipboard-check",
      title: "Submission risk framing",
      body: "\"Can you defend this as your own work?\" — framed across three layers: text pattern, thinking ownership, and academic grounding.",
    },
    {
      icon: "ti-brain",
      title: "Critical thinking control",
      body: "Five dimensions — specific context, student judgement, reasoning trail, evidence grounding, AI dependency — tell you what to think harder about.",
    },
    {
      icon: "ti-puzzle",
      title: "Works inside Word & Google Docs",
      body: "Scan highlighted text right where you write — DraftProof add-ins for Microsoft Word and Google Docs, no copy-paste.",
    },
  ],
  rewriteLearnMore: "See how the rewrite works →",
  rewriteCards: [
    {
      icon: "ti-writing",
      title: "Auto before/after rewrite",
      body: "Scan flags the passages; the rewrite generates a worked before/after for each — the anchors, sources, and specifics to add — so you revise in your own words before you submit.",
    },
    {
      icon: "ti-bulb",
      title: "A solution to learn from",
      body: "The rewrite is a teaching example, not a submit-ready answer. You see what grounded writing looks like, then make it yours.",
    },
    {
      icon: "ti-git-compare",
      title: "Before/after diff you can act on",
      body: "A highlighted, paragraph-by-paragraph diff shows exactly what changed and why — so you can apply each fix in your own words.",
    },
  ],
  ctaTitle: "See your writing through a new lens.",
  ctaBody: "Start your first scan — free with your first credits.",
  ctaButton: "Start review",
};
