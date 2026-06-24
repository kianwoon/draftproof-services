// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "Why DraftProof",
  title: "Prepare. Improve. Prove.",
  lead: "AI detectors flag your writing after you submit — DraftProof works before. It helps you prepare your draft and keep evidence of responsible AI use, rather than replacing your institution's checker.",
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
  cards: [
    {
      icon: "ti-writing",
      title: "Auto before/after rewrite",
      body: "Scan flags the passages; the rewrite generates a worked before/after for each — the anchors, sources, and specifics to add — so you revise in your own words before you submit.",
    },
    {
      icon: "ti-brain",
      title: "A solution to learn from",
      body: "The rewrite is a teaching example, not a submit-ready answer. You see what grounded writing looks like, then make it yours.",
    },
    {
      icon: "ti-shield-check",
      title: "Policy risk (dual-mode)",
      body: "Two scores from one engine — AI-Allowed and AI-Restricted — matching your institution's actual policy, not a single generic verdict.",
    },
    {
      icon: "ti-clipboard-check",
      title: "Submission & thinking ownership",
      body: "\"Can you defend this as your own work?\" — across text patterns, thinking ownership, and academic grounding, with five dimensions that show what to think harder about.",
    },
  ],
  ctaTitle: "See your writing through a new lens.",
  ctaBody: "Start your first scan — free with your first credits.",
  ctaButton: "Start review",
};
