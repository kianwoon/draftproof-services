// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "Why DraftProof",
  title: "Detect. Understand. Improve.",
  lead: "Every other detector tells you that you failed. DraftProof shows you how to pass — by teaching you to write better.",
  tableLabel: "How we compare",
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
  cards: [
    {
      icon: "ti-writing",
      title: "Grounded rewrite coaching",
      body: "Shows a worked example of your paragraph with concrete anchors, sources, and specifics — so you see what grounded writing looks like.",
    },
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
  ],
  ctaTitle: "See your writing through a new lens.",
  ctaBody: "Start your first scan — free with your first credits.",
  ctaButton: "Start review",
};
