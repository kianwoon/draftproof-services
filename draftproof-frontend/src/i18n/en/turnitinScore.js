// allow-hardcode: human-crafted SEO landing-page copy (marketing prose), not a scoring/matching oracle
export const turnitinScore = {
  "eyebrow": "Turnitin AI score",
  "title": "What does your Turnitin AI score actually mean?",
  "lead": "The percentage at the top of a Turnitin AI report is one of the most misunderstood numbers in academia. It is not a plagiarism score, not a similarity score, and not proof of misconduct. Here's how to read it — honestly.",
  "ctaPrimaryLabel": "Check your own writing",
  "ctaSecondaryLabel": "See the content checker",
  "heroStat": {
    "label": "Best for",
    "value": "Understanding a flag",
    "detail": "Before you panic or argue your case"
  },
  "intro": [
    {
      "title": "What the percentage is",
      "body": "It's Turnitin's estimate of how much of your qualifying text matches patterns its model associates with AI-generated writing. An estimate — produced by a model that can be wrong."
    },
    {
      "title": "What it is not",
      "body": "Not a plagiarism or similarity score. Not a measure of how good your writing is. And not, on its own, evidence of academic misconduct — Turnitin itself says scores require human judgement."
    }
  ],
  "sections": [
    {
      "eyebrow": "Reading the number",
      "title": "What the percentages actually mean.",
      "type": "cards",
      "items": [
        {
          "title": "The 20% floor",
          "body": "Turnitin hides scores between 1–19% behind an asterisk because detection in that range has low confidence and more false positives. A blank or * doesn't mean you're clean — the signal is just too weak to report."
        },
        {
          "title": "Higher percentages",
          "body": "A higher number means more of your text matched AI-like patterns — not that a specific passage is definitely AI. The model flags style, not authorship."
        },
        {
          "title": "False positives are real",
          "body": "Plain, well-structured, or non-native English writing can read as AI-like. Genuine human work does get flagged. The score is a prompt to look closer, not a verdict."
        }
      ]
    },
    {
      "eyebrow": "Why it happens",
      "title": "Why human writing gets flagged.",
      "type": "cards",
      "items": [
        {
          "title": "Generic, unanchored claims",
          "body": "Sentences that are true but could appear in any essay — no specifics, no sources — look statistically predictable, which is exactly what detectors react to."
        },
        {
          "title": "Over-smoothed phrasing",
          "body": "Heavy editing, grammar tools, and template structures flatten the natural unevenness of human writing."
        },
        {
          "title": "Missing evidence",
          "body": "Claims without citations or concrete grounding are the dominant risk signal — more than any single word choice."
        }
      ]
    },
    {
      "eyebrow": "What to do",
      "title": "What actually helps after a flag.",
      "type": "steps",
      "items": [
        {
          "title": "Don't just reword it",
          "body": "Synonym-swapping keeps the same generic, unanchored claims — the thing detectors react to. It rarely changes the score and often makes the writing worse."
        },
        {
          "title": "Add the specifics only you have",
          "body": "Ground claims in real sources, examples, and your own reasoning. Concrete, anchored writing is both stronger and less AI-like."
        },
        {
          "title": "Check before you submit",
          "body": "Run your draft through DraftProof to see which paragraphs drive the risk and exactly what's missing — paragraph by paragraph."
        }
      ]
    }
  ],
  "linksEyebrow": "Keep reading",
  "linksTitle": "Related guides",
  "links": [
    {
      "label": "Why detectors flag human writing",
      "to": "/why",
      "body": "The mechanics behind a false positive."
    },
    {
      "label": "Turnitin vs AI detectors",
      "to": "/turnitin-vs-ai-detectors",
      "body": "How Turnitin, GPTZero, and Originality.ai actually compare."
    },
    {
      "label": "How to reduce AI detection (honestly)",
      "to": "/reduce-ai-detection",
      "body": "Why beating the detector is the wrong goal."
    },
    {
      "label": "Academic integrity & AI",
      "to": "/academic-integrity-ai",
      "body": "Where AI use is fine — and where it isn't."
    }
  ],
  "ctaTitle": "See what's driving your score.",
  "ctaBody": "DraftProof shows you which paragraphs look risky and what they're missing — so you can fix the writing, not game the number."
};
