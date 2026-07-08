// allow-hardcode: human-crafted SEO landing-page copy (marketing prose), not a scoring/matching oracle
export const turnitinVsDetectors = {
  "eyebrow": "AI detector comparison",
  "title": "Turnitin vs AI detectors: how they compare, and what actually matters",
  "lead": "Turnitin, GPTZero, and Originality.ai all try to answer the same question in different ways — and they don't always agree with each other. Here's what each one actually measures, where each one gets it wrong, and why DraftProof isn't in this table as another rival score.",
  "ctaPrimaryLabel": "Check your own writing",
  "ctaSecondaryLabel": "See the content checker",
  "heroStat": {
    "label": "Best for",
    "value": "Understanding the landscape",
    "detail": "Before any single score changes your plans"
  },
  "intro": [
    {
      "title": "Different tools, different methods",
      "body": "Turnitin, GPTZero, and Originality.ai each use their own model and training data. The same document can score very differently across all three — that's documented, expected behavior, not a bug in any one of them."
    },
    {
      "title": "DraftProof isn't a fourth competing score",
      "body": "DraftProof doesn't replace or out-score any of these tools. It's a pre-submission step — it shows you which paragraphs are weakly grounded before an institution's detector ever sees your draft."
    }
  ],
  "sections": [
    {
      "eyebrow": "Side by side",
      "title": "How the three detectors compare.",
      "type": "comparison",
      "columns": ["Turnitin", "GPTZero", "Originality.ai", "DraftProof"],
      "rows": [
        {
          "label": "What it's built for",
          "values": [
            "Institution-side originality + AI writing reports",
            "Standalone AI-text detection, education and publishing",
            "Standalone AI-text + plagiarism detection, publishing/SEO",
            "Pre-submission prep — find weak spots before you submit"
          ]
        },
        {
          "label": "How it scores",
          "values": [
            "BERT-based classifier; 0–100% of qualifying text",
            "Perplexity/burstiness-derived, 7-component proprietary model",
            "0–100% sentence-level score across 4 detection models",
            "Paragraph-level risk + grounding/citation gaps, not a single %"
          ]
        },
        {
          "label": "Published accuracy",
          "values": [
            "~98% accuracy, <1% false-positive claim above 20%",
            "~99% accuracy claim",
            "~83.4% detection rate, ~5% false-positive claim",
            "Not a pass/fail score — shows findings you can verify yourself"
          ]
        },
        {
          "label": "Known weak spot",
          "values": [
            "Scores under 20% are explicitly lower-confidence (its own asterisk)",
            "Independent testing found much higher false positives on non-native English writing",
            "False-negative rate rises sharply on paraphrased/humanized AI text",
            "Doesn't issue an institutional verdict — you still review the output"
          ]
        },
        {
          "label": "Who sees the result",
          "values": [
            "Your institution, via its Turnitin account",
            "Whoever runs the check — instructor or student tool",
            "Whoever runs the check — often publishers/agencies",
            "Only you, before you submit anywhere"
          ]
        }
      ]
    },
    {
      "eyebrow": "The mechanics",
      "title": "What each detector is actually measuring.",
      "type": "cards",
      "items": [
        {
          "title": "Turnitin",
          "body": "Uses a BERT-based model trained on years of student submissions plus AI-generated text from GPT-3 through GPT-4-class models. Per Turnitin's own guidance, the percentage is an estimate of qualifying text that resembles AI-writing patterns — not a similarity or plagiarism score, and not proof of misconduct on its own."
        },
        {
          "title": "GPTZero",
          "body": "Started from perplexity (how predictable the wording is) and burstiness (how much sentence rhythm varies), and now blends those into a larger proprietary model. Independent reporting and academic testing have both raised concerns about false positives, particularly on non-native English writing — GPTZero has since worked on ESL-specific debiasing."
        },
        {
          "title": "Originality.ai",
          "body": "Scores text sentence-by-sentence across four detection models trained on a large labeled dataset, and bundles plagiarism/readability/fact-checking in the same scan. Its own published numbers show accuracy drops the most on AI text that's been run through a paraphrasing or humanizing tool first."
        }
      ]
    },
    {
      "eyebrow": "Why this happens",
      "title": "Why detector scores disagree with each other.",
      "type": "steps",
      "items": [
        {
          "title": "Different training data, different blind spots",
          "body": "Each detector learned from a different mix of human and AI text. A style one model has never seen scores differently than one it was trained on extensively."
        },
        {
          "title": "No shared, audited ground truth",
          "body": "There's no universal reference set all detectors are scored against publicly — each vendor reports its own accuracy claims, from its own test set."
        },
        {
          "title": "That's exactly why one number shouldn't decide anything",
          "body": "If three well-funded tools can disagree on the same paragraph, treat any single score as a prompt to look closer — not a verdict, from Turnitin or anyone else."
        }
      ]
    }
  ],
  "linksEyebrow": "Keep reading",
  "linksTitle": "Related guides",
  "links": [
    {
      "label": "What does your Turnitin AI score mean?",
      "to": "/turnitin-ai-score",
      "body": "A closer read of the number your institution sees."
    },
    {
      "label": "Academic integrity & AI",
      "to": "/academic-integrity-ai",
      "body": "Where AI use is fine — and where it isn't."
    },
    {
      "label": "Why detectors flag human writing",
      "to": "/why",
      "body": "The mechanics behind a false positive."
    }
  ],
  "ctaTitle": "See what a detector would flag — before you submit.",
  "ctaBody": "DraftProof checks your draft for the same weak grounding and generic phrasing detectors react to, paragraph by paragraph, so you can fix it before any of these tools see it."
};
