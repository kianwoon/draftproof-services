export const landing = {
  "heroPill": "Grounding and critical-thinking review for students",
  "heroTitle": "Make sure the thinking in your draft is still yours.",
  "heroTitleHighlight": "still yours",
  "heroTitlesLabel": "Rotating headline messages",
  "heroTitles": [
    { "text": "Make sure the thinking in your draft is still yours.", "highlight": "still yours" },
    { "text": "Catch the claims you can't back up before your professor does.", "highlight": "before your professor does" },
    { "text": "Read your draft under your school's rules, not a generic score.", "highlight": "not a generic score" },
    { "text": "Ground every point with your own evidence and voice.", "highlight": "your own evidence and voice" }
  ],
  "heroLead": "DraftProof points to the claims you can’t yet back up and the passages that sound machine-written, then asks what you actually meant — so you ground every point with your own evidence and reasoning before submission.",
  "runCheck": "Review my content",
  "viewSample": "View sample report",
  // allow-hardcode: i18n display copy. "5" mirrors the WELCOME_CREDITS default
  // (draftproof-api/app/config.py); update if that env value changes. Not a scoring oracle.
  // Split into pre/link/post so "Sign up" renders as a link to the sign-in page.
  "heroFreeCreditsPre": "New here? ",
  "heroFreeCreditsLink": "Sign up",
  "heroFreeCreditsPost": " and get 5 free credits to try — no card needed.",
  "trustNote": "Not a Turnitin replacement · Not a misconduct verdict · Prepare before submission, with evidence of responsible AI use.",
  // allow-hardcode: i18n display copy for the landing use-case carousel — UI text rendered to users,
  // never compared against document content or used as a scoring/matching oracle (same category as heroReviewSteps/anchorCards).
  "useCasesLabel": "How DraftProof helps in your situation",
  "useCasesEyebrow": "Built around how your school actually treats AI",
  "useCasesHeading": "More than a score — guidance for your exact situation",
  "useCasesTabsLabel": "Use case sections",
  "useCasesNext": "Show next use case",
  // allow-hardcode: i18n display copy (carousel slide text shown to users) — not a scoring/matching oracle.
  "useCases": [
    {
      "id": "score-vs-coaching",
      "eyebrow": "A score isn't help",
      "tag": "The problem with checkers",
      "title": "A percentage doesn't tell you what to fix.",
      "body": [
        "Turnitin and free AI checkers hand back one number and leave you to guess. A percentage tells you that you're flagged — it never tells you which sentence, why, or what to do next.",
        "DraftProof reads the draft instead: it marks the exact claims and passages at risk and gives you a highlighted before/after for each one, so you can see what to add, anchor, or rewrite in your own words before you submit."
      ],
      // allow-hardcode: i18n display copy (carousel slide points/guardrails/punch shown to users) — not a scoring/matching oracle.
      "points": [
        "Pinpoints the exact claims and passages at risk",
        "Explains why each passage reads as machine-written",
        "Shows a before/after you edit with your own content",
        "Turns the score into specific, fixable steps",
        "Honest: tells you when a detector could still flag it"
      ],
      "guardrails": [
        "Not a score to game. Not a misconduct verdict.",
        "A number says you're flagged — DraftProof says why, and what to fix."
      ],
      "punch": "A checker hands you a verdict; DraftProof hands you a revision you can act on."
    },
    {
      "id": "ai-allowed",
      "eyebrow": "If your school allows AI",
      "tag": "AI-assisted, done right",
      "title": "Prove your AI-assisted work is grounded and yours.",
      "body": [
        "Where AI is permitted, the bar isn't whether you used it — it's quality and control. DraftProof weighs grounding, judgment and specificity, and treats surface AI-wording lightly.",
        "It reads your draft the way a course that allows AI would: does each claim rest on real evidence and your own reasoning? You see exactly where the work is well-grounded and where it still reads as generic — so you can make it unmistakably yours."
      ],
      // allow-hardcode: i18n display copy (carousel slide points/guardrails/punch shown to users) — not a scoring/matching oracle.
      "points": [
        "Rewards real evidence, reasoning and your own framing",
        "Checks each claim is anchored to a source or example",
        "Surface AI wording weighs light — using AI is fine",
        "Flags generic, unsupported assertions to ground",
        "Lenient bands tuned for permitted, declared AI use"
      ],
      "guardrails": [
        "Not a score to game. Not a misconduct verdict.",
        "Where AI is allowed, the bar is quality and control — not whether you used it."
      ],
      "punch": "Acceptable AI-assisted work is grounded, specific, and still yours."
    },
    {
      "id": "ai-banned",
      "eyebrow": "If your school bans AI",
      "tag": "Prove the thinking is yours",
      "title": "Show the work is authored by you, not a model.",
      "body": [
        "Where AI is prohibited, DraftProof weighs authorship voice and surface AI-text more, and flags the passages that read machine-written — so you can rewrite them in your own voice before a reader ever questions them.",
        "It can't prove how you wrote it, and never pretends to. It surfaces the passages a reader might challenge and the factors only you can confirm — so the defence is the one that actually holds: the thinking, and the words, are yours. Never a bypass, never an accusation."
      ],
      // allow-hardcode: i18n display copy (carousel slide points/guardrails/punch shown to users) — not a scoring/matching oracle.
      "points": [
        "Surfaces passages that don't sound like your voice",
        "Weighs authorship voice and surface AI-text more",
        "Shows where to rewrite in your own words and reasoning",
        "Stricter bands — built for a no-AI policy",
        "Confirm-yourself factors you control, never assumed"
      ],
      "guardrails": [
        "Not a score to game. Not an accusation that you used AI.",
        "We can't prove how you wrote it — only you can, by owning the thinking."
      ],
      "punch": "Authoring the thinking yourself is the real defence, not disguising the text."
    }
  ],
  "quickSummary": "DraftProof Quick Summary",
  "livePreview": "Live preview",
  "preSubmissionReview": "Content pre-submission review",
  "runningCheck": "Checking content risks before submission...",
  "reviewTier": "Review tier",
  "medium": "Medium",
  "grounding": "Grounding",
  "strong": "Strong",
  "citationGaps": "Citation gaps",
  "foundCount": "{{count}} found",
  "sourceIntegrity": "Source integrity",
  "verified": "Verified",
  "primaryFix": "Primary fix",
  "oneCitation": "1 citation",
  "actionable": "Actionable",
  "heroReviewStepsLabel": "DraftProof review preview steps",
  "heroReviewSteps": [
    {
      "id": "scan-draft",
      "visual": "scan",
      "stepLabel": "Step 1",
      "shortLabel": "01",
      "title": "Scanning your draft",
      "body": "DraftProof reads claims, citations, source links, and AI-like writing patterns before submission.",
      "scanLines": [
        {
          "label": "Claim",
          "text": "Hollywood has become one of the most powerful cultural exports.",
          "tone": "warning"
        },
        {
          "label": "Source",
          "text": "No linked source appears near this sentence.",
          "tone": "warning"
        },
        {
          "label": "Context",
          "text": "Draft keeps the student's argument and course focus.",
          "tone": "positive"
        }
      ],
      "scanStatus": [
        {
          "label": "Map",
          "value": "Reading",
          "tone": "positive"
        },
        {
          "label": "Cite",
          "value": "Missing",
          "tone": "warning"
        },
        {
          "label": "Voice",
          "value": "Kept",
          "tone": "positive"
        }
      ],
      "primaryLabel": "Current focus",
      "primaryValue": "Claims without anchors",
      "badge": "Scanning"
    },
    {
      "id": "find-risks",
      "visual": "findings",
      "stepLabel": "Step 2",
      "shortLabel": "02",
      "title": "Content risks found",
      "body": "The review separates evidence gaps from wording signals so you know what may be questioned.",
      "findings": [
        {
          "label": "Citation gap",
          "body": "A broad claim needs a nearby citation or source anchor.",
          "badge": "High",
          "tone": "warning"
        },
        {
          "label": "Weak grounding",
          "body": "The source relationship is not visible to the reader.",
          "badge": "Medium",
          "tone": "warning"
        },
        {
          "label": "Source integrity",
          "body": "Existing source details look usable after citation placement.",
          "badge": "Verified",
          "tone": "positive"
        }
      ],
      "summary": [
        {
          "label": "Review tier · Medium",
          "tone": "warning"
        },
        {
          "label": "2 citation risks",
          "tone": "warning"
        },
        {
          "label": "Source verified",
          "tone": "positive"
        }
      ],
      "primaryLabel": "Main risk",
      "primaryValue": "Evidence gap",
      "badge": "Found"
    },
    {
      "id": "fix-plan",
      "visual": "diff",
      "stepLabel": "Step 3",
      "shortLabel": "03",
      "title": "Revision plan ready",
      "body": "DraftProof turns the findings into a focused order of fixes you can act on before submission.",
      "diffRows": [
        {
          "label": "Before",
          "text": "Hollywood has become one of the most powerful cultural exports in history.",
          "tone": "remove"
        },
        {
          "label": "After",
          "text": "Hollywood reaches global audiences through films, streaming, and music when supported by a source.",
          "tone": "add"
        }
      ],
      "fixSteps": [
        "Add source",
        "Tie source to claim",
        "Keep meaning"
      ],
      "primaryLabel": "Primary fix",
      "primaryValue": "Add source to claim",
      "badge": "Actionable"
    }
  ],
  "builtFor": "Built for",
  "audienceDetails": "DraftProof audiences and review details",
  "students": "Undergraduates",
  "gradWriters": "Grad & thesis writers",
  "eslWriters": "ESL & multilingual writers",
  "independentWriters": "Independent researchers",
  "tokenRate": "1 credit per 1,000 words",
  "pdfReport": "PDF report",
  "citationGrounding": "Citation + similarity review",
  "contentAwareRewrite": "Grounding-aware revision guidance",
  "contentCarouselLabel": "DraftProof grounding and AI-risk context",
  "contentCarouselTabsLabel": "Grounding context sections",
  "contentCarouselNext": "Show next grounding context section",
  "humanWrittenEyebrow": "AI-like signals",
  "humanWrittenTitle": "Human-written work can still look AI-generated.",
  "humanWrittenBody1": "AI writing reports don't know who wrote each sentence. They read patterns — predictability, rhythm, generic phrasing, weak grounding — that can resemble AI-generated text, even in human work.",
  "humanWrittenBody2": "DraftProof explains those signals before submission, so you can revise with stronger evidence, clearer reasoning, and better source grounding.",
  "humanWrittenSignalsLabel": "Writing signals DraftProof can explain",
  "humanWrittenSignals": [
    "AI-like predictability",
    "Generic academic phrasing",
    "Over-smooth rewriting",
    "Weak evidence and grounding",
    "Low sentence variation",
    "Missing author reasoning"
  ],
  "humanWrittenGuardrail1": "We do not promise to bypass detectors.",
  "humanWrittenGuardrail2": "We do not guarantee any Turnitin result.",
  "humanWrittenPunch": "Human-written does not always mean low-risk. DraftProof explains why.",
  "humanizerEyebrow": "The humanizer trap",
  "humanizerTitle": "\"AI humanizers\" are the wrong fix for a grounding problem.",
  "humanizerBody1": "AI \"humanizers\" promise undetectable text by reshuffling words. That makes writing awkward, blurs your meaning, and leaves the real problem untouched: claims that still need clearer evidence or author-owned reasoning.",
  "humanizerBody2": "Turnitin's own guidance says these reports can misidentify text modified by paraphrasing or bypasser tools, and need human judgment. DraftProof doesn't play that game — it shows what to ground, verify, and finish in your own words.",
  "humanizerSignalsLabel": "What AI humanizers actually do",
  "humanizerSignals": [
    "Scramble wording instead of fixing claims",
    "Read awkward or inconsistent",
    "Quietly change your meaning",
    "Leave citation gaps untouched",
    "Create bypasser-tool review risk",
    "Hide the work you should be able to explain"
  ],
  "humanizerGuardrail1": "DraftProof does not play the bypass game.",
  "humanizerGuardrail2": "We strengthen your evidence, reasoning, and review trail honestly.",
  "humanizerPunch": "Owning the claim, source, and wording is stronger than disguising the text.",
  "humanizerSourceLabel": "Source: Turnitin AI writing report guidance",
  "humanizerSourceUrl": "https://guides.turnitin.com/hc/en-us/articles/22774058814093-Using-the-AI-Writing-Report",
  "anchorEyebrow": "Content anchors",
  "anchorTitle": "Passing an AI scan is not the same as writing a strong submission.",
  "anchorBody1": "A detector can be satisfied while the paper still fails the reader. Thin claims, weak source links, and missing assignment context leave nothing real to revise — shuffling the words just moves the same weakness around.",
  "anchorBody2": "DraftProof treats content as the root input: what the claim says, which source backs it, what you can explain, and what the assignment asks for. The strongest revision keeps those anchors visible instead of hiding them under smoother wording.",
  "anchorCardsLabel": "Content anchors DraftProof reviews",
  "anchorCards": [
    {
      "label": "01",
      "title": "Claim content",
      "body": "What the paragraph is actually trying to prove, not just whether the sentence sounds natural."
    },
    {
      "label": "02",
      "title": "Source anchor",
      "body": "Which citation, reading, dataset, case, or lecture detail supports the claim."
    },
    {
      "label": "03",
      "title": "Author anchor",
      "body": "What the student can explain from their own reasoning, observation, method, or course context."
    },
    {
      "label": "04",
      "title": "Assignment anchor",
      "body": "Whether the answer matches the prompt, rubric, discipline, and expected evidence type."
    }
  ],
  "anchorWorkflowLabel": "Grounding workflow before revision",
  "anchorWorkflow": [
    "Find the claim before changing the wording.",
    "Tie the claim to a source, example, method, or author-owned explanation.",
    "Only then revise the sentence so the stronger content remains visible."
  ],
  "criticalEyebrow": "Critical thinking control",
  "criticalTitle": "Are you in control of the thinking, or is the AI?",
  "criticalBody1": "AI hands back one confident, polished answer fast. The risk in education isn't using AI for research or brainstorming — it's quietly handing over the judgement: accepting the first answer without questioning it, comparing alternatives, or checking the evidence.",
  "criticalBody2": "DraftProof now turns your draft into reflective questions anchored to what you actually wrote — your context, your judgement, your reasoning, and your evidence. They are prompts to sharpen your thinking, not a score or a verdict. The answers are yours to write.",
  "criticalSignalsLabel": "Questions DraftProof asks about your own draft",
  // allow-hardcode: static landing-page UI copy (reflective question prompts shown
  // to the reader), mirroring report.criticalThinking.dimensions — never compared
  // against document text, not a scoring/matching oracle.
  "criticalSignals": [
    "What specific assignment, case, or observation is this claim drawn from?",
    "Which source or data point could you cite to verify it?",
    "What reasoning led you to this position — and did you weigh alternatives?",
    "Where did you challenge the first answer instead of keeping it?",
    "What here can you explain in your own words, without the AI?"
  ],
  "criticalGuardrail1": "Not a score. Not a misconduct verdict.",
  "criticalGuardrail2": "The questions are about your draft — the answers are yours to write.",
  "criticalPunch": "Staying in control of the thinking is the real defence, not disguising the text.",
  "studentMisuseEyebrow": "Student misconception",
  "studentMisuseTitle": "\"Write broken English, bad grammar... can get me away?\"",
  "studentMisuseBody1": "Wrong move. Schools didn't add AI checks because good English is suspicious — they added them to see whether you understood the material, questioned the output, checked the evidence, and can explain your reasoning.",
  "studentMisuseBody2": "Bad spelling, broken grammar, and cheap word reshuffling do not create real authorship. They make the work weaker. A stronger submission shows ownership: your claim, your source, your course context, your examples, and your explanation.",
  "studentMisuseSignalsLabel": "What the student should repair instead",
  "studentMisuseSignals": [
    "Thin claims with no real support",
    "Sources that do not match the sentence",
    "Generic points the student cannot explain",
    "Missing course or assignment context",
    "Examples copied without digestion",
    "Wording that hides the author trail"
  ],
  "studentMisuseGuardrail1": "Do not make the draft worse to look human.",
  "studentMisuseGuardrail2": "Make the thinking visible before you re-scan.",
  "studentMisusePunch": "The fix is clearer ownership, not messier English.",
  "sampleEyebrow": "Sample Report",
  "reportStrategyCarouselLabel": "Sample report and content-aware revision guidance",
  "reportStrategyCarouselTabsLabel": "Sample report and revision guidance sections",
  "reportStrategyCarouselNext": "Show next report or guidance section",
  "sampleTitle": "See exactly what to fix before you submit.",
  "sampleBody": "DraftProof turns a scan into a clear review plan: what may be questioned, which claims need grounding, why the score moved, and how to revise responsibly without losing your own meaning.",
  "samplePoint1": "Explain the score instead of guessing from one percentage",
  "samplePoint2": "Prioritize citation, source, similarity, and AI-style risks",
  "samplePoint3": "Keep a PDF review trail before submission",
  "reportValueLabel": "What DraftProof report gives you",
  "reportValueCards": [
    {
      "title": "Explain the score",
      "body": "See the signal profile behind AI-style writing, grounding risk, similarity, and authorship uncertainty."
    },
    {
      "title": "Prioritize the fix",
      "body": "Focus first on citation gaps, weak source support, and high-impact writing issues."
    },
    {
      "title": "Review responsibly",
      "body": "Use rewrite guidance as a teaching draft for stronger grounding, then finish it with your own real details."
    },
    {
      "title": "Keep a record",
      "body": "Download a PDF report showing what was checked before submission."
    }
  ],
  "runOwnScan": "Run your own scan",
  "helpEyebrow": "How DraftProof Helps",
  "helpTitle": "Prepare your content before review.",
  "helpLead": "DraftProof turns vague submission anxiety into a clear revision plan. It shows which claims may be questioned, why they look weak, and what to ground before you submit.",
  "whyEyebrow": "Why Students Use DraftProof",
  "whyTitle": "Turnitin changed how students think about originality, citations, and AI-style writing.",
  "strategyEyebrow": "Content-Aware Help",
  "strategyTitle": "Different assignments need different revision guidance.",
  "strategyBody": "A student draft, research paragraph, policy note, and personal reflection should not be revised the same way. DraftProof first understands what you wrote, then suggests where to add evidence, context, or author-owned reasoning.",
  "strategyProofStrong": "DraftProof does not blindly humanize text.",
  "strategyProofBody": "It helps students fix citation, grounding, similarity, and AI-like wording risks in the content they actually wrote, then treats rewrite output as something to review, not submit untouched.",
  "engineEyebrow": "How It Works",
  "engineTitle": "Four checks. One clear report.",
  "engineLead": "DraftProof analyses your content across four dimensions before you submit.",
  "beliefsEyebrow": "What DraftProof Believes",
  "beliefsTitle": "Writing tools should be fair, transparent, and useful.",
  "positiveBelief": "We believe users deserve clear, evidence-based feedback that helps them improve their work.",
  "ctaPill": "Before submission, review what a reader may question.",
  "ctaTitle": "DraftProof helps you ground the content before submission.",
  "ctaBody": "Before you submit, check that the work is original, properly cited, clearly written, and backed by sources or your own reasoning.",
  "ctaButton": "Review my content",
  // allow-hardcode: i18n display copy. "5" mirrors the WELCOME_CREDITS default; not a scoring oracle.
  "ctaFreeCredits": "New accounts get 5 free credits — no card needed.",
  "ctaSmall": "1 credit per 1,000 words · PDF report included · No bypass claims",
  "reportPreviewLabel": "DraftProof sample content review report preview",
  "reportPreviewTabsLabel": "Sample report sections",
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "Authorship Breakdown",
      "summary": "4-way composition"
    },
    {
      "id": "aiSignal",
      "label": "AI Signal",
      "summary": "Authorship pattern"
    },
    {
      "id": "actionPlan",
      "label": "Action Plan",
      "summary": "What to fix"
    },
    {
      "id": "findings",
      "label": "Findings",
      "summary": "Paragraph detail"
    },
    {
      "id": "criticalThinking",
      "label": "Critical Thinking",
      "summary": "Sharpen your thinking"
    }
  ],
  // allow-hardcode: static sample-report UI copy (illustrative reflective questions
  // shown on the landing page), anchored to the fixed sample essay above — never
  // compared against any user document, not a scoring/matching oracle.
  "sampleCriticalQuestions": [
    {
      "quote": "Hollywood has become one of the most powerful cultural exports in history.",
      "question": "Which specific films, figures, or sources show this — and over what time period are you claiming it?"
    },
    {
      "quote": "American movies, music, fashion, and social media trends are consumed globally.",
      "question": "What example from your own reading or observation backs \"consumed globally\" rather than a general impression?"
    },
    {
      "quote": "the United States has a strong cultural influence.",
      "question": "What reasoning led you to call this influence \"strong\", and did you weigh any counter-examples?"
    }
  ],
  "findingsSampleType": "AI Likelihood",
  "findingsSampleDescription": "The paragraph uses a standard transition \"In addition\" and several other familiar phrases that make the writing sound formulaic.",
  "findingsSamplePosition": "2/5",
  "findingsSampleCount": "3 flagged sentences in paragraph",
  // allow-hardcode: illustrative sample flagged-sentence evidence shown on the landing
  // page — fixed marketing example built from the same sample paragraph used elsewhere
  // on this page, never compared against document content, not a scoring/matching oracle.
  "sampleFlaggedSentences": [
    {
      "text": "In addition to economics, the United States has a strong cultural influence.",
      "score": 61,
      "suggestion": "Anchor to a specific example or source instead of a general claim."
    },
    {
      "text": "The entertainment industry in Hollywood has become one of the most powerful cultural exports in history.",
      "score": 74,
      "suggestion": "Beyond its economy, American cinema reaches audiences in over 190 countries each year."
    }
  ],
  "sampleActionItems": [
    {
      "title": "Add citation support",
      "body": "Two claims need clearer source backing before submission.",
      "label": "High priority"
    },
    {
      "title": "Strengthen source grounding",
      "body": "One paragraph should explain how the cited source supports the point.",
      "label": "Medium priority"
    },
    {
      "title": "Revise generic phrasing",
      "body": "Replace broad AI-style wording with specific reasoning and evidence.",
      "label": "Quick win"
    }
  ],
  "checks": [
    {
      "title": "Citation gaps",
      "body": "Identifies claims that need a source but don't have one."
    },
    {
      "title": "Source integrity",
      "body": "Checks whether cited sources actually support the claim made."
    },
    {
      "title": "Generic phrasing",
      "body": "Flags writing that sounds generic, ungrounded, or unsupported by evidence."
    },
    {
      "title": "Authorship signals",
      "body": "Surfaces review-only patterns and the human anchors already present in the draft."
    }
  ],
  "helpCards": [
    {
      "title": "Find what may be questioned",
      "body": "Identify citation gaps, weak source support, similarity risk, and AI-like writing signals before your instructor or review system sees them."
    },
    {
      "title": "Understand the reason",
      "body": "See whether the concern comes from missing evidence, generic phrasing, uneven source use, or overly uniform writing, instead of guessing from a single score."
    },
    {
      "title": "Revise responsibly",
      "body": "Use the report and rewrite to strengthen claims, add source grounding, clarify wording, and keep your own meaning intact."
    },
    {
      "title": "Keep a review trail",
      "body": "Download a PDF report so you have a record of what was checked and what you improved before submission."
    }
  ],
  "whyCards": [
    {
      "title": "We learn from synthetic information",
      "body": "Search engines, chatbots, and writing assistants summarise knowledge before we reach the original source. Writing can become detached from its evidence.",
      "note": "DraftProof bridges that gap"
    },
    {
      "title": "Traditional media is no longer the only source",
      "body": "Information now moves through AI newsrooms, generated summaries, and reported material. Polished does not mean proven.",
      "note": "Check the source, check the claim"
    },
    {
      "title": "AI detection alone is not enough",
      "body": "A score is not feedback. DraftProof asks better questions: is the claim supported, and what needs fixing?",
      "note": "Actionable, not just a verdict"
    }
  ],
  "contentStrategies": [
    {
      "type": "Content draft",
      "strategy": "Make the argument clearer",
      "detail": "Keep your main point and make the reasoning easier to follow."
    },
    {
      "type": "Research with citations",
      "strategy": "Improve wording without moving sources",
      "detail": "Clean up weak sentences while keeping citations and source claims in place."
    },
    {
      "type": "Technical writing",
      "strategy": "Keep exact meaning intact",
      "detail": "Protect terms, steps, definitions, and requirements from being over-polished."
    },
    {
      "type": "Legal, policy, or medical text",
      "strategy": "Make careful edits only",
      "detail": "Change only what needs attention where wording mistakes can create risk."
    },
    {
      "type": "Lists and tables",
      "strategy": "Keep the structure easy to follow",
      "detail": "Improve wording without breaking rows, bullets, comparisons, or findings."
    },
    {
      "type": "Short answers",
      "strategy": "Add context or ask for more",
      "detail": "Tell you when the text is too short to judge instead of guessing."
    },
    {
      "type": "Personal reflection",
      "strategy": "Keep your own voice",
      "detail": "Preserve your experience and viewpoint instead of making it sound generic."
    },
    {
      "type": "Marketing copy",
      "strategy": "Match the audience and format",
      "detail": "Improve the message without forcing it into an academic writing style."
    }
  ],
  "sampleStats": [
    {
      "label": "Risk tier",
      "value": "Low Risk",
      "tone": "positive"
    },
    {
      "label": "Total findings",
      "value": "50"
    },
    {
      "label": "Authorship rating",
      "value": "Good"
    },
    {
      "label": "Raw AI-style signal",
      "value": "29%",
      "tone": "positive"
    },
    {
      "label": "Writing score",
      "value": "33%",
      "tone": "accent"
    }
  ],
  "beliefs": [
    "Every AI-like sentence is not misconduct.",
    "Every similarity match is not plagiarism.",
    "Students should not be judged by black-box scores.",
    "Rewriting everything does not make writing more honest."
  ]
};
