export const landing = {
  "heroPill": "Grounding and critical-thinking review for students",
  "heroTitle": "Show where your content needs stronger evidence before anyone questions it.",
  "heroTitleHighlight": "stronger evidence",
  "heroLead": "DraftProof finds the citation gaps, weak grounding, and AI-like signals in your draft, turns them into a focused revision plan, and asks the reflective questions that keep you — not the AI — in control of the thinking before submission.",
  "runCheck": "Review my content",
  "viewSample": "View sample report",
  "trustNote": "Not a detector bypass · Not a misconduct verdict · A grounding and integrity review you can act on.",
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
  "students": "College students",
  "researchers": "University students",
  "educators": "Graduate students",
  "policyWriters": "ESL writers",
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
  "ctaSmall": "1 credit per 1,000 words · PDF report included · No bypass claims",
  "reportPreviewLabel": "DraftProof sample content review report preview",
  "reportPreviewTabsLabel": "Sample report sections",
  "reportPreviewTabs": [
    {
      "id": "aiSignal",
      "label": "AI Signal",
      "summary": "Authorship pattern"
    },
    {
      "id": "scoreProfile",
      "label": "Score Profile",
      "summary": "Why it moved"
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
  "findingsSampleId": "S004–S006",
  "findingsSampleType": "AI Likelihood",
  "findingsSampleParagraph": "In addition to economics, the United States has a strong cultural influence. American movies, music, fashion, and social media trends are consumed globally. The entertainment industry in Hollywood has become one of the most powerful cultural exports in history.",
  "findingsSampleDescription": "The paragraph uses a standard transition \"In addition\" and several other familiar phrases that make the writing sound formulaic.",
  "findingsSignalStrength": "Signal Strength",
  "findingsSampleChip1": "8 Findings In Paragraph",
  "findingsSampleChip2": "MEDIUM Priority",
  "findingsSampleChip3": "Auto-Fixable",
  "findingsAlsoDetected": "Also Detected",
  "findingsSampleAlso": "Generic Phrasing",
  "findingsMainIssue": "Main Issue to Fix",
  "findingsSampleMainIssue": "Use of generic transitional phrase and predictable wording that reduces originality.",
  "findingsRewriteHint": "Rewrite Hint",
  "findingsSampleRewriteHint": "Example: \"Beyond its economy, American cinema reaches audiences in over 190 countries each year.\"",
  "transformationPattern": "Transformation Pattern",
  "humanUncertain": "Human / uncertain pattern",
  "lowConfidence": "Low Confidence",
  "notVerdict": "Not a Verdict",
  "aiSignal": "AI Signal",
  "lowAiSignal": "Low AI Signal",
  "calibratedTopk": "41% calibrated top-k · below 20% reference",
  "originalScan": "Original Scan",
  "originalScanScore": "18%",
  "calibratedAiRisk": "Calibrated AI risk 15%",
  "humanAnchorDiscount": "Human anchor discount 38%",
  "calibrationConfidence": "Calibration confidence 61%",
  "reportingSuppression": "Reporting suppression 39%",
  "turnitinReference": "Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage because low-range results are less reliable. DraftProof scores are review signals, not verdicts.",
  "authorshipRating": "Authorship Rating",
  "good": "GOOD",
  "calibratedRisk": "11% calibrated risk",
  "estimatedContribution": "Estimated Contribution",
  "contributionBody": "Human anchoring dominates, with limited AI transformation signal.",
  "humanContribution": "Human Contribution",
  "aiTransformation": "AI Transformation",
  "scoreProfile": "Score Profile",
  "whyScoreMoved": "Why the score moved",
  "scoreProfileBody": "DraftProof groups scanner signals so you can see whether the score came from AI-like texture, weak grounding, or stronger human anchors.",
  "aiStyleSignal": "AI-style signal",
  "sourceGroundingSignal": "Source grounding",
  "humanAnchorSignal": "Human anchor",
  "sampleScoreSignals": [
    {
      "label": "AI-style risk",
      "value": "Low",
      "detail": "Calibrated after human anchors",
      "tone": "warning"
    },
    {
      "label": "Source grounding",
      "value": "Review",
      "detail": "Claims checked for support",
      "tone": "quality"
    },
    {
      "label": "Fix priority",
      "value": "Clear",
      "detail": "Highest-impact issues first",
      "tone": "positive"
    }
  ],
  "actionPlan": "Action Plan",
  "actionPlanTitle": "Fix the highest-impact issues first",
  "actionPlanBody": "The report turns scan signals into a practical review order before you revise or download the PDF.",
  "sampleActionItems": [
    {
      "title": "Add citation support",
      "body": "Two claims need clearer source backing before submission.",
      "tone": "warning"
    },
    {
      "title": "Strengthen source grounding",
      "body": "One paragraph should explain how the cited source supports the point.",
      "tone": "quality"
    },
    {
      "title": "Revise generic phrasing",
      "body": "Replace broad AI-style wording with specific reasoning and evidence.",
      "tone": "positive"
    }
  ],
  "sampleReportNotes": [
    "No single transformation pattern dominates",
    "Human anchor reduced AI certainty",
    "PDF report included"
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
      "body": "Use the report and guided revision to strengthen claims, add source grounding, clarify wording, and keep your own meaning intact."
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
