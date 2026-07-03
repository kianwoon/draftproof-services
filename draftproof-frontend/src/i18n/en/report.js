export const report = {
  "loading": "Analyzing your report...",
  "notFound": "Report not found.",
  "loadFailed": "Failed to load report",
  "back": "Back to Reports",
  "eyebrow": "Analysis Report",
  "eyebrowRewritten": "Rewritten Draft Analysis",
  "rewrittenHeroNote": "Measured on the rewritten draft — typically still flagged. This is a review draft, not a clean or final verdict.",
  "documentTitle": "Content Analysis",
  "words": "{{count}} words",
  "downloadPdf": "Download PDF",
  "downloadRewrittenPdf": "Download Rewritten PDF",
  "retentionNotice": "Available in DraftProof for 3 days. A PDF copy is also sent to your mailbox.",
  "overview": "Report overview",
  "originalScanSummary": "Original scan summary",
  "originalScanBaseline": "Original scan baseline",
  // allow-hardcode: i18n UI copy (axis/level labels + reason strings keyed by code),
  // not a scoring/matching oracle. Never compared against document text. KEEP IN SYNC
  // with detect.submission_risk presentation maps + zh/report.js.
  "submissionRisk": {
    "ariaLabel": "Submission risk",
    "kicker": "Submission risk",
    "mainReasonPrefix": "Main reason:",
    "selfDeclare": "Unknown — self-declare",
    "externalTrigger": "AI-likelihood ~{{score}}% — NOT a Turnitin score; don't compare it to the 20% line. Detectors over-flag fluent writing, so it's a heads-up, not a verdict.",
    "note": "This is about whether you can stand behind this as your own work — not whether it looks AI-written. Declaration, course policy, and group contribution aren't in the text; only you can declare those.",
    "compactNote": "AI-likelihood ~{{score}}% (composite detector) — NOT a Turnitin score; detectors over-flag fluent writing, so it's a heads-up, not a verdict. A separate beta deep-scan estimate appears in the Authorship clarity breakdown below; different detectors can disagree. Declaration, course policy, and group contribution aren't in the text — only you can declare those.",
    "compactNoteFused": "AI-likelihood ~{{score}}% (fused: composite + deep-scan detectors) — NOT a Turnitin score; detectors over-flag fluent writing, so it's a heads-up, not a verdict. The deep-scan detail appears in the Authorship clarity breakdown below. Declaration, course policy, and group contribution aren't in the text — only you can declare those.",
    "compactNoteFusedAnchored": "AI-likelihood ~{{score}}% (fused: composite + deep-scan detectors) — DraftProof's flag line is {{flagLine}}, and this is NOT a Turnitin score: do not compare it to Turnitin's 20% line. Detectors over-flag fluent writing, so it's a heads-up, not a verdict. Deep-scan detail is in the Authorship clarity breakdown below. Declaration, course policy, and group contribution aren't in the text — only you can declare those.",
    "ownershipLead": {
      "low": "You can defend this as your own work.",
      "medium": "Strengthen this before you can fully defend it as your own.",
      "high": "Strengthen this before you can fully defend it as your own."
    },
    "detectorWarning": "Heads-up: AI detectors over-flag fluent writing, so they may flag this even though it's your own work — a warning, not a verdict. Your protection is grounding your claims and keeping your drafts.",
    "levels": {
      "low": "Low",
      "medium": "Medium",
      "high": "High",
      "unknown": "Unknown"
    },
    "axes": {
      "text_pattern": "Text-pattern risk",
      "ownership": "Ownership risk",
      "citation": "Citation risk",
      "defence_readiness": "Defence-readiness risk",
      "policy_declaration": "Policy / declaration risk"
    },
    "reasons": {
      "ownership": "weak ownership of the thinking — thin judgement, reasoning, or grounding",
      "citation": "claims that are not clearly tied to a source",
      "defence_readiness": "hard to defend as your own work in an interview",
      "text_pattern": "AI-like text patterns that may trigger a detector"
    },
    // allow-hardcode: i18n UI copy for the DraftProof-scale legend table (owner-specified
    // real FPR numbers from poc/detect_v7/weights.json tier_authority._notes / display_bands —
    // human-reviewed craft text, not a scoring/matching oracle).
    "scale": {
      "toggle": "What do these numbers mean?",
      "headers": {
        "score": "DraftProof score",
        "reads": "Reads as",
        "measured": "What we measured"
      },
      "rows": {
        "low": {
          "reads": "Low",
          "measured": "~6% or fewer real ESL students score this high — most human writing lands well under this"
        },
        "medium": {
          "reads": "Medium",
          "measured": "uncommon for genuine human writing (0.4% false-positive rate measured)"
        },
        "high": {
          "reads": "High",
          "measured": "rare for genuine human writing (<1% false-positive rate measured)"
        },
        "critical": {
          "reads": "Critical",
          "measured": "essentially never seen in genuine human writing in our testing (0% false-positive rate measured)"
        }
      },
      "notTurnitinComparable": "DraftProof's score is not comparable to Turnitin's percentage — the two tools measure different things on different scales."
    }
  },
  "repairSummary": {
    "ariaLabel": "Repair summary",
    "kicker": "Repair Summary",
    "mainRisk": "Main risk",
    "statusFallback": "Review needed",
    "mainRiskFallback": "Review the highlighted sections before submission.",
    "nextActionHighlighted": "Fix {{count}} highlighted paragraph(s), then re-scan the edited draft.",
    "nextActionFallback": "Review the draft, add concrete evidence where needed, then re-scan.",
    "confidenceNote": "DraftProof reports writing signals, not a verdict. Use this as a repair plan before submission.",
    "editDraftHint": "Edit highlighted sections before you re-scan."
  },
  "groundingDiagnosis": {
    "primaryDriver": "Primary driver",
    "drivers": {
      "concrete_grounding": { "label": "Grounding gap", "action": "Add concrete anchors, named evidence, and specifics." },
      "authorship_trace": { "label": "Authorship trace", "action": "Show your decision trail and draft process." },
      "llm_patterning": { "label": "LLM patterning", "action": "Vary your structure and phrasing." },
      "language_texture": { "label": "Language texture", "action": "Cut padding and add concrete meaning." }
    },
    // allow-hardcode: i18n bar labels keyed by bucket code (presentation), never
    // matched against document text. KEEP IN SYNC with render.py _DIAG_BUCKET_LABELS.
    "buckets": {
      "concrete_grounding": "Grounding gap",
      "authorship_trace": "Authorship uncertainty",
      "llm_patterning": "AI-like patterning",
      "language_texture": "Generic language texture"
    },
    "bucketsHeading": "Risk contributors",
    "lowerIsBetter": "Lower is better",
    "tentative": "limited text — diagnosis tentative"
  },
  // allow-hardcode: i18n UI copy (band labels + coaching strings keyed by code),
  // not a scoring/matching oracle — same category as groundingDiagnosis above.
  // Never compared against document text.
  "criticalThinking": {
    "title": "Critical Thinking",
    "questionsTitle": "Questions to sharpen your thinking",
    "questionsIntro": "These are about your own draft. Use them to push your thinking further as you revise — the answers are yours to write.",
    "leadPrefix": "Work on",
    "strongHeading": "Strong student control",
    "scoreLabel": "Control score",
    "tentative": "limited text — diagnosis tentative",
    "bands": {
      "strong_control": "Strong student control",
      "acceptable_control": "Acceptable control",
      "weak_control": "Weak control",
      "high_dependency": "High dependency risk",
      "very_high_dependency": "Very high dependency risk",
      "insufficient_data": "Not enough text to assess"
    },
    "dimensions": {
      "specific_context": { "label": "Specific context", "action": "Anchor claims to a real assignment, case, example, or observation." },
      "student_judgement": { "label": "Student judgement", "action": "Take a position and justify why you chose it." },
      "reasoning_trail": { "label": "Reasoning trail", "action": "Show how you moved from evidence to conclusion." },
      "ai_dependency": { "label": "AI dependency", "action": "Challenge the first answer instead of keeping it as-is." },
      "evidence_grounding": { "label": "Evidence grounding", "action": "Connect each claim to a source, example, or data point." }
    },
    "reviewFlagsTitle": "Review flags (not part of the score)",
    "llmDimensions": {
      "alternative_comparison": "Alternative comparison",
      "reflection": "Reflection / ownership"
    },
    // allow-hardcode: i18n UI copy — display labels keyed by the LLM output-enum
    // code (ALLOWED_LABELS in critical_thinking_llm.py), never matched against text.
    "highlightLabels": {
      "single_path_answer": "Single-path answer",
      "no_judgement": "No judgement",
      "no_reflection": "No reflection",
      "broad_claim": "Broad claim",
      "missing_evidence": "Missing evidence",
      "over_polished_closure": "Over-polished closure"
    }
  },
  "whatToFixFirst": {
    "kicker": "Repair Plan",
    "title": "What to fix first",
    "paragraphFallbackTitle": "Review this highlighted paragraph",
    "authorshipFallbackTitle": "Strengthen authorship evidence",
    "authorshipFallbackBody": "Add concrete details that only you can verify.",
    "exampleTitle": "Use a stronger example {{count}}",
    "reviewTitle": "Review the highlighted draft",
    "reviewBody": "Open the paragraph notes and strengthen the sections that are driving the report.",
    "openParagraph": "Open paragraph"
  },
  "summary": {
    "riskTier": "Risk Tier",
    "lowConfidence": "Low confidence — the score sits near a band boundary or the sample is short, so treat this tier as provisional and review before relying on it.",
    "totalFindings": "Total Findings",
    "authorshipRating": "Authorship Rating",
    "rawAiSignal": "Raw AI-Style Signal",
    "writingScore": "Writing Score"
  },
  "tiers": {
    "low": "Low Risk",
    "moderate": "Moderate Risk",
    "high": "High Risk",
    "green": "Low Risk",
    "amber": "Moderate Risk",
    "orange": "High Risk",
    "red": "Critical Risk"
  },
  "severities": {
    "critical": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "info": "INFO"
  },
  "transformation": {
    "scorecard": "Transformation pattern scorecard",
    "kicker": "Writing-signal pattern",
    "originalVsRewritten": "Original vs rewritten pattern",
    "patternAnalysis": "Pattern analysis",
    "rewrittenOutcome": "Rewritten Outcome",
    "rewrittenAiSignal": "Rewritten AI Signal",
    "aiSignal": "AI Signal",
    "notRated": "Not Rated",
    "confidence": "{{value}} confidence",
    "rewriteComparison": "Rewrite Comparison",
    "notVerdict": "Not A Verdict",
    "originalScan": "Original Scan",
    "rewrittenScan": "Rewritten Scan",
    "originalPatternFallback": "Pattern analysis",
    "rewrittenPatternFallback": "Rewritten contribution pattern",
    "originalRating": "Original Rating",
    "rewrittenRating": "Rewritten Rating",
    "verdict": {
      "stillFlagged": "Still flagged by detectors",
      "riskRemains": "Detector risk remains",
      "riskLow": "Detector risk low",
      "reviewed": "Rewrite reviewed",
      "grounded": "Grounded draft — review",
      "groundingDetail": "grounding improved — review and finish in your own words"
    },
    "estimatedContribution": "Estimated Contribution",
    "calibratedAiRisk": "Calibrated AI risk {{value}}%",
    "humanAnchorDiscount": "Human anchor discount {{value}}%",
    "calibrationConfidence": "Calibration confidence {{value}}%",
    "reportingSuppression": "Reporting suppression {{value}}%",
    "smartSignalsLabel": "Score signal gauges",
    "smartSignalsValue": "{{label}} {{value}}%",
    "smartSignals": {
      "aiRisk": "AI Risk",
      "humanAnchor": "Human Anchor",
      "confidence": "Confidence",
      "suppression": "Suppression"
    },
    "contributionEstimate": "{{variant}} human contribution versus AI transformation estimate",
    "original": "Original",
    "rewritten": "Rewritten",
    "humanContribution": "Human Contribution",
    "aiTransformation": "AI Transformation",
    "signalProfile": "Signal Profile",
    "signals_one": "{{count}} signal",
    "signals_other": "{{count}} signals",
    "improvedFromTo": "Improved from {{from}}% to {{to}}%.",
    "notPresent": "Signal not present in this scan.",
    "summaryMixed": "Mixed authorship pattern: human anchoring and AI transformation signals are both visible.",
    "summaryAiDominates": "AI transformation dominates this scan pattern.",
    "summaryAiStronger": "AI transformation signals are stronger than the human anchor.",
    "summaryHumanStronger": "Human contribution remains the stronger signal.",
    "mainDrivers": "Main drivers: {{drivers}}.",
    "rewrittenContributionEstimate": "Rewritten contribution estimate from the completed rewrite scan.",
    "noSinglePattern": "No single transformation pattern dominates",
    "turnitinReferenceSummary": "Why a low external estimate may show as ~%",
    "turnitinReferenceNote": "Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage because low-range results are less reliable.",
    // allow-hardcode: i18n display labels for transformation pattern CODES
    // (presentation), never matched against document text. KEEP IN SYNC with
    // poc/detect/transformation.py _LABELS.
    "labels": {
      "fully_ai_written": "Fully AI-written pattern",
      "ai_cleaned_human_writing": "AI-cleaned human writing pattern",
      "ai_paraphrased": "AI-paraphrased source pattern",
      "ai_expanded": "AI-expanded outline pattern",
      "ai_stitched_patchwork": "Patchwork flow detected",
      "ai_cited_weakly_grounded": "AI-cited but weakly grounded pattern",
      "human_uncertain": "Human / uncertain pattern"
    },
    "subtitles": {
      "ai_stitched_patchwork": "Some sections may feel joined from separate points. Add clearer bridges and source anchors."
    },
    "confidenceLevels": {
      "low": "Low",
      "moderate": "Moderate",
      "high": "High"
    },
    "confidenceBadge": {
      "low": "Low confidence — review, not verdict",
      "moderate": "Moderate confidence",
      "high": "High confidence"
    },
    "confidenceTooltip": "This signal is weak. Treat it as a writing-quality clue, not an authorship judgement.",
    "evidence": {
      "no_single_transformation_pattern_dominates": "no single transformation pattern dominates",
      "human_anchor_reduced_ai_certainty": "human anchor reduced AI certainty"
    }
  },
  "scoreProfile": {
    "sectionLabel": "Score profile",
    "kicker": "Score Profile",
    "title": "Why the score moved",
    "titleOriginal": "What's driving the score",
    "summaryRewrite": "The rewrite is compared against the original scan so you can see which signal groups improved and which still need review.",
    "summaryOriginal": "DraftProof groups the score drivers so you can see which signals had the most influence on the report.",
    "trackedSignals": "Tracked signals",
    "noSignal": "No signal",
    "groupFallback": "Related scanner signals used to explain the score.",
    "leadingSignal": "Leading signal {{value}}%",
    "directionLower": "Lower is better",
    "directionHigher": "Higher is better",
    "improvedSignals": "{{count}} improved",
    "viewDetails": "View detailed score signals",
    "viewDetailsHint": "Original and rewritten bars are shown together when rewrite data is available.",
    "originalToRewritten": "Original {{original}}% -> rewritten {{rewritten}}%"
  },
  "signalGroups": {
    "ai_authorship": {
      "label": "AI Authorship Signals",
      "description": "Pattern, predictability, and model-like texture signals."
    },
    "human_authenticity": {
      "label": "Human / Authenticity Signals",
      "description": "Grounding and anchor signals that reduce authorship certainty."
    },
    "quality_calibration": {
      "label": "Quality & Calibration Signals",
      "description": "Evidence quality, calibration confidence, and reporting caution."
    },
    "other": {
      "label": "Other Signals",
      "description": ""
    }
  },
  "signals": {
    "fallbackDescription": "Scanner signal used to interpret the transformation pattern.",
    "labels": {
      "topk_pattern": "Raw Top-k Predictability",
      "topk_pattern_raw": "Raw Top-k Predictability",
      "topk_calibrated_risk": "Calibrated Top-k Risk",
      "ai_likelihood": "AI likelihood",
      "adjusted_ai_risk": "Adjusted AI risk",
      "calibrated_ai_risk": "Calibrated AI risk",
      "human_anchor_score": "Human anchor",
      "human_anchor_discount": "Human anchor discount",
      "rewrite_smoothness": "Rewrite smoothness",
      "semantic_uniformity_risk": "Semantic uniformity",
      "discourse_regularity_risk": "Discourse regularity",
      "source_similarity": "Source similarity",
      "surface_similarity": "Surface similarity",
      "paraphrase_transformation_risk": "Paraphrase transformation",
      "outline_to_text_expansion": "Expansion pattern",
      "section_style_variance": "Patchwork variance",
      "grounding_quality_risk": "Grounding risk",
      "citation_grounding_risk": "Citation grounding",
      "signal_agreement_score": "Signal agreement",
      "calibration_confidence": "Calibration confidence",
      "reporting_suppression": "Reporting suppression",
      "predictability": "Predictability",
      "writing_quality": "Writing quality",
      "genericity": "Genericity",
      "ai_signal_deberta": "Learned-classifier AI signal"
    },
    "descriptions": {
      "topk_pattern": "How often the writing chooses very expected words. A very high score means the wording follows common predictable paths. Lower is better.",
      "topk_pattern_raw": "How often the writing chooses very expected words. A very high score means the wording follows common predictable paths. Lower is better.",
      "topk_calibrated_risk": "How predictable the wording looks after DraftProof adjusts the raw score. Very high scores can make writing look machine-smoothed. Lower is better.",
      "ai_likelihood": "Overall AI-style writing pattern detected in the text. This is a signal, not proof of authorship. Lower is better.",
      "adjusted_ai_risk": "AI-style risk after DraftProof accounts for human details and other context. Lower is better.",
      "calibrated_ai_risk": "The final authorship-risk score after DraftProof applies cautious checks. Lower is better.",
      "human_anchor_score": "Concrete personal, local, or experience-based details that make the writing feel more human-authored. Higher is better.",
      "human_anchor_discount": "How much the human details reduce AI concern. Higher means the writing has stronger human evidence.",
      "rewrite_smoothness": "How polished and even the writing flow appears. Higher scores may suggest the text was heavily smoothed. Lower is better.",
      "semantic_uniformity_risk": "How evenly the ideas move from paragraph to paragraph. Very even pacing can look less natural. Lower is better.",
      "discourse_regularity_risk": "How predictable the paragraph flow feels. Very regular flow can make the writing look templated. Lower is better.",
      "source_similarity": "How close the meaning is to known source material. Higher scores may suggest heavy reliance on a source.",
      "surface_similarity": "How close the wording is to known source material. Higher scores may suggest copied or lightly changed text.",
      "paraphrase_transformation_risk": "Whether source meaning appears to be kept while the wording is heavily changed. Lower is better.",
      "outline_to_text_expansion": "Whether short ideas appear expanded into fuller writing in a structured way. Lower is better.",
      "section_style_variance": "Whether sections look stitched together or written in different passes. Lower is better.",
      "grounding_quality_risk": "How well claims are grounded in concrete, lived, specific detail (the grounding the rewrite actually improves). Lower is better. Separate from the citation row, which only your real sources can fix.",
      "citation_grounding_risk": "Whether claims are backed by real citations or sources. The rewrite cannot add real citations for you, so this stays high until you add your own — it is separate from the concrete, lived-detail grounding the rewrite improves. An integrity signal, not proof of AI.",
      "signal_agreement_score": "How much the different DraftProof checks point in the same direction. Higher means the result is more consistent.",
      "calibration_confidence": "How confident DraftProof is that the available evidence is strong enough to report. Higher means more confidence.",
      "reporting_suppression": "DraftProof is being cautious because some evidence is uncertain or not strong enough. Higher means more caution was applied.",
      "ai_signal_deberta": "Heatmap of a second, learned AI-writing detector's per-passage score. Graduated: clean (human-like), mild, strong, and high-confidence (>=99%). Advisory only — it over-flags fluent/ESL writing at the high end, so review the passage rather than treating the color as a verdict."
    }
  },
  "authorship": {
    "insufficient_sample": {
      "label": "Too Short to Assess",
      "short": "Too Short"
    },
    "ai_generated_signals": {
      "label": "AI-Generated / AI-Paraphrased Signals",
      "short": "AI Signals"
    },
    "likely_ai": {
      "label": "Likely AI-Assisted",
      "short": "Likely AI-Assisted"
    },
    "possible_ai_assisted": {
      "label": "Possible AI-Assisted",
      "short": "Possible AI-Assisted"
    },
    "unlikely_ai": {
      "label": "Unlikely AI-Assisted",
      "short": "Unlikely AI-Assisted"
    },
    "low_ai_signal": {
      "label": "Good",
      "short": "Good"
    },
    "low_signal": {
      "label": "Low AI Signal",
      "short": "Low Signal"
    },
    "likely_ai_fallback": {
      "label": "Likely AI",
      "short": "Likely AI"
    }
  },
  "seal": {
    "wordsNotEnough": "{{count}} words · not enough text",
    "sentencesNotEnough": "{{count}} sentences · not enough text",
    "notEnough": "Not enough text",
    "wordsSampleLimited": "{{count}} words · sample limited",
    "sentencesSampleLimited": "{{count}} sentences · sample limited",
    "sampleLimited": "Sample limited",
    "calibratedTopk": "{{value}} calibrated top-k",
    "topk": "{{value}} top-k",
    "calibratedRisk": "{{value}} calibrated risk",
    "belowReference": "below 20% reference",
    "thresholdExceeded": "review threshold exceeded",
    "rawSignal": "{{value}} raw signal"
  },
  "aiSignalStamp": {
    "low": "Low AI Signal",
    "review": "AI Review",
    "likely": "Likely AI",
    "high": "High AI Signal"
  },
  "rewrite": {
    "queued": "Queued",
    "processing": "Rewriting AI sections",
    "retrying": "Retrying rewrite",
    "complete": "Rewrite complete",
    "canceled": "Rewrite canceled",
    "failed": "Rewrite failed",
    "checkingFailed": "Failed to check rewrite status",
    "startFailed": "Failed to start rewrite",
    "cancelFailed": "Failed to cancel rewrite",
    "queuing": "Queuing rewrite",
    "noRewriteableTitle": "No rewriteable AI sections",
    "noRewriteableMessage": "This report only has review-only signals. There is nothing DraftProof can rewrite automatically.",
    "progressMessages": [
      "Rewriting flagged passages while preserving the original meaning.",
      "Checking revised text against the source draft.",
      "Improving clarity, specificity, and academic tone.",
      "Preparing highlighted rewrite results for this report."
    ],
    "waiting": "Waiting for the rewrite worker to start.",
    "retryingDetail": "The rewrite worker is retrying this step automatically.",
    "checking": "Still checking the rewrite status every few seconds.",
    "longestStep": "This is usually the longest step. Longer reports can stay here while DraftProof rewrites and verifies flagged sections.",
    "cancelTitle": "Cancel this rewrite?",
    "cancelMessage": "DraftProof will stop tracking this rewrite, release the reserved credits, and ignore any late worker result for this job.",
    "canceling": "Canceling...",
    "cancelRewrite": "Cancel rewrite",
    "starting": "Starting rewrite...",
    "resume": "Resume Rewrite",
    "rewriteAiSections": "Auto Rewrite / Correction",
    "tokenEstimate": "{{tokens}} required for this rewrite, based on {{words}} submitted words.",
    "emailPdfNotice": "We will email a PDF copy of the rewritten result to your account email when it is ready.",
    "keepOpenInline": "Keep this report open; results will appear when ready.",
    "done": "Done",
    "elapsed": "Elapsed: {{elapsed}}",
    "keepOpen": "Keep this page open; results will appear automatically.",
    "emailPdfProgress": "A rewritten-content PDF copy will also be sent to your account email.",
    "notificationTitle": "Rewrite complete",
    "notificationBody": "Your DraftProof rewrite is ready to review.",
    "completion": "Rewrite completion",
    "accepted": "Grounding draft accepted",
    "topkBlocked": "Top-k blocker remains",
    "originalPreserved": "Original preserved",
    "completeTitle": "Rewrite complete",
    "acceptedDetail": "A revision candidate improved the target signals and was kept for review.",
    "topkBlockedDetail": "Top-k predictability remains high, so the author should review and finish the wording manually.",
    "originalPreservedDetail": "No revision candidate improved the target signals enough to keep; review the guidance before trying again.",
    "selectedDetail": "A revision candidate was selected after scanning.",
    "finishedDetail": "Rewrite finished and the result is ready to review.",
    "aiBefore": "Rewrite AI authorship before",
    "aiAfter": "Rewrite AI authorship after",
    "humanShift": "Human Shift Score",
    "humanContribution": "Rewrite human contribution",
    "aiTransformation": "Rewrite AI transformation",
    "groundingRisk": "Rewrite grounding risk",
    "viewResult": "View Rewrite Result"
  },
  // allow-hardcode: i18n UI strings for the per-paragraph severity bar, not a detect/scoring word-list.
  "severityBar": {
    "caption": "Finding density by document",
    "ariaLabel": "Finding-severity heatmap across {{count}} paragraphs",
    "tooltip": "Paragraph {{index}}: {{count}} finding(s), worst: {{tier}}",
    "tooltipClean": "Paragraph {{index}}: no findings",
    "legendLow": "Clean / fewer findings",
    "legendHigh": "Denser / more severe",
    "tier": {
      "critical": "critical",
      "high": "high",
      "medium": "medium",
      "low": "low",
      "info": "info"
    }
  },
  "submitted": {
    "sectionLabel": "Submitted content with scan signals",
    "kicker": "Submitted Content",
    "title": "Signal highlights",
    "highlightedSections": "highlighted sections",
    "tabIssues": "Issues · {{count}}",
    "tabDocument": "Read full document",
    "moreDetail": "＋ More detail",
    "lessDetail": "− Less detail",
    "issuesEmptyTitle": "No major issues detected",
    "issuesEmptyBody": "Nothing is driving the report score. Open \"Read full document\" to review the full text.",
    "legend": "Signal color legend",
    "documentText": "Submitted document text",
    "selectedSignal": "Selected signal detail",
    "paragraphRepairTitle": "Paragraph repair note",
    "signalBadge": "Signal: {{value}}",
    "position": "{{current}} of {{total}} highlighted sections",
    "positionShort": "{{current}}/{{total}}",
    "editParagraph": "Edit this paragraph",
    "copyGuidance": "Copy guidance",
    "previousIssue": "Previous issue",
    "nextIssue": "Next issue",
    "cleanParagraphTitle": "No major issue detected",
    "cleanParagraphBody": "This paragraph is not driving the report score. You can still edit it normally before re-scanning.",
    "signalStrength": "{{value}}% signal strength",
    "signalStrengthLabel": "Signal strength",
    "priority": "{{value}} priority",
    "paragraphSignals": "{{count}} findings in paragraph",
    "alsoDetected": "Also detected",
    "recommendation": "How to improve this paragraph",
    "criticalThinking": "Critical thinking",
    "flaggedSentences": "Sentences the classifier flagged",
    "noSignal": "No highlighted signal",
    "mapReady": "Content map ready",
    "mapReadyBody": "The submitted text is available for review. New scans include richer signal highlights for each affected paragraph.",
    "editor": {
      "editDraft": "Manual Rewrite / Correction",
      "kicker": "Revision workspace",
      "title": "Edit submitted content",
      "rewriteTitle": "Edit rewritten draft",
      "priorScanNotice": "Signals shown here came from the previous scan. Re-scan the edited draft to refresh findings.",
      "rewriteNotice": "You're editing the AI rewrite — refine it into your own words, then re-scan to refresh findings.",
      "close": "Close",
      "noDraft": "No local draft",
      "saving": "Saving draft...",
      "saved": "Draft saved {{value}}",
      "saveError": "Draft could not be saved",
      "documentEditor": "Editable document",
      "changed": "Edited draft, needs re-scan",
      "unchanged": "No edits yet",
      "discardDraft": "Discard draft",
      "rescanDraft": "Re-scan edited version",
      "rescanning": "Scanning...",
      "emptyDraft": "Add text before re-scanning.",
      "rescanQueueing": "Queuing edited scan...",
      "rescanProcessing": "Scanning edited draft...",
      "rescanFailed": "Edited scan failed. Please try again.",
      "rescanTimedOut": "The edited scan is taking longer than expected. Please check Reports in a moment.",
      "translateCnEn": "中文 → English",
      "translateHint": "Writing in Chinese? Highlight it, then click 中文 → English to draft in English.",
      "translating": "Translating…",
      "undoTranslate": "Undo translate",
      "translateSelectFirst": "Select the text you want to translate first.",
      "translateError": "Translation failed. Please try again in a moment.",
      "translateNote": "Machine translation is a starting point — edit it into your own words before re-scanning.",
      "trackedPreview": "Tracked changes",
      "trackedPreviewBody": "Inserted text is highlighted; removed text is struck through.",
      "copyTrackedChanges": "Copy",
      "copiedTrackedChanges": "Copied",
      "copyTrackedChangesFailed": "Copy failed",
      "amberGuideKicker": "Edit yourself",
      "amberGuideHeading": "Fix the amber sentences",
      "amberGuideCopy": "These sentences were kept as-is — the rewrite couldn't safely improve them without inventing facts, so grounding them is yours to do. They're highlighted amber in your draft; click one to jump to it, then rewrite it in your own words with concrete, specific detail. The worked examples below show the move.",
      "amberListHeading": "Kept as-is — rewrite these",
      "amberItemLabel": "Amber sentence",
      "amberJumpHint": "Click to jump to this sentence in your draft",
      "amberResolved": "Edited — no longer flagged",
      "affectedParagraphs": "Affected paragraphs",
      "affectedParagraph": "Affected paragraph",
      "paragraphUnchanged": "Paragraph unchanged",
      "paragraphEdited": "Paragraph edited or moved",
      "signal": "Signal"
    }
  },
  "ok": "OK",
  // allow-hardcode: i18n UI copy for the two policy scores (labels/levels/issues/fixes
  // keyed by code) — presentation, never matched against text. KEEP IN SYNC with
  // poc/report/render.py _POLICY_* maps + zh/report.js.
  "policyRisk": {
    "heading": "Policy risk",
    "subheading": "How this draft may read under your school's AI policy",
    "allowedLabel": "If AI is allowed (with declaration)",
    "restrictedLabel": "If AI is not allowed",
    "mainIssuePrefix": "Main issue:",
    "bestFixPrefix": "Best fix:",
    "confirmTitle": "Confirm yourself to lower a score",
    "confirmAllowed": "I can explain how AI was used",
    "confirmRestricted": "I have drafts / notes for this work",
    "disclaimer": "These scores do not prove AI use. They estimate how risky the draft may look under different school policies.",
    "levels": {
      "low": "Low", "moderate": "Moderate", "high": "High", "severe": "Severe", "unknown": "Unknown"
    },
    "issues": {
      "grounding_gap": "weak source grounding",
      "judgment_gap": "limited own judgment",
      "prompt_specificity_gap": "generic, not tied to the task",
      "surface_ai_text_signal": "polished, AI-like surface style",
      "authorship_voice_gap": "limited personal authorship voice"
    },
    "fixes": {
      "grounding_gap": "Tie claims to real sources and named evidence.",
      "judgment_gap": "Show your decisions, comparisons, and critique.",
      "prompt_specificity_gap": "Anchor to the assignment, module, and concrete details.",
      "surface_ai_text_signal": "Vary structure and phrasing; add your own voice.",
      "authorship_voice_gap": "Add your personal reasoning path and draft choices."
    }
  },
  "aiLikelihood": {
    "title": "AI-writing signal",
    "mainFixLabel": "Main thing to fix",
    // allow-hardcode: scoped-verdict presentation phrases keyed by tier/band code
    // (display only, never matched against text). KEEP IN SYNC with render.py
    // _SIGNAL_PHRASE / _FLAG_PHRASE.
    "verdictSignal": {
      "green": "Low AI-writing signal",
      "amber": "Moderate AI-writing signal",
      "orange": "Elevated AI-writing signal",
      "red": "Elevated AI-writing signal"
    },
    "verdictFlag": {
      "low": "low on our scale — but detectors over-flag fluent writing, so they may still flag it (a warning, not a verdict)",
      "elevated": "may draw external-detector attention (not a Turnitin score)",
      "high": "likely to draw external-detector attention (a warning, not a verdict)"
    },
    "verdictFix": "The main writing issue to fix is the {{driver}}.",
    "draftproof": "DraftProof (conservative)",
    "draftproofNote": "Conservative AI-style estimate — mostly writing fluency that grounding won't fully move. The grounding fix above is the real thing to act on.",
    "external": "Turnitin / external (estimated)",
    "externalBand": {
      "low": "unlikely to be flagged",
      "elevated": "possibly flagged",
      "high": "likely to be flagged"
    },
    "externalUnavailable": "External estimate unavailable — re-scan to populate.",
    "whyDiffer": "This is a calibrated estimate of how strict third-party detectors (Turnitin, GPTZero) may rate your text — based on limited data and imperfect, since these detectors over-flag fluent writing (even genuine human writing). Treat it as a heads-up, not a verdict; your safeguard is grounding the content and finishing it in your own words.",
    "externalDemoted": "Third-party AI detectors (Turnitin, GPTZero) are imperfect and over-flag fluent writing — including genuine human writing — so a predicted score is not a reliable signal of your result, and we don't surface one here. Your safeguard is grounding the content, finishing it in your own words, and keeping your drafts as authorship evidence.",
    "calibrateHeading": "How DraftProof calibrates this"
  },
  // allow-hardcode: i18n display labels — UI strings keyed by dimension name, not a scoring/matching oracle
  "authenticityDashboard": {
    "ariaLabel": "Authenticity dashboard",
    "title": "Authenticity",
    "na": "—",
    "overall": "Overall",
    "ciTentative": "AI-assistance estimate {{low}}–{{high}} (tentative — not a detector verdict)",
    "ciUnavailable": "AI-assistance range unavailable (not enough text to estimate)",
    "note": "These dimensions overlap and are guidance for your review, not independent measurements or a Turnitin prediction.",
    "bands": { "Low": "Low", "Moderate": "Moderate", "Medium": "Medium", "High": "High" },
    "tiles": {
      "learning_ownership": "Learning Ownership",
      "grounding": "Grounding",
      "citation_quality": "Citation Quality",
      "reasoning_consistency": "Reasoning Consistency (coming soon)",
      "ai_assistance": "AI Assistance"
    },
    // allow-hardcode: i18n display copy keyed by dimension name — UI caveat strings, not a scoring/matching oracle
    "caveats": {
      "learning_ownership": "How much you steer the thinking — derived from the critical-thinking signal.",
      "grounding": "Tentative — short submission.",
      "reasoning_consistency": "Coming in a later release."
    }
  },
  "debertaSignal": {
    "ariaLabel": "Second-opinion AI signal",
    "title": "Second-opinion AI signal",
    "scoreLabel": "of passages read as AI-like under a separate detector",
    "unavailable": "Second-opinion detector not available for this scan.",
    "aboveFloor": "{{pct}}% of passages score at the detector's high-confidence AI threshold. Advisory only — post-hoc detectors miss paraphrased text and carry residual ESL bias. Review the flagged passages rather than treating the number as a verdict.",
    "belowFloor": "Below our 20% reliability floor, so this signal offers no overall verdict. We set a high bar to avoid unfairly flagging fluent ESL writing — other detectors with a lower bar may read the flagged passages (and the document) differently. Treat 'no verdict' as 'review the flagged passages', not as 'clean'.",
    "passagesFlaggedForReview": "{{n}} of {{total}} passages flagged for review",
    "flaggedPassages": "Flagged passages",
    "reviewPassages": "Review these passages",
    "noteAgree": "Both detection methods flag this document for review.",
    "noteDisagree": "The two detection methods read this document differently. This is common — they use different signals. Review the flagged passages yourself."
  },
  "authorshipBreakdown": {
    "ariaLabel": "Authorship clarity breakdown",
    "title": "Authorship clarity breakdown",
    "categories": {
      "student_owned": "Student-owned",
      "ai_assisted_polished": "AI-assisted / polished",
      "ai_paraphrased": "AI-paraphrased",
      "ai_generated_like": "AI-generated-like"
    },
    "bands": {
      "Strong": "Strong",
      "Some": "Some",
      "Little": "Little",
      "None": "None"
    },
    "lowConfidenceCaveat": "Mixed signals in this document — treat this breakdown as a starting point for review, not a verdict.",
    "betaChip": "Beta",
    // allow-hardcode: i18n UI copy strings (labels/band names), not a scoring or matching list.
    "fusedHeadline": {
      "label": "DraftProof AI-likelihood",
      "bands": {
        "low": "Low",
        "moderate": "Moderate",
        "high": "High",
        "critical": "Critical"
      },
      "evidence": "Behind this score: composite detector {{composite}}%, deep-scan detector {{deepScan}}% (sentence-level). See the DraftProof scale above."
    },
    "deepScan": {
      "label": "Deep-scan AI estimate",
      "insufficientChip": "insufficient evidence",
      "notTurnitin": "sentence-level signal — not a Turnitin score",
      "bandDefersToTier": "{{band}} signal · {{tier}} overall",
      "bands": {
        "amber": "Amber",
        "orange": "Orange",
        "red": "Red"
      }
    },
    "subtitle": "How this document's writing signals distribute across four authorship styles. The shares always add up to 100% — a composition of the mix, not an AI-probability. The deep-scan estimate below comes from a separate beta detector and may differ from Text-pattern risk in the summary above — different models, and both are signals rather than verdicts.",
    "uncertaintyFlags": {
      "deep_scan_uncalibrated": "Includes an early-stage detector signal that is still being validated.",
      "deep_scan_below_reliability_floor": "The deep-scan reading on this document is below the reliability floor — treat it as weak evidence.",
      "paraphrase_without_original_draft": "Paraphrase detection is limited without an original draft to compare against."
    },
    "feedbackPrompt": "This breakdown is in beta. Does it match your sense of the document?",
    "feedbackAction": "Tell us",
    "disclaimer": "DraftProof provides authorship clarity signals and writing-risk analysis. It does not determine misconduct. Final judgement belongs to the school, teacher, or relevant academic policy."
  },
  "advancedSignals": {
    "summary": "Advanced signals"
  }
};
