"""Predictability detector — wraps the token-level scanner into BaseDetector.

Identifies sentences whose token-level probability follows common or formulaic
language patterns under the reference model. This may indicate generic writing,
templated phrasing, or weak specificity. It is not evidence of AI authorship.
"""

import re
import sys
import os

from .base import BaseDetector, DetectResult, Finding, classify_finding_category

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_BIBLIO_HEADING_RE = re.compile(
    r"(?m)^(?:references|bibliography|works?\s+cited|reference\s+list|sources)\s*$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+")
_DOI_RE = re.compile(r"\bdoi:\s*\S+", re.I)
_MIN_BODY_WORDS = 8

# Generic metadata/header lines to skip (works for assignments, reports, articles)
_HEADER_LINE_RE = re.compile(
    r"^(?:"
    r"word\s+count\s*[:.]?"             # Word count: 799
    r"|question\s+\d+"                  # Question 1, Q2
    r"|q\d+\s*[\.\:]"                   # Q1. or Q1:
    r"|(?:tutor-?marked|tma|assignment|semester|module|course|student)\s"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s+semester"
    r")",
    re.IGNORECASE,
)


def _pre_filter_body(content: str) -> str:
    """Strip reference sections, URLs, DOIs, and short/header/metadata fragments.

    Designed to be content-type agnostic: works for academic essays, reports,
    articles, and general prose. Only removes non-prose noise that would
    pollute predictability scoring.
    """
    # Remove everything after "References" / "Bibliography" heading
    match = _BIBLIO_HEADING_RE.search(content)
    if match:
        content = content[:match.start()]
    # Remove URLs and DOIs
    content = _URL_RE.sub("", content)
    content = _DOI_RE.sub("", content)
    # Remove lines that look like metadata, headers, or labels
    lines = content.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        # Skip known metadata/header patterns
        if _HEADER_LINE_RE.match(stripped):
            continue
        # Skip very short fragments that aren't sentences (< 4 words, no period)
        # Catches initials like "H.", "R.", labels, etc.
        if len(stripped.split()) <= 3 and not stripped.endswith((".", "!", "?")):
            continue
        filtered.append(line)
    return "\n".join(filtered)


class PredictabilityDetector(BaseDetector):

    def __init__(self, model_name: str = "gpt2-medium"):
        self._model_name = model_name
        self._scanner = None

    @property
    def name(self) -> str:
        return "predictability"

    @property
    def detector_version(self) -> str:
        return "0.2.0"

    @property
    def model_name_prop(self) -> str:
        return self._model_name

    @property
    def policy_message(self) -> str:
        return (
            "This signal does not determine AI authorship. "
            "It highlights formulaic or generic phrasing that may benefit from "
            "stronger source grounding or more specific language."
        )

    @staticmethod
    def _classify_predictability_subtype(sentence: str) -> str:
        """Classify the *reason* a sentence is predictable, not just that it is.

        Returns a subtype string used as finding_type metadata. The root
        finding_type stays {high,medium}_predictability for compatibility;
        the subtype is added to metadata so the rewrite module knows the
        real issue.
        """
        s = sentence.lower().strip()

        # Concluding/summary sentences
        conclusion_markers = [
            "in conclusion", "to summarize", "to sum up", "in summary",
            "overall,", "ultimately,", "perhaps", "should embrace",
            "it is clear that", "it is evident that",
        ]
        if any(m in s for m in conclusion_markers):
            return "formulaic_conclusion"

        # Template personal reflections
        personal_markers = [
            "i believe that", "i think that", "in my opinion",
            "i feel that", "personally,", "from my perspective",
            "a teacher's role should", "a student should",
        ]
        if any(m in s for m in personal_markers):
            return "template_personal_reflection"

        # Generic policy/action claims
        policy_markers = [
            "should embrace these tools", "guide students on how to",
            "use them properly", "important to note that",
            "needs to be addressed", "must be taken into account",
        ]
        if any(m in s for m in policy_markers):
            return "generic_policy_claim"

        # Broad education/academic claims
        edu_markers = [
            "plays an important role in", "has transformed the way",
            "increasingly important in today's", "rapidly evolving",
            "has gained significant attention",
        ]
        if any(m in s for m in edu_markers):
            return "broad_education_claim"

        return "statistical_predictability"

    def _ensure_scanner(self):
        if self._scanner is None:
            from predictability.scanner import PredictabilityScanner
            self._scanner = PredictabilityScanner(model_name=self._model_name)

    def detect(self, content: str, **kwargs) -> DetectResult:
        self._ensure_scanner()
        body_text = _pre_filter_body(content)
        result = self._scanner.scan_text(body_text)

        word_count = len(content.split())
        sentence_count = len(result.get("sentences", []))

        confidence, confidence_reason = self._assess_confidence(content, **kwargs)
        if sentence_count < 3:
            confidence = "low"
            confidence_reason = (
                f"Only {sentence_count} sentences detected; "
                "predictability signals are unstable at this length."
            )
        # Propagate short-text confidence from scanner
        sample_conf = result.get("sample_confidence")
        if sample_conf == "low":
            confidence = "low"
            confidence_reason = result.get(
                "sample_confidence_reason",
                "Short text: predictability signals are unstable."
            )

        findings = []
        for idx, s in enumerate(result["sentences"]):
            if getattr(s, "error", None):
                continue
            if s.risk_label in ("high", "medium", "review"):
                subtype = self._classify_predictability_subtype(s.sentence)
                if s.risk_label == "review":
                    finding_type = "review_predictability"
                    recommendation = "Review-level signal: normal prose predictability. No action needed unless paired with another risk signal."
                    action_type = "review_only"
                    evidence_strength = "weak"
                else:
                    finding_type = f"{s.risk_label}_predictability"
                    if s.risk_label == "high":
                        recommendation = "Sentence uses formulaic phrasing. Consider restructuring with specific evidence, cited claims, or original phrasing."
                        action_type = "add_specific_example"
                    else:
                        recommendation = "Review only. Moderate predictability is normal in academic writing unless paired with other risk signals."
                        action_type = "review_manually"
                    evidence_strength = "strong" if s.predictability_risk > 0.7 else "moderate"

                findings.append(Finding(
                    finding_type=finding_type,
                    risk_level=s.risk_label,
                    evidence_strength=evidence_strength,
                    detail=(
                        f"Sentence scored {s.predictability_risk:.1%} predictability "
                        f"(common ratio: {s.top_10_ratio:.1%}, "
                        f"category: {subtype.replace('_', ' ')})"
                    ),
                    evidence=s.sentence,
                    recommendation=recommendation,
                    suggested_action_type=action_type,
                    location=self._locate_sentence(content, s.sentence, idx),
                    metadata={
                        "score": s.predictability_risk,
                        "avg_probability": s.avg_probability,
                        "avg_surprisal": s.avg_surprisal,
                        "top10_ratio": s.top_10_ratio,
                        "top50_ratio": s.top_50_ratio,
                        "subtype": subtype,
                    },
                    signal_category=classify_finding_category(finding_type),
                ))

        matched_phrases = set()
        for s in result["sentences"]:
            for p in getattr(s, "matched_generic_phrases", []):
                matched_phrases.add(p)
        for phrase in matched_phrases:
            findings.append(Finding(
                finding_type="generic_phrase",
                risk_level="medium",
                evidence_strength="moderate",
                detail=f"Generic phrase detected: '{phrase}'",
                evidence=phrase,
                recommendation="Replace with domain-specific or original phrasing.",
                suggested_action_type="review_manually",
                location=self._locate_phrase(content, phrase),
                signal_category="genericity",
            ))

        for shift in result.get("style_shifts", []):
            findings.append(Finding(
                finding_type="style_shift",
                risk_level="low",
                evidence_strength="weak",
                detail=f"Predictability {shift['direction']} (Δ{shift['magnitude']:.1%})",
                evidence=shift.get("to", "")[:100],
                recommendation="Review for consistency in writing voice.",
                suggested_action_type="review_manually",
                metadata=shift,
                signal_category="predictability",
            ))

        return DetectResult(
            scanner=self.name,
            overall_risk=result["overall_risk"],
            confidence=confidence,
            confidence_reason=confidence_reason,
            risk_distribution=result["risk_distribution"],
            findings=findings,
            policy_message=self.policy_message,
            raw=result,
            detector_version=self.detector_version,
            model_name=self._model_name,
        )

    @staticmethod
    def _locate_sentence(content: str, sentence: str, fallback_idx: int) -> dict:
        start = content.find(sentence[:40])
        if start >= 0:
            return {
                "sentence_index": fallback_idx,
                "start_char": start,
                "end_char": start + len(sentence),
            }
        return {"sentence_index": fallback_idx}

    @staticmethod
    def _locate_phrase(content: str, phrase: str) -> dict:
        start = content.find(phrase)
        if start >= 0:
            return {"start_char": start, "end_char": start + len(phrase)}
        return {}
