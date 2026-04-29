"""Post-processing pipeline for detection findings.

Sits between scanners and report builder:

    Scanners → [PostProcessor] → ReportBuilder
                      ↓
             False Positive Filter
             Context Reclassifier
             Severity Adjustor
             Confidence Updater

Three filter stages:

1. GlossaryFilter    — auto-extract domain terms, downgrade if predictability
                       is driven by domain vocabulary
2. AcademicFilter    — detect citation/definition patterns that are predictable
                       by convention, not by AI generation
3. SeverityAdjustor  — reclassify findings with reason, recalculate overall tier,
                       preserve originals in raw_findings

Usage:
    from detect.postprocess import PostProcessor

    processor = PostProcessor(filters=[
        GlossaryFilter(),
        AcademicFilter(),
        SeverityAdjustor(minimum_gap=0.10),
    ])
    filtered = processor.process(detect_result, original_text=content)
"""

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Set, Tuple

from .base import Finding, DetectResult


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class PostProcessConfig:
    """Tunable thresholds for post-processing filters.

    Create with defaults, override per-domain:
        config = PostProcessConfig()                       # defaults
        config = PostProcessConfig(glossary_overlap=0.5)   # stricter overlap
        config = PostProcessConfig.from_dict({             # from profile dict
            "glossary_overlap": 0.5,
            "severity_gap": 0.15,
        })
    """
    # GlossaryFilter: fraction of evidence words that must match domain terms
    glossary_overlap: float = 0.40
    # GlossaryFilter: max glossary entries to keep (avoids noise)
    glossary_max_terms: int = 50
    # GlossaryFilter: minimum repetitions for a word to be considered domain term
    glossary_min_repeat: int = 3
    # SeverityAdjustor: how far above threshold a score must be to keep its level
    severity_gap: float = 0.10

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "PostProcessConfig":
        """Build config from a dict, ignoring unknown keys."""
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Filter protocol ──────────────────────────────────────────────────

class BaseFilter(ABC):
    """Every filter receives findings + context, returns modified findings."""

    @abstractmethod
    def apply(
        self,
        findings: List[Finding],
        context: str,
        metadata: Dict[str, Any],
    ) -> List[Finding]:
        """Return a (possibly modified) list of findings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ── 1. Domain Glossary Filter ────────────────────────────────────────

@dataclass
class _GlossaryEntry:
    term: str
    count: int
    source: str  # "capitalized" | "definition" | "repeated"


class GlossaryFilter(BaseFilter):
    """Auto-extract domain terms from the text and downgrade findings
    where high predictability is driven by domain vocabulary.

    Extraction strategy (hybrid):
      - Capitalized multi-word phrases (≥2 words)
      - Terms from definition patterns ("X refers to", "X is defined as")
      - Repeated uncommon words (frequency > 2, length > 5, not stop words)
    """

    STOP_WORDS = frozenset({
        "about", "above", "after", "again", "against", "between", "through",
        "during", "before", "following", "below", "beneath", "beside",
        "besides", "beyond", "certain", "could", "every", "however",
        "itself", "often", "other", "others", "overall", "particular",
        "rather", "should", "since", "some", "their", "there", "these",
        "things", "those", "though", "under", "unless", "until", "upon",
        "various", "whereas", "whether", "which", "while", "within",
        "without", "would", "although", "because", "therefore", "moreover",
        "furthermore", "nevertheless", "consequently", "specifically",
        "addition", "according", "importantly", "essentially",
    })

    def __init__(self, glossary: Optional[Set[str]] = None,
                 config: Optional[PostProcessConfig] = None):
        self._manual_glossary = glossary
        self._config = config or PostProcessConfig()

    @property
    def name(self) -> str:
        return "glossary"

    def apply(
        self,
        findings: List[Finding],
        context: str,
        metadata: Dict[str, Any],
    ) -> List[Finding]:
        glossary = self._build_glossary(context)
        metadata["glossary_terms"] = [e.term for e in glossary]

        result = []
        for f in findings:
            if f.finding_type not in (
                "high_predictability",
                "medium_predictability",
                "review_predictability",
            ):
                result.append(f)
                continue

            overlap = self._domain_overlap(f.evidence, glossary)
            if overlap >= self._config.glossary_overlap:
                new_risk = self._downgrade(f.risk_level)
                reason = (
                    f"Downgraded from {f.risk_level} to {new_risk}: "
                    f"sentence predictability driven by domain terminology "
                    f"({overlap:.0%} domain-term overlap)."
                )
                result.append(replace(
                    f,
                    risk_level=new_risk,
                    evidence_strength=self._weaken(f.evidence_strength),
                    metadata={**f.metadata, "reclassified": True, "reclassify_reason": reason},
                ))
            else:
                result.append(f)

        return result

    def _build_glossary(self, text: str) -> List[_GlossaryEntry]:
        """Hybrid auto-extraction: capitalized phrases + definitions + repeated terms."""
        if self._manual_glossary:
            return [_GlossaryEntry(t, 1, "manual") for t in self._manual_glossary]

        entries: Dict[str, _GlossaryEntry] = {}

        # 1. Capitalized multi-word phrases
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            term = match.group(1).lower()
            if term not in entries:
                entries[term] = _GlossaryEntry(term, 1, "capitalized")
            else:
                entries[term].count += 1

        # 2. Terms from definition patterns
        def_patterns = [
            r"(?:^|\.\s)([A-Z][a-z]+(?:\s+[a-z]+){0,3})\s+(?:refers?\s+to|is\s+defined\s+as|is\s+(?:a|the|an)\s+(?:method|approach|process|concept|technique|theory|model|framework)\s+of)",
            r"(?:called|known\s+as|termed)\s+[\"']([A-Za-z\s]+?)[\"']",
        ]
        for pat in def_patterns:
            for match in re.finditer(pat, text):
                term = match.group(1).strip().lower()
                if len(term) > 3 and term not in entries:
                    entries[term] = _GlossaryEntry(term, 1, "definition")

        # 3. Repeated uncommon words
        words = re.findall(r"\b([a-z]{6,})\b", text.lower())
        counts = Counter(words)
        for word, cnt in counts.items():
            if cnt >= self._config.glossary_min_repeat and word not in self.STOP_WORDS and word not in entries:
                entries[word] = _GlossaryEntry(word, cnt, "repeated")

        # Sort by count descending; keep top N to avoid noise
        sorted_entries = sorted(entries.values(), key=lambda e: e.count, reverse=True)
        return sorted_entries[:self._config.glossary_max_terms]

    @staticmethod
    def _domain_overlap(evidence: str, glossary: List[_GlossaryEntry]) -> float:
        """Fraction of evidence words that match glossary terms."""
        words = evidence.lower().split()
        if not words:
            return 0.0
        glossary_set = {e.term for e in glossary}
        matched = sum(
            1 for w in words
            if any(g in w or w in g for g in glossary_set)
        )
        return matched / len(words)

    @staticmethod
    def _downgrade(level: str) -> str:
        return {"high": "medium", "medium": "low", "low": "low", "review": "low"}.get(level, level)

    @staticmethod
    def _weaken(strength: str) -> str:
        return {"strong": "moderate", "moderate": "weak", "weak": "weak"}.get(strength, strength)


# ── 2. Academic Convention Filter ────────────────────────────────────

class AcademicFilter(BaseFilter):
    """Detect citation patterns, definition structures, and named entities
    that are predictable by academic convention — not by AI generation.

    Patterns recognized:
      - Citations: (Author, Year), (Author et al., Year), [1], [1,2]
      - Quoted terms: "Term" or 'Term'
      - Definitions: "X refers to", "X is the method of", "X can be defined as"
      - Named entities: Capitalized multi-word phrases (≥3 words)
    """

    # Patterns that indicate academic convention
    _CITATION_RE = re.compile(
        r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)"    # (Author, 2024) or (Smith et al., 2024)
        r"|\([A-Z][a-z]+(?:\s+et\s+al\.)?\s+\d{4}\)"     # (Author 2024) no comma
        r"|[A-Z][a-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\)"     # Author (2024) — author outside parens
        r"|\[\d+(?:[,;\s]+\d+)*\]"                        # [1] or [1, 2]
    )
    _QUOTE_RE = re.compile(r'["“”]([^"“”]{2,40})["“”]')
    _DEF_RE = re.compile(
        r"(?:refers?\s+to|is\s+defined\s+as|is\s+the\s+(?:method|process|approach|concept)"
        r"|can\s+be\s+(?:defined|described|understood)\s+as|denotes|signifies)",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "academic"

    def apply(
        self,
        findings: List[Finding],
        context: str,
        metadata: Dict[str, Any],
    ) -> List[Finding]:
        has_citations = bool(self._CITATION_RE.search(context))
        has_definitions = bool(self._DEF_RE.search(context))
        quoted_terms = [m.group(1) for m in self._QUOTE_RE.finditer(context)]

        metadata["academic_signals"] = {
            "citations": has_citations,
            "definitions": has_definitions,
            "quoted_terms": quoted_terms[:20],
        }

        if not has_citations and not has_definitions and not quoted_terms:
            return findings

        result = []
        for f in findings:
            if f.finding_type not in (
                "high_predictability",
                "medium_predictability",
                "review_predictability",
            ):
                result.append(f)
                continue

            evidence = f.evidence
            reasons = []

            if has_citations and self._CITATION_RE.search(evidence):
                reasons.append("contains citation pattern (predictable by convention)")

            if self._DEF_RE.search(evidence):
                reasons.append("contains definition structure (predictable by convention)")

            for qt in quoted_terms:
                if qt.lower() in evidence.lower():
                    reasons.append(f"contains quoted term '{qt}' (predictable by convention)")
                    break

            if reasons:
                new_risk = "low" if f.risk_level in ("high", "medium") else f.risk_level
                reason = (
                    f"Downgraded from {f.risk_level} to {new_risk}: "
                    + "; ".join(reasons)
                )
                result.append(replace(
                    f,
                    risk_level=new_risk,
                    evidence_strength="weak",
                    metadata={**f.metadata, "reclassified": True, "reclassify_reason": reason},
                ))
            else:
                result.append(f)

        return result

    def _sentence_context(self, evidence: str, full_text: str) -> str:
        """Extract surrounding context (±200 chars) for the evidence."""
        idx = full_text.find(evidence[:30])
        if idx < 0:
            return evidence
        start = max(0, idx - 100)
        end = min(len(full_text), idx + len(evidence) + 100)
        return full_text[start:end]


# ── 3. Severity Adjustment ──────────────────────────────────────────

class SeverityAdjustor(BaseFilter):
    """Reclassify findings based on margin above threshold.

    Only flags findings that are well above the decision boundary.
    Findings close to the threshold get downgraded — they're ambiguous.

    Also recalculates the overall risk distribution and tier after
    all previous filters have run.
    """

    def __init__(self, minimum_gap: float = 0.10,
                 config: Optional[PostProcessConfig] = None):
        """
        Args:
            minimum_gap: Override for severity gap. If config is provided,
                         this is ignored in favor of config.severity_gap.
            config: PostProcessConfig with severity_gap threshold.
        """
        self._config = config or PostProcessConfig()
        # config takes precedence over the direct argument
        self.minimum_gap = self._config.severity_gap

    @property
    def name(self) -> str:
        return "severity"

    # Threshold boundaries (consistent with PredictabilityScanner)
    _THRESHOLDS = {"high": 0.70, "medium": 0.45, "review": 0.35}

    def apply(
        self,
        findings: List[Finding],
        context: str,
        metadata: Dict[str, Any],
    ) -> List[Finding]:
        result = []
        reclassifications = []

        for f in findings:
            # Skip non-predictability findings — they don't have score-based thresholds
            if f.finding_type not in (
                "high_predictability",
                "medium_predictability",
                "review_predictability",
            ):
                result.append(f)
                continue

            score = self._extract_score(f.detail)
            if score is None:
                result.append(f)
                continue

            level = f.risk_level
            threshold = self._THRESHOLDS.get(level, 0.0)

            if level == "high" and score < threshold + self.minimum_gap:
                new_level = "medium"
                reason = (
                    f"Score ({score:.0%}) is within {self.minimum_gap} gap of high "
                    f"threshold ({threshold:.0%}); downgraded to {new_level}."
                )
                result.append(replace(
                    f,
                    risk_level=new_level,
                    evidence_strength=self._weaken(f.evidence_strength),
                    metadata={**f.metadata, "reclassified": True, "reclassify_reason": reason},
                ))
                reclassifications.append({"from": level, "to": new_level, "score": score})
            elif level == "medium" and score < threshold + self.minimum_gap:
                new_level = "low"
                reason = (
                    f"Score ({score:.0%}) is within {self.minimum_gap} gap of medium "
                    f"threshold ({threshold:.0%}); downgraded to {new_level}."
                )
                result.append(replace(
                    f,
                    risk_level=new_level,
                    evidence_strength="weak",
                    metadata={**f.metadata, "reclassified": True, "reclassify_reason": reason},
                ))
                reclassifications.append({"from": level, "to": new_level, "score": score})
            else:
                result.append(f)

        metadata["severity_reclassifications"] = reclassifications
        return result

    @staticmethod
    def _extract_score(detail: str) -> Optional[float]:
        """Extract predictability score from detail string like 'Sentence scored 0.7234...'"""
        match = re.search(r"scored\s+([\d.]+)", detail)
        return float(match.group(1)) if match else None

    @staticmethod
    def _weaken(strength: str) -> str:
        return {"strong": "moderate", "moderate": "weak", "weak": "weak"}.get(strength, strength)


# ── PostProcessor orchestrator ───────────────────────────────────────

@dataclass
class PostProcessResult:
    """Wrapper around DetectResult with post-processing metadata."""
    result: DetectResult
    raw_findings: List[Finding]
    filters_applied: List[str]
    filter_metadata: Dict[str, Any] = field(default_factory=dict)
    findings_reclassified: int = 0
    findings_suppressed: int = 0


class PostProcessor:
    """Run configured filters on a DetectResult, returning a filtered result.

    Preserves original findings in raw_findings for transparency.
    Recalculates overall_risk and risk_distribution after filtering.
    """

    def __init__(self, filters: Optional[List[BaseFilter]] = None,
                 config: Optional[PostProcessConfig] = None,
                 domain_terms: Optional[List[str]] = None):
        self._config = config or PostProcessConfig()
        self._domain_terms = set(domain_terms) if domain_terms else None
        self.filters = filters or [
            GlossaryFilter(glossary=self._domain_terms, config=self._config),
            AcademicFilter(),
            SeverityAdjustor(config=self._config),
        ]

    def process(
        self,
        detect_result: DetectResult,
        original_text: str = "",
    ) -> PostProcessResult:
        """Apply all filters sequentially to the findings.

        Args:
            detect_result: Raw output from a scanner.
            original_text: The full document text (needed for glossary extraction
                           and academic pattern detection).

        Returns:
            PostProcessResult with filtered DetectResult + metadata.
        """
        findings = list(detect_result.findings)
        raw_findings = list(detect_result.findings)
        filter_metadata: Dict[str, Any] = {}

        for filt in self.filters:
            findings = filt.apply(findings, original_text, filter_metadata)

        # Count reclassifications
        reclassified = sum(
            1 for f in findings
            if f.metadata.get("reclassified")
        )

        # Recalculate risk distribution and overall risk
        new_distribution = self._recompute_distribution(findings)
        new_risk = self._recompute_risk(findings, detect_result.overall_risk)
        new_confidence, new_confidence_reason = self._recompute_confidence(
            findings, detect_result
        )

        filtered_result = DetectResult(
            scanner=detect_result.scanner,
            overall_risk=new_risk,
            confidence=new_confidence,
            confidence_reason=new_confidence_reason,
            risk_distribution=new_distribution,
            findings=findings,
            policy_message=detect_result.policy_message,
            raw=detect_result.raw,
            detector_version=detect_result.detector_version,
            model_name=detect_result.model_name,
            config_hash=detect_result.config_hash,
            classification_target=detect_result.classification_target,
            likelihood_score=detect_result.likelihood_score,
            feature_summary=detect_result.feature_summary,
        )

        return PostProcessResult(
            result=filtered_result,
            raw_findings=raw_findings,
            filters_applied=[f.name for f in self.filters],
            filter_metadata=filter_metadata,
            findings_reclassified=reclassified,
            findings_suppressed=len(raw_findings) - len(findings),
        )

    @staticmethod
    def _recompute_distribution(findings: List[Finding]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for f in findings:
            dist[f.risk_level] = dist.get(f.risk_level, 0) + 1
        return dist

    @staticmethod
    def _recompute_risk(findings: List[Finding], original_risk: float) -> float:
        """Recalculate risk based on the highest remaining risk level."""
        if not findings:
            return min(original_risk, 0.2)  # Floor at 0.2 if all findings filtered
        level_scores = {"high": 0.7, "medium": 0.5, "low": 0.3, "review": 0.4}
        max_risk = max(level_scores.get(f.risk_level, 0.1) for f in findings)
        # Weighted average: 60% scanner probability, 40% categorical severity
        return round(original_risk * 0.6 + max_risk * 0.4, 4)

    @staticmethod
    def _recompute_confidence(
        findings: List[Finding],
        original: DetectResult,
    ) -> Tuple[str, str]:
        """If findings were significantly reclassified, adjust confidence."""
        reclassified = sum(1 for f in findings if f.metadata.get("reclassified"))
        total = len(findings)
        if total == 0:
            return original.confidence, original.confidence_reason

        reclass_rate = reclassified / total
        if reclass_rate > 0.5:
            level = "low" if original.confidence == "high" else original.confidence
            reason = (
                f"{reclassified}/{total} findings reclassified by post-processing; "
                f"original confidence adjusted. {original.confidence_reason}"
            )
            return level, reason

        return original.confidence, original.confidence_reason
