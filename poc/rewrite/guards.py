"""Guards for rewrite safety: protected spans, semantic drift, transactional apply."""

import re
import math
from dataclasses import dataclass, field
from typing import List, Set, Tuple


# ── Protected span detection ─────────────────────────────────────────

@dataclass
class ProtectedSpan:
    start_char: int
    end_char: int
    reason: str  # "direct_quote" | "citation" | "numeric" | "reference"
    text: str = ""


# Citation patterns (reused from citation scanner)
_APA_RE = re.compile(r"\((?:[A-Z][a-zäëïöü]+(?:\s+(?:et\s+al\.?|and|[&])\s*[A-Z][a-zäëïöü]+)*)\s*,\s*\d{4}[a-z]?(?:\s*[,.]\s*(?:pp\.?\s*\d+|p\.?\s*\d+))?\)")
_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")
_VANCOUVER_RE = re.compile(r"(?:\.|,|\))\s*(?:\^)?(\d{1,3}(?:\s*[,\-]\s*\d{1,3})*)")
_NARRATIVE_RE = re.compile(r"[A-Z][a-zäëïöü]+(?:\s+(?:et\s+al\.?|and)\s*[A-Z][a-zäëïöü]+)*\s*\(\d{4}[a-z]?\)")

# Quote patterns
_QUOTE_RE = re.compile(r'["“”].*?["“”]', re.DOTALL)
_SINGLE_QUOTE_RE = re.compile(r"(?:(?<=[\s(\[{,])[''']|['''](?=\s))['''].+?['''](?=[\s)\]}.,;:!?]|$)", re.DOTALL)

# Numeric patterns (standalone numbers, percentages, dates)
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?%?\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b', re.I)

_QUOTE_ATTRIBUTION_RE = re.compile(
    r"\b(?:according to|said|says|stated|wrote|reported|argued|argues|claimed|claims|"
    r"explained|explains|notes?|observed|observes|found|finds|concluded|concludes|"
    r"interview|survey|participant|respondent|learner|teacher|student may say|may say)\b",
    re.I,
)


def _quote_content(quote: str) -> str:
    return str(quote or "").strip().strip('"“”\'‘’').strip()


def _quote_words(quote: str) -> List[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", _quote_content(quote))


def _is_hard_quote(text: str, start: int, end: int, quote: str) -> bool:
    """Return True for quotes that should be preserved exactly.

    Short writer-emphasis phrases such as "what students know" are soft
    semantic content. Direct/evidence quotes, attributed quotes, long quotes,
    and citation-adjacent quotes remain hard protected anchors.
    """
    content = _quote_content(quote)
    if not content:
        return False
    words = _quote_words(quote)
    if len(words) >= 8:
        return True
    context = text[max(0, start - 90): min(len(text), end + 90)]
    if _QUOTE_ATTRIBUTION_RE.search(context):
        return True
    if re.search(r"\([A-Z][^)]*\d{4}[^)]*\)|\[[0-9,\-\s]+\]", context):
        return True
    if content.endswith("?") and len(words) <= 10:
        return False
    if re.search(r"[.!?]$", content) and content[:1].isupper() and len(words) >= 4:
        return True
    return False


def detect_protected_spans(text: str) -> List[ProtectedSpan]:
    """Find spans that must never be auto-rewritten."""
    spans = []

    # Direct quotes
    for m in _QUOTE_RE.finditer(text):
        if _is_hard_quote(text, m.start(), m.end(), m.group()):
            spans.append(ProtectedSpan(m.start(), m.end(), "direct_quote", m.group()))
    for m in _SINGLE_QUOTE_RE.finditer(text):
        if len(m.group()) > 4 and _is_hard_quote(text, m.start(), m.end(), m.group()):
            spans.append(ProtectedSpan(m.start(), m.end(), "direct_quote", m.group()))

    # Citation markers
    for pat in [_APA_RE, _NUMERIC_RE, _VANCOUVER_RE, _NARRATIVE_RE]:
        for m in pat.finditer(text):
            spans.append(ProtectedSpan(m.start(), m.end(), "citation", m.group()))

    # Standalone numbers and dates
    for m in _NUMBER_RE.finditer(text):
        spans.append(ProtectedSpan(m.start(), m.end(), "numeric", m.group()))

    # Sort by start position
    spans.sort(key=lambda s: s.start_char)
    return spans


def text_is_protected(text: str, start: int, end: int, protected: List[ProtectedSpan]) -> bool:
    """Check if a span overlaps with any protected span."""
    for p in protected:
        if start < p.end_char and end > p.start_char:
            return True
    return False


def mask_protected_spans(text: str, protected: List[ProtectedSpan], mask: str = "[PROTECTED]") -> Tuple[str, List[ProtectedSpan]]:
    """Replace protected spans with a mask token for rewrite prompt."""
    if not protected:
        return text, protected
    result = list(text)
    offset = 0
    adjusted = []
    for p in sorted(protected, key=lambda s: s.start_char):
        original_len = p.end_char - p.start_char
        start = p.start_char + offset
        end = p.end_char + offset
        result[start:end] = list(mask)
        diff = len(mask) - original_len
        adjusted.append(ProtectedSpan(start, start + len(mask), p.reason, p.text))
        offset += diff
    return "".join(result), adjusted


# ── Semantic drift guard ──────────────────────────────────────────────

@dataclass
class DriftCheck:
    accepted: bool
    similarity: float
    reasons: List[str] = field(default_factory=list)


def _extract_named_entities(text: str) -> Set[str]:
    """Extract proper nouns from text.

    Multi-word capitalized sequences are always entities (e.g. "New York").
    Single capitalized words are only counted if they appear mid-sentence
    (not after sentence-ending punctuation).
    """
    entities = set()

    heading_words = {
        "abstract", "introduction", "conclusion", "references", "appendix",
        "background", "discussion", "method", "methods", "results",
    }
    discourse_words = {
        "according", "as", "at",
        "also", "although", "because", "but", "finally", "first",
        "furthermore", "however", "instead", "meanwhile", "moreover",
        "from", "hence", "lastly", "next", "overall", "second", "therefore",
        "thus", "when", "while",
    }
    generic_academic_terms = {
        "behaviors", "challenges", "consumers", "costs", "customers",
        "age", "competence", "correctness", "digital", "employees", "figure", "figures", "guidelines",
        "hairdressing", "ideas", "lifelong", "observed", "outcome",
        "illusion", "pedagogy", "practice", "programmes", "relational", "resilience",
        "reviews", "servers", "services", "shoppers", "solutions",
        "staff", "structure", "students", "taxonomy", "teachers", "technique",
        "tools", "users",
    }
    quantifier_prefixes = {
        "another", "different", "many", "millions", "several", "some",
        "throughout", "various",
    }

    # Multi-word capitalized sequences (always proper nouns), but do not treat
    # headings joined to the first sentence across a newline as entities.
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)\b', text):
        entity = m.group(1)
        entity_words = entity.split()
        first = entity_words[0].lower()
        second = entity_words[1].lower() if len(entity_words) > 1 else ""
        if (
            first in heading_words
            or first in discourse_words
            or first in quantifier_prefixes
            or first in generic_academic_terms
            or (first == "the" and second in generic_academic_terms)
        ):
            continue
        entities.add(entity)

    # Single capitalized words: only if NOT after sentence boundary
    for m in re.finditer(r'(?<!^)(?<![.!?]\s)\b([A-Z][a-z]{2,})\b', text):
        word = m.group(1)
        lower_word = word.lower()
        if lower_word in discourse_words or lower_word in generic_academic_terms:
            continue
        # Check it's not a sentence start
        start = m.start()
        prefix = text[max(0, start - 3):start]
        previous_text = text[:start]
        previous_non_space = re.search(r"\S\s*$", previous_text)
        if not previous_non_space:
            continue
        previous_char = previous_non_space.group(0).strip()
        if previous_char in {".", "!", "?", "\n", "\r"} or "\n" in prefix:
            continue
        if not re.search(r'[.!?]["\')\]]?\s*$', prefix):
            entities.add(word)

    return entities


def _extract_numbers(text: str) -> Set[str]:
    """Extract standalone numbers and percentages."""
    return set(_NUMBER_RE.findall(text))


def _extract_citations(text: str) -> Set[str]:
    """Extract citation markers."""
    cites = set()
    for pat in [_APA_RE, _NUMERIC_RE, _VANCOUVER_RE, _NARRATIVE_RE]:
        for m in pat.finditer(text):
            cites.add(m.group())
    return cites


def _extract_quotes(text: str) -> Set[str]:
    """Extract hard quoted text for drift checks."""
    quotes = set()
    for m in _QUOTE_RE.finditer(text):
        if _is_hard_quote(text, m.start(), m.end(), m.group()):
            quotes.add(m.group())
    return quotes


def _semantic_key(value: str) -> str:
    """Normalize anchor text for semantic-preservation comparisons."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _entity_present(entity: str, text: str) -> bool:
    if re.search(rf"\b{re.escape(entity)}\b", text, flags=re.IGNORECASE):
        return True
    entity_key = _semantic_key(entity)
    text_key = _semantic_key(text)
    if entity_key and entity_key in text_key:
        return True
    stripped = re.sub(
        r"^(?:as|according to|at|in|on|from|by)\s+",
        "",
        str(entity or "").strip(),
        flags=re.IGNORECASE,
    )
    stripped_key = _semantic_key(stripped)
    return bool(stripped_key and stripped_key in text_key)


def _quote_semantic_key(quote: str) -> str:
    content = _quote_content(quote)
    content = re.sub(r"[.!?,;:]+$", "", content).strip()
    return _semantic_key(content)


def _extract_quote_semantic_keys(text: str, *, hard_only: bool = True) -> Set[str]:
    keys = set()
    quotes = _extract_quotes(text)
    if not hard_only:
        quotes = set(quotes)
        for match in _QUOTE_RE.finditer(text):
            content_key = _quote_semantic_key(match.group())
            if content_key:
                quotes.add(match.group())
    for quote in quotes:
        key = _quote_semantic_key(quote)
        if key:
            keys.add(key)
    return keys


def _keyword_cosine(text_a: str, text_b: str) -> float:
    """Keyword-frequency cosine similarity (no external deps)."""
    def _tokenize(t):
        return re.findall(r'\b[a-z]{3,}\b', t.lower())

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    freq_a = {}
    for w in tokens_a:
        freq_a[w] = freq_a.get(w, 0) + 1
    freq_b = {}
    for w in tokens_b:
        freq_b[w] = freq_b.get(w, 0) + 1

    all_words = set(freq_a) | set(freq_b)
    dot = sum(freq_a.get(w, 0) * freq_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v ** 2 for v in freq_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in freq_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def check_semantic_drift(
    original: str,
    rewritten: str,
    threshold: float = 0.88,
) -> DriftCheck:
    """Multi-signal semantic drift check.

    Hard gates: named entities, numbers, citations, quotes must be preserved.
    Soft signal: keyword cosine similarity.
    """
    reasons = []

    # Named entity preservation
    orig_entities = _extract_named_entities(original)
    lost_entities = {
        entity
        for entity in orig_entities
        if not _entity_present(entity, rewritten)
    }
    if lost_entities:
        for e in list(lost_entities)[:5]:
            reasons.append(f"lost_named_entity: '{e}'")

    # Number preservation
    orig_numbers = _extract_numbers(original)
    new_numbers = _extract_numbers(rewritten)
    lost_numbers = orig_numbers - new_numbers
    if lost_numbers:
        for n in list(lost_numbers)[:5]:
            reasons.append(f"number_changed: '{n}'")

    # Citation preservation
    orig_citations = _extract_citations(original)
    new_citations = _extract_citations(rewritten)
    lost_citations = orig_citations - new_citations
    if lost_citations:
        for c in list(lost_citations)[:3]:
            reasons.append(f"citation_lost: '{c}'")

    # Quote preservation
    orig_quotes = _extract_quote_semantic_keys(original, hard_only=True)
    new_quotes = _extract_quote_semantic_keys(rewritten, hard_only=False)
    lost_quotes = orig_quotes - new_quotes
    if lost_quotes:
        reasons.append(f"quote_lost: count {len(lost_quotes)}")

    # Keyword cosine similarity
    similarity = _keyword_cosine(original, rewritten)

    # Hard gate: any loss = reject
    if reasons:
        return DriftCheck(accepted=False, similarity=similarity, reasons=reasons)

    # Soft gate: similarity threshold
    if similarity < threshold:
        reasons.append(f"keyword_similarity: {similarity:.3f} < {threshold}")
        return DriftCheck(accepted=False, similarity=similarity, reasons=reasons)

    return DriftCheck(accepted=True, similarity=similarity, reasons=[])


# ── Regression memory ─────────────────────────────────────────────────

@dataclass
class RejectedRewrite:
    span_text: str
    reason: str
    candidate_text: str
    drift_similarity: float


class RegressionMemory:
    """Stores rejected rewrites to avoid retrying same failures."""

    def __init__(self):
        self._rejected: List[RejectedRewrite] = []

    def record(self, span_text: str, reason: str, candidate: str, similarity: float = 0.0):
        self._rejected.append(RejectedRewrite(span_text, reason, candidate, similarity))

    def was_rejected(self, span_text: str, reason: str) -> bool:
        return any(r.span_text == span_text and r.reason == reason for r in self._rejected)

    @property
    def count(self) -> int:
        return len(self._rejected)

    def summary(self) -> List[dict]:
        return [{"span": r.span_text[:60], "reason": r.reason, "similarity": round(r.drift_similarity, 3)} for r in self._rejected]


# ── Protected span preservation check ────────────────────────────────

def protected_spans_preserved(
    original: str,
    rewritten: str,
    protected: List[ProtectedSpan],
) -> bool:
    """Verify all protected spans appear in the rewritten text."""
    for span in protected:
        text = original[span.start_char:span.end_char]
        if text and text not in rewritten:
            return False
    return True


# ── Affected region (for local re-detection) ─────────────────────────

def affected_region(
    sentence_index: int,
    sentences: List[str],
) -> List[int]:
    """Return sentence indices to re-detect after a local rewrite.

    Includes the changed sentence ± 1 neighbor and the paragraph it's in.
    """
    indices = set()

    # Neighbors
    for offset in (-1, 0, 1):
        idx = sentence_index + offset
        if 0 <= idx < len(sentences):
            indices.add(idx)

    # Paragraph: detect paragraph boundaries (empty-line splits)
    # Walk backward to find paragraph start
    para_start = sentence_index
    while para_start > 0 and sentences[para_start - 1].strip():
        # Simple heuristic: if previous sentence ends with period, same paragraph
        para_start -= 1
        if para_start > 0 and not sentences[para_start - 1].strip():
            break

    # Walk forward to find paragraph end
    para_end = sentence_index
    while para_end < len(sentences) - 1 and sentences[para_end].strip():
        para_end += 1

    for i in range(para_start, para_end + 1):
        if 0 <= i < len(sentences):
            indices.add(i)

    return sorted(indices)


# ── Transactional apply ──────────────────────────────────────────────

@dataclass
class TransactionResult:
    accepted: bool
    text: str           # final text (new if accepted, snapshot if rejected)
    reason: str         # "" if accepted, rejection reason if not
    drift_similarity: float = 0.0
    voice_warnings: List[str] = field(default_factory=list)


def transactional_apply(
    snapshot: str,
    candidate: str,
    protected: List[ProtectedSpan],
    config,  # RewriteConfig
    voice_guard=None,  # VoiceGuard (optional)
) -> TransactionResult:
    """Apply a rewrite candidate transactionally: guard → keep/revert.

    Checks (in order):
    1. Protected span preservation (hard gate)
    2. Semantic drift (hard gate)
    3. Voice erosion (hard gate if VoiceGuard provided)
    4. Char delta within budget (soft gate)

    If any hard gate fails, returns the snapshot unchanged.
    """
    from rewrite.planner import EDIT_RADIUS

    # 1. Protected span preservation
    if not protected_spans_preserved(snapshot, candidate, protected):
        return TransactionResult(
            accepted=False,
            text=snapshot,
            reason="protected_span_lost",
        )

    # 2. Semantic drift
    drift = check_semantic_drift(snapshot, candidate, config.max_semantic_drift)
    if not drift.accepted:
        return TransactionResult(
            accepted=False,
            text=snapshot,
            reason=f"drift_rejected: {'; '.join(drift.reasons[:2])}",
            drift_similarity=drift.similarity,
        )

    # 3. Voice guard
    voice_warnings = []
    if voice_guard:
        from rewrite.voice import VoiceGuard
        voice_check = voice_guard.check(snapshot, candidate)
        if not voice_check.accepted:
            return TransactionResult(
                accepted=False,
                text=snapshot,
                reason=f"voice_eroded: {voice_check.reject_reason}",
                drift_similarity=drift.similarity,
            )
        voice_warnings = voice_check.warnings

    # 4. Char delta budget check
    orig_len = max(len(snapshot), 1)
    cand_len = len(candidate)
    char_delta = abs(cand_len - orig_len) / orig_len
    budget = config.budget
    if char_delta > budget.max_changed_char_ratio:
        return TransactionResult(
            accepted=False,
            text=snapshot,
            reason=f"char_delta_exceeded: {char_delta:.2f} > {budget.max_changed_char_ratio}",
            drift_similarity=drift.similarity,
        )

    # All checks passed — apply
    return TransactionResult(
        accepted=True,
        text=candidate,
        reason="",
        drift_similarity=drift.similarity,
        voice_warnings=voice_warnings,
    )


# ── Predictability regression guard ──────────────────────────────────

@dataclass
class RegressionCheck:
    accepted: bool
    orig_risk: float
    new_risk: float
    delta: float
    reason: str = ""


class PredictabilityGuard:
    """Per-sentence regression guard using the GPT-2 predictability scanner.

    After each rewrite, scores ONLY the changed sentence + 1 neighbor
    before and after. If predictability_risk goes UP, the change is reverted.

    Scans max 3 sentences per check (~0.75s) instead of the full document.
    """

    def __init__(self, scanner=None):
        self._scanner = scanner
        self._reverted = 0
        self._accepted = 0

    def _get_scanner(self):
        if self._scanner is None:
            from predictability.scanner import PredictabilityScanner
            self._scanner = PredictabilityScanner()
        return self._scanner

    @staticmethod
    def _extract_window(text: str, sentence: str) -> Tuple[List[str], int]:
        """Extract up to 3 sentences around the target sentence.

        Returns (sentences, target_index) where target_index is the
        position of the target sentence in the list.
        """
        # Split into sentences using simple regex
        sents = re.split(r'(?<=[.!?])\s+', text)
        sents = [s.strip() for s in sents if s.strip()]

        target_idx = -1
        for i, s in enumerate(sents):
            # Match by first 40 chars to handle rewrites
            if sentence[:40] in s or s[:40] in sentence:
                target_idx = i
                break

        if target_idx < 0:
            # Fallback: find by position in text
            pos = text.find(sentence[:30])
            if pos >= 0:
                char_count = 0
                for i, s in enumerate(sents):
                    char_count += len(s) + 1
                    if char_count > pos:
                        target_idx = i
                        break

        if target_idx < 0:
            return [sentence], 0

        # Window: ±1 sentence, max 3
        start = max(0, target_idx - 1)
        end = min(len(sents), target_idx + 2)
        window = sents[start:end]
        rel_target = target_idx - start
        return window, rel_target

    def check(
        self,
        orig_text: str,
        candidate_text: str,
        changed_sentence: str,
        candidate_sentence: str = "",
    ) -> RegressionCheck:
        orig_window, _ = self._extract_window(orig_text, changed_sentence)
        new_window, _ = self._extract_window(
            candidate_text,
            candidate_sentence or changed_sentence,
        )

        # Filter to eligible sentences (>= 6 words)
        orig_eligible = [s for s in orig_window if len(s.split()) >= 6]
        new_eligible = [s for s in new_window if len(s.split()) >= 6]

        if not orig_eligible or not new_eligible:
            return RegressionCheck(True, 0, 0, 0, "skip: too few eligible sentences")

        scanner = self._get_scanner()

        orig_risk = sum(
            scanner.scan_sentence(s).predictability_risk for s in orig_eligible
        ) / len(orig_eligible)
        new_risk = sum(
            scanner.scan_sentence(s).predictability_risk for s in new_eligible
        ) / len(new_eligible)

        delta = new_risk - orig_risk
        accepted = delta <= 0.02  # allow only scanner noise, not real regression

        if accepted:
            self._accepted += 1
        else:
            self._reverted += 1

        return RegressionCheck(
            accepted=accepted,
            orig_risk=round(orig_risk, 4),
            new_risk=round(new_risk, 4),
            delta=round(delta, 4),
            reason="" if accepted else f"predictability regressed +{delta:.4f}",
        )

    @property
    def stats(self) -> dict:
        return {"accepted": self._accepted, "reverted": self._reverted}
