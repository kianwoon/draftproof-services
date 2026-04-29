"""Style Analyzer -- document-level AI structural signature detection.

Detects patterns that token-level predictability misses:
- Paragraph length symmetry (low variance = AI)
- Transition phrase density (too many seamless connectors)
- Sentence length uniformity (monotone rhythm)
- Inspirational/boilerplate framing
- Tone monotonicity (no hedging, uncertainty, humor)
- Linear topic progression (no digressions)
- Perfection score (too flawless = AI)
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


# ── Lexicon resources ────────────────────────────────────────────────

TRANSITION_PHRASES = [
    # additive / sequential
    "furthermore", "moreover", "in addition", "additionally",
    "not only", "but also", "as well as", "along with",
    "similarly", "likewise", "in the same way",
    # causal
    "as a result", "consequently", "therefore", "thus",
    "for this reason", "due to", "because of this",
    # contrastive (AI loves these for fake nuance)
    "however", "nevertheless", "on the other hand", "in contrast",
    "while it is true", "although", "despite this",
    # summary / conclusion
    "ultimately", "in conclusion", "to summarize", "in summary",
    "overall", "in essence", "fundamentally",
    # exemplification
    "for example", "for instance", "such as", "including",
    "in particular", "specifically", "namely",
    # sequential
    "firstly", "secondly", "thirdly", "finally", "lastly",
    "to begin with", "next", "then",
]

INSPIRATIONAL_PHRASES = [
    "enduring", "essential and enduring", "remains one of the most",
    "fundamentally change", "confidence it instills",
    "personal well-being", "professional confidence",
    "creative flair", "emotional intelligence",
    "manual dexterity", "lifelong learning",
    "at its core", "deep understanding",
    "critical part", "most critical",
    "profoundly", "powerful form of",
    "self-expression", "guardian of",
    "delicate balance", "rigid requirements",
    "fluid needs", "constant state of",
    "rapid pace", "growing focus",
    "minimize their environmental footprint",
    "personal evolution", "social status",
    "cultural identity",
]

HEDGING_PHRASES = [
    "perhaps", "maybe", "i think", "i suspect", "i'd guess",
    "arguably", "in my experience", "anecdotally",
    "i've noticed", "it seems", "sometimes", "occasionally",
    "often enough", "more often than not",
    "your mileage may vary", "ymmv",
    "i could be wrong", "as far as i can tell",
]

HUMOR_MARKERS = [
    # self-deprecation
    "admittedly", "confession", "to be honest", "tbh",
    # irony markers
    "— and ", "…", "...", "oh, and", "(spoiler:",
    # conversational asides
    "if you will", "so to speak", "as it were",
    "for want of a better", "for lack of a better",
    # parenthetical humor
    "(", ")", "—",
]


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    name: str
    score: float          # 0.0 = very human, 1.0 = very AI-like
    label: str            # low / medium / high
    detail: str           # human-readable explanation
    finding: str = ""     # specific quote or evidence

    THRESHOLDS = (0.35, 0.55)  # medium, high

    def __post_init__(self):
        if self.score >= self.THRESHOLDS[1]:
            self.label = "high"
        elif self.score >= self.THRESHOLDS[0]:
            self.label = "medium"
        else:
            self.label = "low"


@dataclass
class StyleProfile:
    dimensions: List[DimensionScore]
    overall_style_risk: float
    findings: List[str]            # actionable bullet points
    paragraph_lengths: List[int]
    sentence_lengths: List[int]
    transition_density: float
    tone_variance: float


# ── Analyzer ──────────────────────────────────────────────────────────

class StyleAnalyzer:

    def analyze(self, text: str) -> StyleProfile:
        paragraphs = self._split_paragraphs(text)
        sentences = self._split_sentences(text)
        dims: List[DimensionScore] = []

        # 1. Paragraph length symmetry
        dims.append(self._paragraph_symmetry(paragraphs))

        # 2. Transition phrase density
        dims.append(self._transition_density(text, sentences))

        # 3. Sentence length variance
        dims.append(self._sentence_rhythm(sentences))

        # 4. Inspirational framing
        dims.append(self._inspirational_framing(text, sentences))

        # 5. Tone monotonicity (hedging + humor + sentiment)
        dims.append(self._tone_monotonicity(text, sentences))

        # 6. Linear progression (no digressions)
        dims.append(self._linear_progression(paragraphs))

        # 7. Perfection / flawlessness
        dims.append(self._perfection_score(text, sentences))

        overall = sum(d.score for d in dims) / len(dims) if dims else 0.0

        para_lens = [len(p.split()) for p in paragraphs if p.strip()]
        sent_lens = [len(s.split()) for s in sentences]
        trans_density = self._count_transitions(text) / max(len(sentences), 1)

        findings = [d.finding for d in dims if d.finding and d.score >= 0.35]

        return StyleProfile(
            dimensions=dims,
            overall_style_risk=round(overall, 4),
            findings=findings,
            paragraph_lengths=para_lens,
            sentence_lengths=sent_lens,
            transition_density=round(trans_density, 4),
            tone_variance=round(self._compute_tone_variance(sentences), 4),
        )

    # ── Splitting helpers ─────────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s.strip()]

    # ── Dimension: paragraph symmetry ─────────────────────────────────

    def _paragraph_symmetry(self, paragraphs: List[str]) -> DimensionScore:
        if len(paragraphs) < 2:
            return DimensionScore("paragraph_symmetry", 0.0, "low",
                                  "Too few paragraphs to assess symmetry.")

        lengths = [len(p.split()) for p in paragraphs]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean if mean > 0 else 0  # coefficient of variation

        # cv < 0.15 is suspiciously uniform, > 0.4 is naturally varied
        score = max(0.0, min(1.0, 1.0 - (cv - 0.10) / 0.35))
        detail = f"CV={cv:.2f} (mean={mean:.0f} words, σ={math.sqrt(variance):.0f})"

        finding = ""
        if score >= 0.55:
            finding = f"Paragraphs are suspiciously uniform in length ({detail}). Vary paragraph sizes — mix short punchy ones with longer expository ones."
        elif score >= 0.35:
            finding = f"Paragraph lengths are somewhat uniform ({detail}). Consider breaking the pattern with a 1–2 sentence paragraph."

        return DimensionScore("paragraph_symmetry", round(score, 4), "", detail, finding)

    # ── Dimension: transition density ─────────────────────────────────

    def _count_transitions(self, text: str) -> int:
        lower = text.lower()
        return sum(1 for t in TRANSITION_PHRASES if t in lower)

    def _transition_density(self, text: str, sentences: List[str]) -> DimensionScore:
        count = self._count_transitions(text)
        n = max(len(sentences), 1)
        density = count / n

        # >0.3 transitions/sentence is very high (AI-like), <0.1 is natural
        score = max(0.0, min(1.0, (density - 0.08) / 0.25))
        detail = f"{count} transition phrases across {n} sentences ({density:.2f}/sentence)"

        finding = ""
        if score >= 0.55:
            found = [t for t in TRANSITION_PHRASES if t in text.lower()]
            finding = f"Transition-heavy: {detail}. Found: {', '.join(found[:6])}. Cut at least half — let ideas connect implicitly."
        elif score >= 0.35:
            finding = f"Moderate transition density ({detail}). Some could be cut for a more natural flow."

        return DimensionScore("transition_density", round(score, 4), "", detail, finding)

    # ── Dimension: sentence rhythm ─────────────────────────────────────

    def _sentence_rhythm(self, sentences: List[str]) -> DimensionScore:
        if len(sentences) < 3:
            return DimensionScore("sentence_rhythm", 0.0, "low",
                                  "Too few sentences to assess rhythm.")

        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean if mean > 0 else 0

        score = max(0.0, min(1.0, 1.0 - (cv - 0.15) / 0.40))
        detail = f"CV={cv:.2f} (mean={mean:.1f} words, σ={math.sqrt(variance):.1f})"

        finding = ""
        if score >= 0.55:
            finding = f"Sentences have near-identical length ({detail}). Mix short fragments with longer ones. Drop a 3-word sentence in."
        elif score >= 0.35:
            finding = f"Sentence lengths are somewhat uniform ({detail}). Add variety."

        return DimensionScore("sentence_rhythm", round(score, 4), "", detail, finding)

    # ── Dimension: inspirational framing ──────────────────────────────

    def _inspirational_framing(self, text: str, sentences: List[str]) -> DimensionScore:
        lower = text.lower()
        matches = [p for p in INSPIRATIONAL_PHRASES if p in lower]
        ratio = len(matches) / max(len(sentences), 1)

        # >0.25 = heavy, <0.08 = light
        score = max(0.0, min(1.0, (ratio - 0.05) / 0.25))

        # Check for inspirational ending pattern
        last_3 = sentences[-3:] if len(sentences) >= 3 else sentences
        last_text = " ".join(last_3).lower()
        ending_matches = [p for p in INSPIRATIONAL_PHRASES if p in last_text]
        if ending_matches:
            score = min(1.0, score + 0.15)

        detail = f"{len(matches)} inspirational phrases detected"
        if ending_matches:
            detail += f" (including {len(ending_matches)} in closing)"

        finding = ""
        if score >= 0.55:
            found = [p for p in INSPIRATIONAL_PHRASES if p in lower][:6]
            finding = f"Heavy inspirational boilerplate ({detail}). Phrases: {', '.join(found)}. Replace with concrete observations or a specific anecdote."
        elif score >= 0.35:
            finding = f"Some inspirational framing ({detail}). Tighten the closing — avoid 'enduring trades' style uplift."

        return DimensionScore("inspirational_framing", round(score, 4), "", detail, finding)

    # ── Dimension: tone monotonicity ───────────────────────────────────

    def _compute_tone_variance(self, sentences: List[str]) -> float:
        lower = " ".join(sentences).lower()
        hedge_count = sum(1 for h in HEDGING_PHRASES if h in lower)
        paren_count = lower.count("(")
        dash_count = sum(1 for s in sentences if "—" in s or "–" in s)

        voice_signals = hedge_count + paren_count + dash_count
        return voice_signals / max(len(sentences), 1)

    def _tone_monotonicity(self, text: str, sentences: List[str]) -> DimensionScore:
        lower = text.lower()

        hedge_count = sum(1 for h in HEDGING_PHRASES if h in lower)
        humor_count = sum(1 for h in HUMOR_MARKERS if h in lower)
        paren_count = lower.count("(")

        # Check for uncertainty words
        uncertainty_words = ["maybe", "perhaps", "might", "could", "possibly",
                           "i think", "i suspect", "seems to", "appears to"]
        uncertainty_count = sum(1 for u in uncertainty_words if u in lower)

        # Check for first person (personal voice)
        first_person = len(re.findall(r'\b(i|my|me|we|our)\b', lower))

        total_signals = hedge_count + uncertainty_count + (1 if first_person > 3 else 0)

        # 0 signals = monotone AI, 3+ = varied human voice
        score = max(0.0, min(1.0, 1.0 - total_signals / 3.5))

        detail = f"hedges={hedge_count}, uncertainty={uncertainty_count}, 1st-person={first_person}"

        finding = ""
        if score >= 0.55:
            finding = f"Monotone admiration — no hedging, no uncertainty, no humor ({detail}). Add a personal anecdote, a caveat, or a moment of self-doubt."
        elif score >= 0.35:
            finding = f"Somewhat monotone tone ({detail}). A hedging phrase or parenthetical aside would help."

        return DimensionScore("tone_monotonicity", round(score, 4), "", detail, finding)

    # ── Dimension: linear progression ─────────────────────────────────

    def _linear_progression(self, paragraphs: List[str]) -> DimensionScore:
        if len(paragraphs) < 3:
            return DimensionScore("linear_progression", 0.0, "low",
                                  "Too few paragraphs to assess progression.")

        # Broad topic markers — catches rewrites too, not just original
        topic_markers = {
            "intro": ["introduction", "is more than", "at its core", "the art of",
                      "sits at the intersection", "more than the simple",
                      "blend of", "gut instinct"],
            "technical": ["technical", "chemistry", "geometry", "precision",
                         "science", "mastery", "foundation", "paradox",
                         "tension", "elevation", "ph", "bleach", "keratin",
                         "bonds", "disulfide", "gummy", "controlled damage"],
            "social": ["social", "consultation", "listening", "empathy",
                      "relationship", "confidant", "vulnerability", "scissors",
                      "strangers in a mirror", "breakup", "divorce",
                      "just a trim", "side part"],
            "digression": ["incidental", "training manual", "then again",
                          "tuesday morning", "landlord", "i once watched",
                          "i have seen"],
            "evolution": ["evolution", "trends", "innovation", "sustainability",
                         "modern", "learning", "reinvents", "tiktok",
                         "reinvents", "biodegradable", "ergonomic"],
            "closing": ["ultimately", "confidence", "enduring", "essential",
                       "impact", "well-being", "real metric", "sits straighter",
                       "stranger reads your face", "too much credit"],
        }

        canonical_order = ["intro", "technical", "social", "evolution", "closing"]
        canonical_rank = {t: i for i, t in enumerate(canonical_order)}

        # ── Per-paragraph: get dominant topic + all present topics ──
        para_topics = []      # dominant topic per paragraph
        para_all_topics = []  # set of all detected topics per paragraph
        for p in paragraphs:
            lower = p.lower()
            topics = {cat: sum(1 for m in markers if m in lower)
                      for cat, markers in topic_markers.items()}
            dominant = max(topics, key=topics.get) if any(topics.values()) else "unknown"
            present = {cat for cat, count in topics.items() if count > 0}
            para_topics.append(dominant)
            para_all_topics.append(present)

        n = len(para_topics)
        canonical_only = [t for t in para_topics if t in canonical_rank]

        # ── Signal 1: strict monotonicity (0.0 = non-linear, 1.0 = perfectly monotone) ──
        # Count inversions: pairs where later canonical topic has lower rank than earlier
        inversions = 0
        pairs = 0
        for i in range(len(canonical_only)):
            for j in range(i + 1, len(canonical_only)):
                pairs += 1
                if canonical_rank[canonical_only[i]] > canonical_rank[canonical_only[j]]:
                    inversions += 1
        strict_monotone = 1.0 - (inversions / max(pairs, 1))

        # ── Signal 2: digression ratio ──
        digression_count = sum(1 for t in para_topics if t in ("digression", "unknown"))
        digression_ratio = digression_count / max(n, 1)

        # ── Signal 3: same-topic adjacency (evolution→evolution is linear) ──
        adjacent_same = sum(1 for i in range(1, n)
                           if para_topics[i] == para_topics[i-1]
                           and para_topics[i] not in ("digression", "unknown"))
        adjacency_ratio = adjacent_same / max(n - 1, 1)

        # ── Signal 4: topic interleaving (multiple topics in one paragraph) ──
        interleaved = sum(1 for topics in para_all_topics
                         if len(topics & set(canonical_order)) >= 2)
        interleaving_ratio = interleaved / max(n, 1)

        # ── Signal 5: out-of-position topics (closing in middle = non-linear) ──
        out_of_position = 0
        for i, t in enumerate(para_topics):
            if t == "closing" and i < n - 2:
                out_of_position += 1
            if t == "intro" and i > 1:
                out_of_position += 1

        # ── Composite scoring ──
        # Weight each signal: monotonicity is the strongest AI indicator
        # Interleaving and out-of-position are the strongest human indicators
        score = (
            strict_monotone * 0.35         # perfectly monotone = AI-like
            + adjacency_ratio * 0.10       # same-topic runs = AI-like
            - digression_ratio * 0.20      # digressions = human-like
            - interleaving_ratio * 0.20    # topic blending = human-like
            - (out_of_position * 0.15)     # out-of-position = human-like
        )

        score = max(0.0, min(1.0, score))

        detail = (f"topics={para_topics}, monotone={strict_monotone:.2f}, "
                  f"digressions={digression_count}, interleaved={interleaved}, "
                  f"out_of_pos={out_of_position}")

        finding = ""
        if score >= 0.55:
            finding = f"Still follows a broadly linear sequence ({detail}). Move a closing observation earlier, open with an anecdote, or weave a technical detail into a social paragraph."
        elif score >= 0.35:
            finding = f"Somewhat linear structure ({detail}). Rearranging one paragraph or merging topics across paragraphs would help."

        return DimensionScore("linear_progression", round(score, 4), "", detail, finding)

    # ── Dimension: perfection / flawlessness ──────────────────────────

    def _perfection_score(self, text: str, sentences: List[str]) -> DimensionScore:
        issues = 0
        details = []

        # Check for informal contractions (signs of human writing)
        informal = len(re.findall(r"(gonna|wanna|kinda|sorta|dunno|stuff|thing|pretty much|a lot)", text.lower()))

        # Check for run-on sentences or fragments
        very_long = sum(1 for s in sentences if len(s.split()) > 40)
        very_short = sum(1 for s in sentences if len(s.split()) < 5 and len(s) > 2)

        # Check for repeated sentence starters
        starters = [s.split()[0].lower() if s.split() else "" for s in sentences]
        from collections import Counter
        starter_counts = Counter(starters)
        repeated_starters = sum(v - 1 for v in starter_counts.values() if v > 1)

        # ── Multi-signal perfection scoring ──
        # Signal 1: well-formedness
        well_formed_ratio = 1.0 - (very_long + very_short) / max(len(sentences), 1)

        # Signal 2: em-dash / parenthetical usage (human-like self-correction)
        em_dash_count = text.count("—") + text.count("–")
        paren_count = text.count("(")
        self_correction = em_dash_count + paren_count

        # Signal 3: colloquialisms and casual phrasing
        casual_markers = ["sort of", "kind of", "pretty much", "to be fair",
                         "honestly", "look,", "right?", "— well",
                         "if you think about it", "come to think of it"]
        casual_count = sum(1 for m in casual_markers if m in text.lower())

        # Signal 4: sentence-ending variety (not all declarative)
        ends_with_question = sum(1 for s in sentences if s.rstrip().endswith("?"))
        ends_with_fragment = sum(1 for s in sentences if not s.rstrip().endswith((".", "!", "?")))

        # Signal 5: incomplete / trailing sentences
        trailing = sum(1 for s in sentences if s.rstrip().endswith(("—", "...", "…")))

        # Composite score
        perfection = well_formed_ratio

        # Each human-like imperfection reduces AI perfection score
        if informal > 0:
            perfection -= 0.12
        if very_short >= 1:
            perfection -= 0.08
        if very_short >= 3:
            perfection -= 0.10  # multiple fragments = very human
        if self_correction >= 2:
            perfection -= 0.08
        if casual_count > 0:
            perfection -= 0.10
        if ends_with_question >= 1:
            perfection -= 0.05
        if ends_with_fragment >= 1:
            perfection -= 0.08
        if trailing >= 1:
            perfection -= 0.06
        if repeated_starters > 3:
            perfection -= 0.04  # slightly human-like quirk

        score = max(0.0, min(1.0, perfection))
        detail = (f"well-formed={well_formed_ratio:.0%}, informal={informal}, "
                  f"casual={casual_count}, self-correct={self_correction}, "
                  f"questions={ends_with_question}, trailing={trailing}")

        finding = ""
        if score >= 0.55:
            finding = f"Too flawless ({detail}). Add a deliberate fragment, a question, a trailing em-dash, or a colloquial aside."
        elif score >= 0.35:
            finding = f"Nearly perfect grammar ({detail}). One more imperfection — a fragment, question, or casual aside — would read more naturally."

        return DimensionScore("perfection", round(score, 4), "", detail, finding)

    # ── Rewrite suggestions ───────────────────────────────────────────

    def get_rewrite_suggestions(self, profile: StyleProfile) -> List[str]:
        """Generate prioritized rewrite suggestions from the style analysis."""
        suggestions = []

        sorted_dims = sorted(profile.dimensions, key=lambda d: -d.score)

        for dim in sorted_dims:
            if dim.score < 0.35:
                continue

            if dim.name == "paragraph_symmetry":
                suggestions.append(
                    "STRUCTURE: Vary paragraph lengths where it improves readability. "
                    "Consider whether a shorter or longer paragraph would serve the content better."
                )
            elif dim.name == "transition_density":
                suggestions.append(
                    "TRANSITIONS: Remove explicit transitions where the logical connection is already clear to the reader. "
                    "Not every paragraph needs a signpost word."
                )
            elif dim.name == "sentence_rhythm":
                suggestions.append(
                    "RHYTHM: Vary sentence lengths for natural rhythm. "
                    "Mix shorter and longer sentences where it improves clarity."
                )
            elif dim.name == "inspirational_framing":
                suggestions.append(
                    "FRAMING: Replace broad inspirational language with concrete, specific observations. "
                    "Ground abstract claims in particular examples."
                )
            elif dim.name == "tone_monotonicity":
                suggestions.append(
                    "TONE: Add qualification where a claim genuinely benefits from nuance. "
                    "Not every statement needs to sound definitive."
                )
            elif dim.name == "linear_progression":
                suggestions.append(
                    "PROGRESSION: Consider reordering paragraphs if a non-linear structure "
                    "would better serve the argument. Weave related topics together where they connect."
                )
            elif dim.name == "perfection":
                suggestions.append(
                    "VOICE: Preserve the writer's existing register. "
                    "Where appropriate, let the natural voice come through — a question, an aside, "
                    "or a slightly informal phrase that the writer would actually use."
                )

        return suggestions
