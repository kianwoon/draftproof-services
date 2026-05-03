"""GPT-2 + LLM Hybrid Rewriter — GPT-2 identifies problem tokens, LLM picks best replacement.

Architecture:
  1. GPT-2 scans the sentence → finds which tokens are most predictable (rank 1-3)
  2. For each problem token, GPT-2 provides ranked alternatives from its distribution
  3. A targeted prompt is built with: original sentence + problem tokens + alternatives
  4. The LLM picks the best replacement for each problem token
  5. Result is scored with GPT-2 to verify improvement

Why this hybrid works:
  - GPT-2 provides PRECISION: knows exactly which tokens triggered the finding
  - GPT-2 provides BOUNDED CHOICE: alternatives are from its distribution, so any
    pick will change the predictability score
  - LLM provides JUDGMENT: sees full sentence context, picks alternatives that
    preserve grammar, meaning, and style
  - Together: metric-aligned + grammatical + meaningful

Fallback chain:
  GPT-2 analysis → LLM targeted rewrite → score verification
  If LLM unavailable → GPT-2 picks lowest-ranked alternative automatically
"""

import re
import logging
from typing import List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

MAX_PREDICTABLE_RANK = 3      # tokens ranked 1-3 by GPT-2 are "predictable"
MIN_TOKEN_LENGTH = 3          # skip short tokens
NUM_ALTERNATIVES = 8          # alternatives per problem token
MIN_ALT_RANK = 5              # sample alternatives starting at rank 5
MAX_ALT_RANK = 50             # up to rank 50
MAX_TARGETS = 3               # analyze up to 3 problem positions

# Tokens that should never be targeted for replacement.
# Includes: punctuation, function words (determiners, prepositions, conjunctions,
# pronouns, auxiliaries, particles). Only content words (nouns, verbs, adjectives,
# adverbs) should be targeted — those are the words that signal AI generation
# when they're predictable.
_SKIP_TOKENS: Set[str] = {
    # Punctuation
    ".", ",", "!", "?", ";", ":", "-", "--", "(", ")", "[", "]",
    "\n", "\t", " ", '"', "'", "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
    # Determiners
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "all", "each", "every",
    "no", "such", "what", "which",
    # Prepositions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "against", "without", "within", "along", "across",
    "behind", "beyond", "toward", "towards", "upon", "among", "around",
    # Conjunctions
    "and", "or", "but", "not", "if", "when", "while", "as", "than",
    "because", "since", "although", "though", "unless", "until", "whether",
    # Pronouns
    "it", "its", "he", "she", "they", "them", "we", "us", "you", "me",
    "him", "her", "who", "whom", "whose", "which", "what", "that", "this",
    "there", "here", "where", "how", "why",
    # Auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "can", "could", "may", "might", "shall", "should",
    "must",
    # Common particles / adverbs that are structural
    "so", "then", "also", "just", "only", "even", "still", "yet",
    "very", "too", "now", "well",
}

# LLM prompt for targeted replacement
_TARGETED_REWRITE_PROMPT = """You are a text editor. Your job is to reduce the predictability of a sentence by replacing specific flagged words with alternatives.

RULES:
- Replace ONLY the flagged words listed below.
- Pick alternatives that preserve the sentence's EXACT meaning.
- The sentence must remain grammatically correct after replacement.
- Keep the same sentence structure — do NOT reorder, split, or merge clauses.
- Copy ALL other words verbatim: proper nouns, numbers, dates, citations, quoted text.
- Output ONLY the rewritten sentence. No quotes, no commentary, no explanation."""


# ── GPT-2 analysis ────────────────────────────────────────────────────

def _find_predictable_positions(token_results: list) -> List[dict]:
    """Identify the most predictable token positions.

    Returns list of {tr_index, input_pos, token, rank, probability}.
    """
    targets = []
    for i, tr in enumerate(token_results):
        token = tr.token.strip()
        if token.lower() in _SKIP_TOKENS:
            continue
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if tr.rank > MAX_PREDICTABLE_RANK:
            continue
        if token.isdigit() or re.match(r'^\d+\.?\d*%?$', token):
            continue
        targets.append({
            "tr_index": i,
            "input_pos": i + 1,
            "token": token,
            "rank": tr.rank,
            "probability": tr.probability,
        })

    targets.sort(key=lambda t: (t["rank"], -t["probability"]))
    return targets[:MAX_TARGETS]


def _get_alternatives(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    token_results_index: int,
    device: str = "cpu",
    num_alternatives: int = NUM_ALTERNATIVES,
    min_rank: int = MIN_ALT_RANK,
    max_rank: int = MAX_ALT_RANK,
) -> List[str]:
    """Get vocabulary alternatives for a token position from GPT-2's distribution.

    Returns alternatives from rank min_rank..max_rank, excluding the original token.
    """
    input_pos = token_results_index + 1
    orig_token = tokenizer.decode([input_ids[0, input_pos].item()]).strip().lower()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    probs = torch.softmax(logits[0, input_pos - 1], dim=-1)
    sorted_indices = torch.argsort(probs, descending=True)

    alternatives = []
    seen = {orig_token}
    for rank in range(min_rank - 1, min(max_rank, len(sorted_indices))):
        token_id = sorted_indices[rank].item()
        token_text = tokenizer.decode([token_id]).strip()

        if not token_text or len(token_text) < 2:
            continue
        if token_text.lower() in seen:
            continue
        if re.match(r'^[\W\d]+$', token_text):
            continue
        # Must be alphabetic (skip subword fragments with leading spaces that
        # decode to odd things)
        if not re.match(r'^[A-Za-z]+$', token_text):
            continue
        seen.add(token_text.lower())
        alternatives.append(token_text)

        if len(alternatives) >= num_alternatives:
            break

    return alternatives


# ── LLM call ──────────────────────────────────────────────────────────

def _call_llm(gateway, sentence: str, targets: List[dict]) -> Optional[str]:
    """Call the LLM with a targeted rewrite prompt.

    The LLM sees the sentence, the flagged tokens, and alternatives.
    It picks the best replacement for each flagged token.
    """
    lines = [
        f"Sentence: {sentence}",
        "",
        "Flagged words to replace:",
    ]
    for t in targets:
        alts = t.get("alternatives", [])
        alt_str = ", ".join(alts[:6]) if alts else "(no alternatives found)"
        lines.append(
            f'  - "{t["token"]}" (GPT-2 rank {t["rank"]}, '
            f"predictability {t['probability']:.1%}) "
            f"→ alternatives: {alt_str}"
        )

    lines.append("")
    lines.append(
        "Replace each flagged word with one of its alternatives. "
        "Output ONLY the rewritten sentence."
    )

    user_msg = "\n".join(lines)

    try:
        resp = gateway.chat(user_msg, system=_TARGETED_REWRITE_PROMPT)
        if resp.is_empty:
            return None
        output = resp.content.strip()
        # Strip wrapping quotes/code blocks
        if output.startswith('"""') and output.endswith('"""'):
            output = output[3:-3].strip()
        elif output.startswith("```") and output.endswith("```"):
            output = output[3:-3].strip()
        return output
    except Exception as exc:
        logger.warning("LLM call for targeted rewrite failed: %s", exc)
        return None


# ── GPT-2-only fallback ───────────────────────────────────────────────

def _gpt2_only_rewrite(
    sentence: str,
    targets: List[dict],
) -> Optional[str]:
    """Fallback: replace each flagged token with its first (lowest-rank) alternative.

    Less grammatically aware than LLM, but requires no external API.
    """
    result = sentence
    for t in targets:
        alts = t.get("alternatives", [])
        if not alts:
            continue
        # Pick first alternative (lowest GPT-2 rank = biggest predictability drop)
        replacement = alts[0]
        # Use word-boundary regex to avoid partial matches
        pattern = r'\b' + re.escape(t["token"]) + r'\b'
        result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
    return result if result != sentence else None


# ── Semantic check ────────────────────────────────────────────────────

def _semantic_check(original: str, candidate: str) -> bool:
    """Verify candidate preserves key content from original."""
    def _content_words(text):
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

    orig_words = _content_words(original)
    cand_words = _content_words(candidate)

    if not orig_words:
        return True

    overlap = orig_words & cand_words
    if len(overlap) / len(orig_words) < 0.50:
        return False

    # Named entities
    for entity in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', original):
        if entity not in candidate:
            return False

    # Numbers
    for num in re.findall(r'\b\d+(?:\.\d+)?%?\b', original):
        if num not in candidate:
            return False

    return True


# ── Main API ──────────────────────────────────────────────────────────

class GPT2Rewriter:
    """GPT-2 + LLM hybrid sentence rewriter.

    GPT-2 identifies the most predictable tokens and their alternatives.
    The LLM (or fallback) picks the best replacements with full context.
    The result is scored to verify actual improvement.
    """

    def __init__(self, scanner=None, gateway=None):
        """
        Args:
            scanner: PredictabilityScanner instance (reuses loaded GPT-2 model).
            gateway: LLMGateway instance for targeted rewrite calls.
                     If None, falls back to GPT-2-only replacements.
        """
        self._scanner = scanner
        self._gateway = gateway

    def _get_scanner(self):
        if self._scanner is None:
            from predictability.scanner import PredictabilityScanner
            self._scanner = PredictabilityScanner()
        return self._scanner

    def analyze_sentence(self, sentence: str) -> List[dict]:
        """Analyze a sentence and return problem tokens with alternatives.

        Returns list of {token, rank, probability, alternatives}.
        Useful for building prompts even outside rewrite_sentence().
        """
        scanner = self._get_scanner()
        result = scanner.scan_sentence(sentence)

        if not result.token_results:
            return []

        tokenizer = scanner.tokenizer
        model = scanner.model
        device = scanner.device

        encoded = tokenizer(sentence, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        targets = _find_predictable_positions(result.token_results)
        for t in targets:
            alts = _get_alternatives(model, tokenizer, input_ids, t["tr_index"], device=device)
            t["alternatives"] = alts

        return targets

    def rewrite_sentence(
        self,
        sentence: str,
        context_before: str = "",
        target_risk: float = None,
    ) -> Optional[str]:
        """Rewrite a sentence to be less predictable.

        Args:
            sentence: The sentence to rewrite.
            context_before: Preceding sentence context (for future use).
            target_risk: If set, accept any candidate below this threshold.

        Returns:
            Best candidate, or None if no improvement found.
        """
        scanner = self._get_scanner()

        # Score original
        orig_result = scanner.scan_sentence(sentence)
        orig_risk = orig_result.predictability_risk

        if orig_risk < 0.30:
            return None

        if not orig_result.token_results:
            return None

        tokenizer = scanner.tokenizer
        model = scanner.model
        device = scanner.device

        encoded = tokenizer(sentence, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        # Step 1: GPT-2 analysis — find problem tokens + alternatives
        targets = _find_predictable_positions(orig_result.token_results)
        if not targets:
            return None

        for t in targets:
            alts = _get_alternatives(model, tokenizer, input_ids, t["tr_index"], device=device)
            t["alternatives"] = alts

        targets_with_alts = [t for t in targets if t["alternatives"]]
        if not targets_with_alts:
            return None

        logger.info(
            f"GPT2 analysis ({len(targets_with_alts)} targets): "
            + ", ".join(
                f'"{t["token"]}"(rank {t["rank"]}, alts: {t["alternatives"][:3]})'
                for t in targets_with_alts
            )
        )

        # Step 2: Get candidate via LLM or GPT-2-only fallback
        candidate = None
        if self._gateway:
            candidate = _call_llm(self._gateway, sentence, targets_with_alts)

        if candidate is None:
            candidate = _gpt2_only_rewrite(sentence, targets_with_alts)

        if candidate is None:
            return None

        # Step 3: Score the candidate
        cand_result = scanner.scan_sentence(candidate)
        new_risk = cand_result.predictability_risk

        # Step 4: Verify improvement
        if new_risk >= orig_risk:
            logger.info(
                f"GPT2 rewrite rejected: risk {orig_risk:.4f} → {new_risk:.4f} (no improvement)"
            )
            return None

        if not _semantic_check(sentence, candidate):
            logger.info(f"GPT2 rewrite rejected: semantic check failed")
            return None

        improvement = orig_risk - new_risk
        logger.info(
            f"GPT2 rewrite accepted: risk {orig_risk:.4f} → {new_risk:.4f} "
            f"(−{improvement:.4f}, label {orig_result.risk_label} → {cand_result.risk_label})"
        )
        return candidate

    def rewrite_sentence_fn(self, context_before: str = "") -> callable:
        """Create a rewrite_fn compatible with the rewrite engine interface.

        Returns a function: (text: str, span_info: str) -> Optional[str]
        """
        def rewrite_fn(text: str, span_info: str) -> Optional[str]:
            return self.rewrite_sentence(text, context_before=context_before)
        return rewrite_fn
