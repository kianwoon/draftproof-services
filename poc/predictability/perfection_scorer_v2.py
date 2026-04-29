"""Perfection scorer v2 — grammar-error pattern detection.

Detects ESL grammar patterns, tense mismatches, article misuse, etc.
Does NOT treat errors as human proof — outputs a grammar profile.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GrammarIssue:
    text: str
    issue_type: str
    position: int
    description: str


@dataclass
class GrammarProfile:
    minor_error_density: float       # errors per 100 words
    error_types: List[str]
    error_type_consistency: str      # consistent_mixed | consistent_esl | consistent_native | inconsistent
    register_consistency: str        # high | medium | low
    interpretation: str
    issues: List[GrammarIssue] = field(default_factory=list)


# ── Error patterns ─────────────────────────────────────────────────

ESL_PATTERNS = [
    # Tense mismatches
    (r'\bmade me (\w+ed|\w+ised?|\w+ized?)\b', 'tense_mismatch',
     'Past tense after "made me" — should be base form (e.g., "made me realised" → "made me realise")'),
    (r'\bI have (\w+ed)\s+(?:yesterday|last|ago)\b', 'tense_mismatch',
     'Present perfect with specific past time reference'),
    (r'\bhe don\'t\b', 'subject_verb', 'Subject-verb agreement: "he don\'t" → "he doesn\'t"'),

    # Article misuse
    (r'\ba ([aeiou]\w+)', 'article',
     'Possible article error: "a" before vowel sound'),
    (r'\ban ([bcdfghjklmnpqrstvwxyz]\w+)', 'article',
     'Possible article error: "an" before consonant sound'),

    # Subject-verb agreement
    (r'\b(the|this|that) (\w+s) (is|was|has) (?:been )?(\w+ing)\b', 'subject_verb',
     'Possible plural subject with singular verb'),
    (r'\beveryone (are|were)\b', 'subject_verb',
     '"everyone" takes singular verb'),
    (r'\beach of the \w+ (are|were)\b', 'subject_verb',
     '"each of" takes singular verb'),

    # Preposition oddities
    (r'\bdifferent (to|than)\b', 'preposition',
     'Preposition choice: "different to/than" vs "different from"'),
    (r'\bconsist (of|in)\b', 'preposition',
     'Preposition with "consist"'),
    (r'\bcomprised of\b', 'preposition',
     '"comprised of" — traditionally "composed of" or "comprises"'),
    (r'\bin the other hand\b', 'preposition',
     '"in the other hand" → "on the other hand"'),

    # Plural/singular mismatches
    (r'\bthere (is|was) (\w+) (?:things|students|people|children)\b', 'plural_singular',
     'Singular verb with plural noun'),
    (r'\bthere (are|were) (?:a|one) \b', 'plural_singular',
     'Plural verb with singular article'),

    # Collocation oddities
    (r'\bdo a (?:mistake|decision|progress)\b', 'collocation',
     'Collocation: common ESL pattern'),
    (r'\bmake a (?:research|homework|effort)\b', 'collocation',
     'Collocation: possible ESL pattern'),
    (r'\bhave a look (on|in)\b', 'collocation',
     'Collocation: "have a look at"'),

    # Awkward but understandable
    (r'\bI got to admit\b', 'register',
     'Informal register in academic context'),
    (r'\bkind of\b', 'register',
     'Informal hedging in academic context'),
    (r'\bstuff like\b', 'register',
     'Very informal in academic context'),
]


class PerfectionScorerV2:
    """Detect grammar patterns and produce a profile."""

    def analyze(self, text: str) -> GrammarProfile:
        words = text.split()
        word_count = max(len(words), 1)

        issues = []
        error_types = []

        for pattern, issue_type, description in ESL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append(GrammarIssue(
                    text=match.group(),
                    issue_type=issue_type,
                    position=match.start(),
                    description=description,
                ))
                if issue_type not in error_types:
                    error_types.append(issue_type)

        # Error density (per 100 words)
        error_density = (len(issues) / word_count) * 100

        # Error type consistency
        consistency = self._assess_consistency(error_types, issues)

        # Register consistency
        register_issues = [i for i in issues if i.issue_type == 'register']
        register = 'low' if len(register_issues) >= 2 else ('medium' if register_issues else 'high')

        # Interpretation
        if error_density < 1.0:
            interp = ("Very few grammar patterns detected. This is not "
                      "sufficient authorship evidence — AI can also produce "
                      "errors and humans can write perfectly.")
        elif error_density < 3.0:
            interp = ("Minor grammar patterns present. These may indicate "
                      "ESL background or informal register, but are not "
                      "conclusive authorship evidence.")
        else:
            interp = ("Multiple grammar patterns detected. Contains "
                      "human-like imperfections, but this is not sufficient "
                      "authorship evidence — AI can also generate errors.")

        return GrammarProfile(
            minor_error_density=round(error_density, 2),
            error_types=error_types,
            error_type_consistency=consistency,
            register_consistency=register,
            interpretation=interp,
            issues=issues,
        )

    def _assess_consistency(self, error_types: List[str],
                            issues: List[GrammarIssue]) -> str:
        if not error_types:
            return "consistent_native"

        # If errors cluster in ESL-specific categories
        esl_types = {'tense_mismatch', 'article', 'collocation', 'preposition'}
        native_types = {'register', 'plural_singular', 'subject_verb'}

        esl_count = sum(1 for i in issues if i.issue_type in esl_types)
        native_count = sum(1 for i in issues if i.issue_type in native_types)

        if esl_count > native_count * 2:
            return "consistent_esl"
        elif native_count > esl_count * 2:
            return "consistent_native"
        else:
            return "mixed"
