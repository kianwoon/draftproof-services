"""Closed-class linguistic reference lists used by the stylometry feature extractor.

allow-hardcode: every list below is a fixed, published, source-cited closed-class
linguistic resource (function-word inventories, a discourse-marker taxonomy, an
academic word list, an irregular-verb table) — human-reviewed reference guidance, not
a corpus-fitted scoring/matching oracle. None of these lists are derived from, tuned
against, or fitted to this repository's own detection/calibration corpora; they are
external measurement references. Explicitly approved for this purpose by the Global
Constraints section of docs/plans/consistency_defence_readiness_build_plan.md
(owner-approved 2026-07-18: "named, source-cited closed-class reference lists ...
a measurement resource, not a corpus-fitted scoring heuristic"). Each list below
cites its published source in a one-line comment.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# FUNCTION_WORDS
#
# Standard closed-class English function-word inventory (determiners, prepositions,
# coordinating + subordinating conjunctions, auxiliary/modal verbs, pronouns, and the
# negation particle) as used in authorship-attribution / stylometry research, e.g.
# Mosteller, F., & Wallace, D. L. (1964). Inference and Disputed Authorship: The
# Federalist. Addison-Wesley; Binongo, J. N. G. (2003). "Who wrote the 15th book of
# Oz? An application of multivariate analysis to authorship attribution." Chance,
# 16(2), 9-17.
# ---------------------------------------------------------------------------
FUNCTION_WORDS: frozenset[str] = frozenset({
    # Determiners
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each",
    "either", "neither", "all", "both", "several", "many", "much", "few",
    "little", "other", "another", "such", "what", "which", "whose",
    # Pronouns (personal, possessive, reflexive, relative, indefinite)
    "i", "me", "you", "he", "him", "she", "we", "us", "they", "them", "it",
    "mine", "yours", "hers", "ours", "theirs", "myself", "yourself",
    "himself", "herself", "itself", "ourselves", "yourselves", "themselves",
    "who", "whom", "someone", "anyone", "everyone", "somebody",
    "anybody", "everybody", "nobody", "something", "anything", "everything",
    "nothing", "one", "oneself",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "besides", "between", "beyond", "by", "concerning", "despite", "down",
    "during", "except", "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "outside", "over", "past", "regarding",
    "since", "through", "throughout", "to", "toward", "towards", "under",
    "underneath", "until", "unto", "up", "upon", "with", "within", "without",
    "via", "per", "amid", "amidst", "amongst", "versus",
    # Coordinating conjunctions
    "and", "but", "or", "nor", "yet", "so",
    # Subordinating conjunctions (single-word forms; see SUBORDINATING_CONJUNCTIONS
    # for the fuller set including multi-word subordinators)
    "because", "although", "though", "unless", "while", "whereas", "if",
    "when", "whenever", "where", "wherever", "as", "than", "whether", "once",
    "lest",
    # Auxiliary / modal verbs
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "ought", "need", "dare",
    # Negation: the bare "not"; the word-tokenizer keeps a leading-apostrophe suffix
    # attached to its host word (e.g. "didn't" is one token, not "did" + "n't"), so the
    # negative-contraction forms are listed here as whole tokens rather than as a
    # standalone "n't" that could never be produced by tokenization.
    "not", "aren't", "isn't", "wasn't", "weren't", "don't", "doesn't",
    "didn't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't",
    "couldn't", "shouldn't", "mustn't",
})

# ---------------------------------------------------------------------------
# SUBORDINATING_CONJUNCTIONS
#
# Standard English subordinating conjunctions, single- and multi-word, used to detect
# clause subordination. cf. Quirk, R., Greenbaum, S., Leech, G., & Svartvik, J. (1985).
# A Comprehensive Grammar of the English Language. Longman, Chapter 14 ("The Complex
# Sentence").
# ---------------------------------------------------------------------------
SUBORDINATING_CONJUNCTIONS: frozenset[str] = frozenset({
    "because", "although", "though", "since", "unless", "while", "whereas",
    "if", "before", "after", "until", "when", "whenever", "where",
    "wherever", "as", "than", "whether", "once", "lest", "whereby",
    "provided that", "providing that", "even though", "even if",
    "as long as", "as soon as", "so that", "in order that", "rather than",
    "as if", "as though", "in case", "now that", "given that",
    "insofar as", "inasmuch as",
})

# ---------------------------------------------------------------------------
# BE_FORMS
#
# The standard English copula/auxiliary "be" paradigm, used as the auxiliary half of
# the passive-voice (be-form + past-participle) heuristic. cf. Quirk, R., Greenbaum,
# S., Leech, G., & Svartvik, J. (1985). A Comprehensive Grammar of the English
# Language. Longman, Chapter 3.
# ---------------------------------------------------------------------------
BE_FORMS: frozenset[str] = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being",
})

# ---------------------------------------------------------------------------
# TRANSITION_MARKERS
#
# Common discourse transition / linking-adverbial phrases, grouped below by rhetorical
# function per Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999).
# Longman Grammar of Spoken and Written English. Longman, Chapter 10 ("Linking
# Adverbials"). Single-word subordinating conjunctions that also read as transitions
# (e.g. "although", "because") are intentionally excluded here — they are covered by
# SUBORDINATING_CONJUNCTIONS instead, so a token is not double-purposed.
# ---------------------------------------------------------------------------
_TRANSITION_ADDITION = {
    "also", "additionally", "in addition", "furthermore", "moreover",
    "besides", "further", "what is more", "as well", "likewise",
}
_TRANSITION_CONTRAST = {
    "however", "nevertheless", "nonetheless", "on the other hand",
    "in contrast", "by contrast", "conversely", "on the contrary",
    "even so", "despite this", "in spite of this", "still",
}
_TRANSITION_CAUSE_EFFECT = {
    "therefore", "thus", "hence", "consequently", "as a result",
    "accordingly", "for this reason", "because of this",
    "as a consequence",
}
_TRANSITION_SEQUENCE = {
    "first", "firstly", "second", "secondly", "third", "thirdly", "next",
    "then", "finally", "subsequently", "meanwhile", "afterward",
    "afterwards", "previously", "initially", "eventually",
    "at the same time",
}
_TRANSITION_EXEMPLIFICATION = {
    "for example", "for instance", "such as", "namely", "specifically",
    "in particular", "to illustrate", "as an illustration",
}
_TRANSITION_SUMMARY = {
    "in conclusion", "to conclude", "in summary", "to summarize",
    "to summarise", "overall", "in short", "all in all", "on the whole",
    "to sum up",
}
_TRANSITION_EMPHASIS = {
    "indeed", "in fact", "of course", "in other words", "that is",
    "that is to say", "more importantly", "above all", "notably",
}
_TRANSITION_COMPARISON = {
    "similarly", "by comparison", "in the same way", "correspondingly",
}

TRANSITION_MARKERS: frozenset[str] = frozenset(
    _TRANSITION_ADDITION
    | _TRANSITION_CONTRAST
    | _TRANSITION_CAUSE_EFFECT
    | _TRANSITION_SEQUENCE
    | _TRANSITION_EXEMPLIFICATION
    | _TRANSITION_SUMMARY
    | _TRANSITION_EMPHASIS
    | _TRANSITION_COMPARISON
)

# ---------------------------------------------------------------------------
# ACADEMIC_VOCAB
#
# Coxhead's Academic Word List (AWL) — all 570 headwords, sublists 1-10. Cite as:
# Coxhead, A. (2000). "A New Academic Word List." TESOL Quarterly, 34(2), 213-238.
# Headwords only (not the ~3000 related inflected/derived word forms); features.py
# applies a small, documented, approximate suffix-stripping step on top of this list
# rather than embedding the full related-word-form table here. Headwords are the
# source list's original British spellings (e.g. "analyse", "maximise", "labour",
# "licence") — American-spelled equivalents ("analyze", "maximize", "labor",
# "license") will under-count academic_vocab_rate; a documented, accepted gap.
# ---------------------------------------------------------------------------
ACADEMIC_VOCAB: frozenset[str] = frozenset({
    "abandon", "abstract", "academy", "access", "accommodate", "accompany", "accumulate",
    "accurate", "achieve", "acknowledge", "acquire", "adapt", "adequate", "adjacent",
    "adjust", "administrate", "adult", "advocate", "affect", "aggregate", "aid",
    "albeit", "allocate", "alter", "alternative", "ambiguous", "amend", "analogy",
    "analyse", "annual", "anticipate", "apparent", "append", "appreciate", "approach",
    "appropriate", "approximate", "arbitrary", "area", "aspect", "assemble", "assess",
    "assign", "assist", "assume", "assure", "attach", "attain", "attitude",
    "attribute", "author", "authority", "automate", "available", "aware", "behalf",
    "benefit", "bias", "bond", "brief", "bulk", "capable", "capacity",
    "category", "cease", "challenge", "channel", "chapter", "chart", "chemical",
    "circumstance", "cite", "civil", "clarify", "classic", "clause", "code",
    "coherent", "coincide", "collapse", "colleague", "commence", "comment", "commission",
    "commit", "commodity", "communicate", "community", "compatible", "compensate", "compile",
    "complement", "complex", "component", "compound", "comprehensive", "comprise", "compute",
    "conceive", "concentrate", "concept", "conclude", "concurrent", "conduct", "confer",
    "confine", "confirm", "conflict", "conform", "consent", "consequent", "considerable",
    "consist", "constant", "constitute", "constrain", "construct", "consult", "consume",
    "contact", "contemporary", "context", "contract", "contradict", "contrary", "contrast",
    "contribute", "controversy", "convene", "converse", "convert", "convince", "cooperate",
    "coordinate", "core", "corporate", "correspond", "couple", "create", "credit",
    "criteria", "crucial", "culture", "currency", "cycle", "data", "debate",
    "decade", "decline", "deduce", "define", "definite", "demonstrate", "denote",
    "deny", "depress", "derive", "design", "despite", "detect", "deviate",
    "device", "devote", "differentiate", "dimension", "diminish", "discrete", "discriminate",
    "displace", "display", "dispose", "distinct", "distort", "distribute", "diverse",
    "document", "domain", "domestic", "dominate", "draft", "drama", "duration",
    "dynamic", "economy", "edit", "element", "eliminate", "emerge", "emphasis",
    "empirical", "enable", "encounter", "energy", "enforce", "enhance", "enormous",
    "ensure", "entity", "environment", "equate", "equip", "equivalent", "erode",
    "error", "establish", "estate", "estimate", "ethic", "ethnic", "evaluate",
    "eventual", "evident", "evolve", "exceed", "exclude", "exhibit", "expand",
    "expert", "explicit", "exploit", "export", "expose", "external", "extract",
    "facilitate", "factor", "feature", "federal", "fee", "file", "final",
    "finance", "finite", "flexible", "fluctuate", "focus", "format", "formula",
    "forthcoming", "found", "foundation", "framework", "function", "fund", "fundamental",
    "furthermore", "gender", "generate", "generation", "globe", "goal", "grade",
    "grant", "guarantee", "guideline", "hence", "hierarchy", "highlight", "hypothesis",
    "identical", "identify", "ideology", "ignorant", "illustrate", "image", "immigrate",
    "impact", "implement", "implicate", "implicit", "imply", "impose", "incentive",
    "incidence", "incline", "income", "incorporate", "index", "indicate", "individual",
    "induce", "inevitable", "infer", "infrastructure", "inherent", "inhibit", "initial",
    "initiate", "injure", "innovate", "input", "insert", "insight", "inspect",
    "instance", "institute", "instruct", "integral", "integrate", "integrity", "intelligent",
    "intense", "interact", "intermediate", "internal", "interpret", "interval", "intervene",
    "intrinsic", "invest", "investigate", "invoke", "involve", "isolate", "issue",
    "item", "job", "journal", "justify", "label", "labour", "layer",
    "lecture", "legal", "legislate", "levy", "liberal", "licence", "likewise",
    "link", "locate", "logic", "maintain", "major", "manipulate", "manual",
    "margin", "mature", "maximise", "mechanism", "media", "mediate", "medical",
    "medium", "mental", "method", "migrate", "military", "minimal", "minimise",
    "minimum", "ministry", "minor", "mode", "modify", "monitor", "motive",
    "mutual", "negate", "network", "neutral", "nevertheless", "nonetheless", "norm",
    "normal", "notion", "notwithstanding", "nuclear", "objective", "obtain", "obvious",
    "occupy", "occur", "odd", "offset", "ongoing", "option", "orient",
    "outcome", "output", "overall", "overlap", "overseas", "panel", "paradigm",
    "paragraph", "parallel", "parameter", "participate", "partner", "passive", "perceive",
    "percent", "period", "persist", "perspective", "phase", "phenomenon", "philosophy",
    "physical", "plus", "policy", "portion", "pose", "positive", "potential",
    "practitioner", "precede", "precise", "predict", "predominant", "preliminary", "presume",
    "previous", "primary", "prime", "principal", "principle", "prior", "priority",
    "proceed", "process", "professional", "prohibit", "project", "promote", "proportion",
    "prospect", "protocol", "psychology", "publication", "publish", "purchase", "pursue",
    "qualitative", "quote", "radical", "random", "range", "ratio", "rational",
    "react", "recover", "refine", "regime", "region", "register", "regulate",
    "reinforce", "reject", "relax", "release", "relevant", "reluctance", "rely",
    "remove", "require", "research", "reside", "resolve", "resource", "respond",
    "restore", "restrain", "restrict", "retain", "reveal", "revenue", "reverse",
    "revise", "revolution", "rigid", "role", "route", "scenario", "schedule",
    "scheme", "scope", "section", "sector", "secure", "seek", "select",
    "sequence", "series", "sex", "shift", "significant", "similar", "simulate",
    "site", "so-called", "sole", "somewhat", "source", "specific", "specify",
    "sphere", "stable", "statistic", "status", "straightforward", "strategy", "stress",
    "structure", "style", "submit", "subordinate", "subsequent", "subsidy", "substitute",
    "successor", "sufficient", "sum", "summary", "supplement", "survey", "survive",
    "suspend", "sustain", "symbol", "tape", "target", "task", "team",
    "technical", "technique", "technology", "temporary", "tense", "terminate", "text",
    "theme", "theory", "thereby", "thesis", "topic", "trace", "tradition",
    "transfer", "transform", "transit", "transmit", "transport", "trend", "trigger",
    "ultimate", "undergo", "underlie", "undertake", "uniform", "unify", "unique",
    "utilise", "valid", "vary", "vehicle", "version", "via", "violate",
    "virtual", "visible", "vision", "visual", "volume", "voluntary", "welfare",
    "whereas", "whereby", "widespread",
})

# ---------------------------------------------------------------------------
# IRREGULAR_PARTICIPLES
#
# Standard English irregular past-participle forms, as commonly tabulated in reference
# grammars, used to support the passive-voice (be-form + past-participle) heuristic
# alongside a regular "-ed" regex fallback. cf. Quirk, R., Greenbaum, S., Leech, G., &
# Svartvik, J. (1985). A Comprehensive Grammar of the English Language. Longman,
# Appendix 3 ("Irregular Verbs").
# ---------------------------------------------------------------------------
IRREGULAR_PARTICIPLES: frozenset[str] = frozenset({
    "arisen", "awoken", "been", "become", "begun", "bent", "bet", "bitten",
    "bled", "blown", "bound", "bred", "broadcast", "broken", "brought",
    "built", "burnt", "burst", "bought", "cast", "caught", "chosen",
    "clung", "come", "cost", "crept", "cut", "dealt", "dived", "done",
    "drawn", "dreamt", "driven", "drunk", "dug", "eaten", "fallen", "fed",
    "felt", "fought", "found", "fit", "fled", "flown", "flung", "forbidden",
    "forgiven", "forgotten", "forsaken", "frozen", "given", "gone", "ground",
    "grown", "had", "heard", "held", "hidden", "hit", "hung", "hurt",
    "kept", "knelt", "knit", "known", "laid", "led", "leant", "leaped",
    "leapt", "learnt", "left", "lent", "let", "lain", "lit", "lost",
    "made", "meant", "met", "mistaken", "mown", "overcome", "paid",
    "proven", "put", "quit", "read", "ridden", "risen", "rung", "run",
    "said", "sawn", "seen", "sent", "set", "sewn", "shaken", "shed",
    "shone", "shot", "shown", "shrunk", "shut", "slain", "slept", "slid",
    "slit", "smelt", "sold", "sought", "sown", "spent", "spilt", "spoken",
    "sped", "spread", "sprung", "spun", "spat", "split", "stolen", "stood",
    "struck", "strung", "striven", "stuck", "stung", "stunk", "stridden",
    "sung", "sunk", "sat", "swept", "swollen", "swum", "swung", "sworn",
    "taken", "taught", "thought", "thrown", "thrust", "told", "torn",
    "trodden", "understood", "undergone", "undertaken", "upset", "wept",
    "withdrawn", "woken", "won", "worn", "woven", "wound", "written",
    "wrung",
})
