from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .json_io import parse_json
from .quality_repair import (
    _clean_patch_text,
    _grammer_gateway,
    _grammer_model,
    _has_llm_api_key,
    _protected_tokens,
    _same_text,
    _word_count,
)
from .text import source_terms


@dataclass(frozen=True)
class NaturalisationOperation:
    find: str
    replace: str
    reason: str


@dataclass(frozen=True)
class NaturalisationCandidate:
    id: str
    reason: str
    text: str


@dataclass(frozen=True)
class NaturalisationResult:
    original_text: str
    repaired_text: str
    operations: list[NaturalisationOperation] = field(default_factory=list)
    skipped_operations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "not_run"
    model: str | None = None

    @property
    def changed(self) -> bool:
        return not _same_text(self.original_text, self.repaired_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed": self.changed,
            "model": self.model,
            "operations": [asdict(operation) for operation in self.operations],
            "skipped_operations": list(self.skipped_operations),
        }


def run_naturalisation_repair_once(
    current: str,
    *,
    original_text: str,
    quality_client: Any | None,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None,
    progress_callback: Callable[[int, str], None] | None,
) -> NaturalisationResult | None:
    if _same_text(current, original_text):
        return None
    if not _naturalisation_enabled():
        return None
    if quality_client is None and not _has_llm_api_key(api_key):
        return NaturalisationResult(
            original_text=current,
            repaired_text=current,
            status="skipped_missing_llm_api_key",
        )
    if cancellation_check is not None:
        cancellation_check()
    if progress_callback is not None:
        progress_callback(89, "Running V6 anti-overrepair cleanup")
    try:
        return run_naturalisation_repair(
            current,
            client=quality_client or _grammer_gateway(
                api_key=api_key,
                base_url=base_url,
                cancellation_check=cancellation_check,
            ),
            model=_grammer_model(),
        )
    except Exception as exc:
        return NaturalisationResult(
            original_text=current,
            repaired_text=current,
            status=f"failed:{type(exc).__name__}",
        )


def run_naturalisation_repair(
    text: str,
    *,
    client: Any,
    model: str | None = None,
) -> NaturalisationResult:
    response = client.chat(
        _naturalisation_prompt(text),
        system=(
            "You are a conservative copy editor. Return valid JSON only. "
            "Fix only visible over-repair artifacts created by a prior rewrite."
        ),
        temperature=0.0,
        top_p=0.2,
        max_tokens=None,
        response_format={"type": "json_object"},
        app_label="naturalisation",
    )
    payload = parse_json(getattr(response, "raw_content", "") or response.content)
    operations = _with_deterministic_candidates(text, _parse_operations(payload))
    repaired, applied, skipped = apply_naturalisation_operations(text, operations)
    return NaturalisationResult(
        original_text=text,
        repaired_text=repaired,
        operations=applied,
        skipped_operations=skipped,
        status="applied" if applied else "no_changes",
        model=getattr(client, "model", model),
    )


def apply_naturalisation_operations(
    text: str,
    operations: list[NaturalisationOperation],
) -> tuple[str, list[NaturalisationOperation], list[dict[str, Any]]]:
    current = text
    applied: list[NaturalisationOperation] = []
    skipped: list[dict[str, Any]] = []
    for operation in operations[:60]:
        reason = _skip_reason(current, operation)
        if reason:
            skipped.append({**asdict(operation), "skip_reason": reason})
            continue
        current = current.replace(operation.find, operation.replace, 1)
        applied.append(operation)
    return current, applied, skipped


def _naturalisation_prompt(text: str) -> str:
    candidates = _naturalisation_candidates(text)
    payload = {
        "task": "anti_overrepair_cleanup",
        "contract": [
            "Return patch operations only. Do not return a full rewritten document.",
            "Merge selectively only where short adjacent sentences are semantically dependent or mechanically repetitive.",
            "Fix repeated subject starts, mechanical decomposition, flat rhythm, awkward passive voice, broken semantic roles, repeated abstract nouns, weak paragraph flow, and inline punctuation line-break artifacts.",
            "Actively inspect for patterns like 'The same noun ... . The same noun ...', repeated sentence openings, awkward passive constructions with 'by', and final paragraphs where 'goal', 'process', or 'system' is repeated mechanically.",
            "Prioritize consecutive sentences that start with the same subject, and adjacent dependent sentences where the second starts with it, they, this, that, or the same noun.",
            "A run of three or more consecutive sentences with the same opening subject is the strongest cleanup candidate.",
            "Review every candidate cluster. Do not limit cleanup to the first obvious paragraph.",
            "Parallel short-list sequences may be merged when adjacent short sentences perform the same local job and every concrete item is preserved.",
            "Rhetorical ladders such as repeated 'no longer' sentences followed by 'the real challenge' may receive only a light merge; do not replace the author's concepts or wording with a new voice.",
            "Do not polish every sentence. Leave clear, purposeful short sentences unchanged.",
            "Do not optimize for AI detectors. Do not make the text smoother, broader, more generic, or more formal than needed.",
            "Preserve all concrete meaning, citations, names, numbers, examples, paragraph order, paragraph boundaries, and author voice.",
            "Each find string must be an exact substring from the submitted text.",
            "Each replacement must stay inside the same paragraph and be no larger than needed.",
        ],
        "merge_rule": "Merge only where short sentences are semantically dependent or mechanically repetitive.",
        "output_schema": {
            "operations": [
                {
                    "find": "exact original substring",
                    "replace": "minimal naturalised substring",
                    "reason": "short_sentence_chain|repeated_subject_start|parallel_short_list_sequence|rhetorical_ladder|mechanical_decomposition|passive_voice|semantic_role|repeated_abstract_noun|paragraph_flow|inline_punctuation_flow",
                }
            ]
        },
        "candidate_clusters": [asdict(candidate) for candidate in candidates[:40]],
        "text": text,
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _naturalisation_candidates(text: str) -> list[NaturalisationCandidate]:
    candidates: list[NaturalisationCandidate] = []
    seen: set[str] = set()
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        sentences = _sentences(paragraph)
        for start, end, reason in _candidate_windows(sentences):
            excerpt = " ".join(sentences[start:end]).strip()
            if excerpt and excerpt not in seen:
                seen.add(excerpt)
                candidates.append(NaturalisationCandidate(id=f"c{len(candidates) + 1:03d}", reason=reason, text=excerpt))
    return candidates


def _candidate_windows(sentences: list[str]) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    count = len(sentences)
    for start in range(count):
        for end in range(min(count, start + 6), start + 1, -1):
            excerpt = " ".join(sentences[start:end])
            if _has_repeated_start_sequence(excerpt):
                windows.append((start, end, "repeated_subject_start"))
                break
            if _has_rhetorical_ladder_sequence(excerpt):
                windows.append((start, end, "rhetorical_ladder"))
                break
            if _has_parallel_short_sequence(excerpt):
                windows.append((start, end, "parallel_short_list_sequence"))
                break
            if _has_dependent_sentence_sequence(excerpt):
                windows.append((start, end, "dependent_short_sentence_chain"))
                break
    return _remove_nested_windows(windows)


def _remove_nested_windows(windows: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    selected: list[tuple[int, int, str]] = []
    for start, end, reason in sorted(windows, key=lambda row: (row[0], -(row[1] - row[0]))):
        if any(start >= prev_start and end <= prev_end for prev_start, prev_end, _ in selected):
            continue
        selected.append((start, end, reason))
    return selected


def _parse_operations(payload: Any) -> list[NaturalisationOperation]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("operations")
    if not isinstance(rows, list):
        return []
    operations: list[NaturalisationOperation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        find = _clean_patch_text(row.get("find"))
        replace = _clean_patch_text(row.get("replace"))
        reason = " ".join(str(row.get("reason") or "").split())[:80]
        if find and replace and find != replace:
            operations.append(NaturalisationOperation(find=find, replace=replace, reason=reason))
    return operations


def _with_deterministic_candidates(
    text: str,
    operations: list[NaturalisationOperation],
) -> list[NaturalisationOperation]:
    deterministic = [
        *_passive_voice_candidates(text),
        *_rhetorical_ladder_candidates(text),
        *_repeated_subject_candidates(text),
        *_parallel_clause_candidates(text),
    ]
    return [*deterministic, *operations]


def _repeated_subject_candidates(text: str) -> list[NaturalisationOperation]:
    candidates: list[NaturalisationOperation] = []
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]+", paragraph))
        run: list[re.Match[str]] = []
        run_key = ""
        for match in sentence_matches:
            key = _sentence_start_key(match.group(0))
            if key and key == run_key:
                run.append(match)
                continue
            candidates.extend(_candidate_for_run(paragraph, run, run_key))
            run = [match]
            run_key = key
        candidates.extend(_candidate_for_run(paragraph, run, run_key))
    return candidates


def _passive_voice_candidates(text: str) -> list[NaturalisationOperation]:
    candidates: list[NaturalisationOperation] = []
    pattern = re.compile(
        r"\b(?P<object>[A-Z][A-Za-z'’-]*(?:\s+[a-z][A-Za-z'’-]*){0,4})\s+"
        r"was\s+received\s+by\s+"
        r"(?P<actor>[a-z][A-Za-z'’-]*(?:\s+(?!from\b)[a-z][A-Za-z'’-]*){0,4})"
        r"(?P<tail>\s+from\s+[^.!?]+)?\.",
    )
    for match in pattern.finditer(str(text or "")):
        obj = match.group("object").strip()
        actor = match.group("actor").strip()
        tail = (match.group("tail") or "").strip()
        replace = f"{actor[:1].upper()}{actor[1:]} received {obj[:1].lower()}{obj[1:]}{(' ' + tail) if tail else ''}."
        candidates.append(NaturalisationOperation(find=match.group(0), replace=replace, reason="passive_voice"))
    return candidates


def _rhetorical_ladder_candidates(text: str) -> list[NaturalisationOperation]:
    candidates: list[NaturalisationOperation] = []
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        matches = list(re.finditer(r"[^.!?]+[.!?]+", paragraph))
        for index in range(0, max(0, len(matches) - 2)):
            sentences = [match.group(0).strip() for match in matches[index:index + 3]]
            excerpt = " ".join(sentences)
            if not _has_rhetorical_ladder_sequence(excerpt):
                continue
            if not all(re.search(r"\bno\s+longer\b", sentence, flags=re.I) for sentence in sentences[:2]):
                continue
            first = _strip_sentence_end(sentences[0])
            second = _lower_leading_common_word(_strip_sentence_end(sentences[1]))
            third = sentences[2]
            candidates.append(
                NaturalisationOperation(
                    find=excerpt,
                    replace=f"{first}, and {second}. {third}",
                    reason="rhetorical_ladder",
                )
            )
    return candidates


def _parallel_clause_candidates(text: str) -> list[NaturalisationOperation]:
    candidates: list[NaturalisationOperation] = []
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]+", paragraph))
        run: list[re.Match[str]] = []
        for match in sentence_matches:
            sentence = match.group(0).strip()
            if _word_count(sentence) <= 10 and _first_main_verb(sentence):
                run.append(match)
                continue
            candidates.extend(_parallel_candidate_for_run(paragraph, run))
            run = []
        candidates.extend(_parallel_candidate_for_run(paragraph, run))
    return candidates


def _parallel_candidate_for_run(
    paragraph: str,
    run: list[re.Match[str]],
) -> list[NaturalisationOperation]:
    if len(run) < 3:
        return []
    find = paragraph[run[0].start():run[-1].end()].strip()
    clauses = [_strip_sentence_end(match.group(0)) for match in run]
    replace = _join_parallel_clauses(clauses) + "."
    if find == replace:
        return []
    return [NaturalisationOperation(find=find, replace=replace, reason="parallel_short_list_sequence")]


def _candidate_for_run(
    paragraph: str,
    run: list[re.Match[str]],
    run_key: str,
) -> list[NaturalisationOperation]:
    if len(run) < 2 or not run_key:
        return []
    find = paragraph[run[0].start():run[-1].end()].strip()
    replace = _merge_repeated_subject_sentences([match.group(0).strip() for match in run], run_key)
    if not replace or find == replace:
        return []
    return [NaturalisationOperation(find=find, replace=replace, reason="repeated_subject_start")]


def _merge_repeated_subject_sentences(sentences: list[str], run_key: str) -> str:
    parts = [_strip_sentence_end(sentence) for sentence in sentences]
    if len(parts) < 2:
        return ""
    predicate_parts = [parts[0]]
    for part in parts[1:]:
        predicate = _strip_repeated_start(part, run_key)
        if not predicate:
            return ""
        predicate_parts.append(predicate)
    if len(predicate_parts) == 2:
        if re.search(r"\bnot\b", predicate_parts[0], flags=re.I) and re.match(r"is\s+to\s+", predicate_parts[1], flags=re.I):
            return predicate_parts[0] + " but to " + re.sub(r"^is\s+to\s+", "", predicate_parts[1], flags=re.I) + "."
        joiner = " but " if re.search(r"\b(?:no longer|not|never)\b", predicate_parts[1], flags=re.I) else " and "
        return predicate_parts[0] + joiner + predicate_parts[1] + "."
    return _join_predicates(predicate_parts) + "."


def _strip_repeated_start(sentence: str, run_key: str) -> str:
    words = run_key.split()
    if not words:
        return ""
    pattern = r"^\s*" + r"\s+".join(re.escape(word) for word in words) + r"\s+"
    return re.sub(pattern, "", sentence, count=1, flags=re.I).strip()


def _strip_sentence_end(sentence: str) -> str:
    return str(sentence or "").strip().rstrip(".!?").strip()


def _join_predicates(parts: list[str]) -> str:
    if len(parts) == 3:
        return f"{parts[0]}, {parts[1]}, and {parts[2]}"
    return f"{parts[0]}, {', '.join(parts[1:-1])}, and {parts[-1]}"


def _join_parallel_clauses(clauses: list[str]) -> str:
    adjusted = [clauses[0], *[_lower_leading_common_word(clause) for clause in clauses[1:]]]
    return _join_predicates(adjusted)


def _lower_leading_common_word(clause: str) -> str:
    words = re.findall(r"^[A-Z][A-Za-z'’-]*", clause)
    if not words:
        return clause
    first = words[0]
    protected = {"AI", "YouTube", "TikTok", "NASA", "Google", "Microsoft", "Apple", "Tesla"}
    if first in protected:
        return clause
    common = {
        "Online", "Search", "Social", "Peer", "Individuals", "People", "Students", "Creators",
        "Schools", "They", "These", "The", "Assessment", "Discussion", "Improvement", "Other",
        "Some", "Many", "Curricula", "Access", "Knowledge",
    }
    if first not in common:
        return clause
    return first[:1].lower() + first[1:] + clause[len(first):]


def _skip_reason(text: str, operation: NaturalisationOperation) -> str:
    if text.count(operation.find) != 1:
        return "find_text_not_unique"
    if "\n\n" in operation.find or "\n\n" in operation.replace:
        return "paragraph_boundary_change"
    find_words = _word_count(operation.find)
    replace_words = _word_count(operation.replace)
    if find_words > 140:
        return "find_text_too_large"
    if replace_words > max(find_words + 24, int(find_words * 1.25) + 1):
        return "replacement_too_expansive"
    if re.search(r"[;:—]", operation.replace) and not re.search(r"[;:—]", operation.find):
        return "overpolish_punctuation"
    if _has_inline_layout_artifact(operation.find) or _has_awkward_passive_artifact(operation.find):
        pass
    elif _sentence_count(operation.find) < 2:
        return "single_sentence_polish"
    if not any((
        _has_inline_layout_artifact(operation.find),
        _has_awkward_passive_artifact(operation.find),
        _has_overrepair_signal(operation.find),
        _has_parallel_short_sequence(operation.find),
        _has_rhetorical_ladder_sequence(operation.find),
    )):
        return "no_overrepair_signal"
    protected_find = _protected_tokens(operation.find)
    protected_replace = _protected_tokens(operation.replace)
    if protected_find != protected_replace:
        return "protected_token_changed"
    dropped_terms = _dropped_content_terms(operation.find, operation.replace)
    if dropped_terms:
        return "content_term_dropped:" + ",".join(dropped_terms[:4])
    return ""


def _has_overrepair_signal(text: str) -> bool:
    sentences = _sentences(text)
    if len(sentences) < 2:
        return False
    if _has_repeated_start_sequence(text):
        return True
    if _has_dependent_sentence_sequence(text):
        return True
    if _has_rhetorical_ladder_sequence(text):
        return True
    abstract_nouns = re.findall(r"\b(?:situation|process|goal|system|model|issue|students?|teachers?)\b", text, flags=re.I)
    return len(abstract_nouns) >= 3


def _has_repeated_start_sequence(text: str) -> bool:
    starts = [_sentence_start_key(sentence) for sentence in _sentences(text)]
    starts = [start for start in starts if start]
    return len(starts) >= 2 and len(starts) != len(set(starts))


def _has_dependent_sentence_sequence(text: str) -> bool:
    if len(_sentences(text)) != 2:
        return False
    starts = [_sentence_start_key(sentence) for sentence in _sentences(text)]
    starts = [start for start in starts if start]
    if len(starts) < 2:
        return False
    dependent = {"it", "they", "this", "that", "these", "those", "such"}
    return any(start in dependent for start in starts[1:])


def _has_parallel_short_sequence(text: str) -> bool:
    sentences = _sentences(text)
    if len(sentences) < 3:
        return False
    counts = [_word_count(sentence) for sentence in sentences]
    if sum(1 for count in counts if count <= 10) < 3:
        return False
    if _has_repeated_phrase_pattern(sentences):
        return True
    verbs = [_first_main_verb(sentence) for sentence in sentences]
    return sum(1 for verb in verbs if verb) >= 3


def _has_rhetorical_ladder_sequence(text: str) -> bool:
    sentences = _sentences(text)
    if len(sentences) < 3:
        return False
    joined = " ".join(sentences).casefold()
    no_longer_count = sum(1 for sentence in sentences if re.search(r"\b(?:is|are|was|were)?\s*no\s+longer\b", sentence, flags=re.I))
    if no_longer_count >= 2 and re.search(r"\b(?:real|main|harder|hardest)\s+(?:challenge|problem|part)\b", joined):
        return True
    if no_longer_count >= 2 and any(re.search(r"\b(?:accurate|useful|trust(?:ing|ed|worthy)?)\b", sentence, flags=re.I) for sentence in sentences):
        return True
    return False


def _has_repeated_phrase_pattern(sentences: list[str]) -> bool:
    patterns = (
        r"\bwho\s+can\b",
        r"\bshould\s+(?:also\s+)?\w+\b",
        r"\bneed\s+to\b",
        r"\black\b",
        r"\bis\s+no\s+longer\b",
        r"\bno\s+longer\b",
    )
    for pattern in patterns:
        if sum(1 for sentence in sentences if re.search(pattern, sentence, flags=re.I)) >= 2:
            return True
    return False


def _first_main_verb(sentence: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)
    if len(words) < 2:
        return ""
    stop = {
        "a", "an", "the", "this", "that", "these", "those", "some", "other",
        "many", "students", "teachers", "people", "individuals", "creators",
    }
    verbish = {
        "provide", "provides", "offer", "offers", "share", "shares", "deliver", "delivers",
        "generate", "generates", "retrieve", "retrieves", "connect", "connects", "foster", "fosters",
        "lack", "lacks", "need", "needs", "reward", "rewards", "gain", "gains", "receive", "receives",
        "include", "includes", "incorporate", "incorporates", "complete", "completes", "improve", "improves",
    }
    for word in words[1:5]:
        key = word.casefold()
        if key in stop:
            continue
        if key in verbish:
            return key
    return ""


def _dropped_content_terms(find: str, replace: str) -> list[str]:
    replace_key = str(replace or "").casefold()
    droppable = {
        "situation", "process", "goal", "system", "model", "issue",
        "students", "student", "teachers", "teacher",
        "continue", "continues", "continued", "make", "makes", "made", "raise", "raises", "raised",
        "include", "includes", "included", "incorporate", "incorporates", "complete", "completes",
        "demonstrate", "demonstrates", "demonstrated", "prove", "proves", "proved",
        "deliver", "delivers", "delivered", "help", "helps", "helped", "foster", "fosters",
        "promote", "promotes", "practice", "practices", "practiced", "practise", "practises",
        "provide", "provides", "provided", "offer", "offers", "offered", "share", "shares",
        "shared", "generate", "generates", "generated", "retrieve", "retrieves", "retrieved",
        "connect", "connects", "connected", "receive", "receives", "received", "improve",
        "improves", "improved",
        "further", "absence", "learners", "learner",
        "mostly",
    }
    dropped: list[str] = []
    for term in source_terms(find, limit=48):
        key = term.casefold()
        if key in droppable or len(key) < 5:
            continue
        if key not in replace_key:
            dropped.append(term)
    return dropped


def _has_inline_layout_artifact(text: str) -> bool:
    return bool(re.search(r"\S\s*\n\s*[,;:)]|\(\s*\n\s*\S|\S\s*\n\s+(?:has|have|is|are|was|were|and|or)\b", str(text or ""), flags=re.I))


def _has_awkward_passive_artifact(text: str) -> bool:
    return bool(re.search(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\s+by\b", str(text or ""), flags=re.I))


def _sentence_count(text: str) -> int:
    return len(_sentences(text))


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in re.finditer(r"[^.!?]+[.!?]+", str(text or "")) if match.group(0).strip()]


def _sentence_start_key(sentence: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)
    if not words:
        return ""
    first = words[0].casefold()
    if first in {"the", "a", "an"} and len(words) > 1:
        return f"{first} {words[1].casefold()}"
    stable_subjects = {
        "teachers", "teacher", "students", "student", "schools", "school", "assessment",
        "model", "situation", "goal", "process", "system", "technology", "curricula",
    }
    if first in stable_subjects:
        return first
    if len(words) > 1 and first in {"education", "knowledge", "access"}:
        return f"{first} {words[1].casefold()}"
    return first


def _naturalisation_enabled() -> bool:
    value = _bool_env("DRAFTPROOF_V6_NATURALISATION_ENABLED")
    return True if value is None else value


def _bool_env(name: str) -> bool | None:
    import os

    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
