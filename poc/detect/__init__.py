"""DraftProof Detect — unified detection module (Integrity Signal Engine)."""

from .base import BaseDetector, DetectResult, Finding, DetectionReport, RewriteDecision
from .predictability import PredictabilityDetector
from .ai_generation import AIGenerationSignalDetector
from .similarity import SimilarityDetector
from .citation import CitationDetector
from .postprocess import (
    PostProcessor,
    PostProcessResult,
    PostProcessConfig,
    GlossaryFilter,
    AcademicFilter,
    SeverityAdjustor,
)
from .run import DetectionRunner
from .profiles import DomainProfile, resolve_profile, auto_detect_domain, list_available_profiles
from .thresholds import ThresholdConfig
from .utils import split_sentences
from .scoring import SignalScores, calculate_authorship_concern, extract_signals
from .mitigation import build_ai_mitigation_plan

__all__ = [
    "BaseDetector",
    "DetectResult",
    "DetectionReport",
    "Finding",
    "RewriteDecision",
    "PredictabilityDetector",
    "AIGenerationSignalDetector",
    "SimilarityDetector",
    "CitationDetector",
    "DetectionRunner",
    "PostProcessor",
    "PostProcessResult",
    "PostProcessConfig",
    "GlossaryFilter",
    "AcademicFilter",
    "SeverityAdjustor",
    "DomainProfile",
    "resolve_profile",
    "auto_detect_domain",
    "list_available_profiles",
    "ThresholdConfig",
    "split_sentences",
    "SignalScores",
    "calculate_authorship_concern",
    "extract_signals",
    "build_ai_mitigation_plan",
]
