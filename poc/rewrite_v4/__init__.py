"""Experimental rewrite pipeline V4.

V4 keeps the scanner as the diagnostic source, but inserts a normalizer between
scanner findings and LLM prompts so models receive editorial tasks rather than
raw detector language.
"""

from .experiment import run_v4_experiment, run_v4_fast_rewrite, run_v4_iterative_rewrite

__all__ = ["run_v4_experiment", "run_v4_fast_rewrite", "run_v4_iterative_rewrite"]
