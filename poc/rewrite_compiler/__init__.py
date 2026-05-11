"""Deterministic rewrite compiler package.

The compiler is a bounded controller layer.  It plans small deterministic
operators, validates and quality-checks them locally, then spends full scans
only on shortlisted finalists.
"""

from .orchestrator import (
    CompilerConfig,
    CompilerDependencies,
    run_rewrite_compiler,
)

__all__ = [
    "CompilerConfig",
    "CompilerDependencies",
    "run_rewrite_compiler",
]
