from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import residual_patterns
from poc.rewrite_v6.residual_patterns import ResidualIssue, detect_residual_patterns


def test_detect_returns_list_of_issues():
    assert detect_residual_patterns("") == []
    assert detect_residual_patterns("A single short paragraph.") == []
