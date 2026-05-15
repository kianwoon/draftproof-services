"""Prompt-template contracts for rewrite V3."""

from .paragraph_portfolio import (
    build_paragraph_portfolio_planner_prompt,
    build_paragraph_portfolio_reconstruction_prompt,
    build_paragraph_portfolio_topk_prompt,
    fallback_paragraph_portfolio_plan,
    parse_paragraph_portfolio_plan,
    parse_paragraph_portfolio_replacements,
    paragraph_portfolio_context,
    validate_paragraph_portfolio_plan,
)

__all__ = [
    "build_paragraph_portfolio_planner_prompt",
    "build_paragraph_portfolio_reconstruction_prompt",
    "build_paragraph_portfolio_topk_prompt",
    "fallback_paragraph_portfolio_plan",
    "parse_paragraph_portfolio_plan",
    "parse_paragraph_portfolio_replacements",
    "paragraph_portfolio_context",
    "validate_paragraph_portfolio_plan",
]
