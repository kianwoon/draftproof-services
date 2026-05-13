"""Structured-output helpers for rewrite V2 LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any

from llm.gateway import model_supports_structured_outputs

from .strategy import clean_candidate_output


def structured_json_request_options(model: str | None, response_format: dict[str, Any]) -> dict[str, Any]:
    if model_supports_structured_outputs(model):
        return {
            "response_format": response_format,
            "provider": {"require_parameters": True},
            "structured_output_mode": "required_schema",
        }
    return {
        "response_format": None,
        "provider": None,
        "structured_output_mode": "prompt_json_fallback",
    }


def json_parse_diagnostics(raw: str) -> dict[str, Any]:
    text = clean_candidate_output(raw)
    if not text:
        return {
            "ok": False,
            "reason": "empty_response",
            "payload": {},
            "raw_preview": "",
        }
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {"ok": True, "reason": None, "payload": payload, "raw_preview": text[:600]}
        return {
            "ok": False,
            "reason": "json_not_object",
            "payload": {},
            "raw_preview": text[:600],
        }
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {
                "ok": False,
                "reason": "no_json_object_found",
                "payload": {},
                "raw_preview": text[:600],
            }
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return {"ok": True, "reason": "extracted_json_object", "payload": payload, "raw_preview": text[:600]}
            return {
                "ok": False,
                "reason": "extracted_json_not_object",
                "payload": {},
                "raw_preview": text[:600],
            }
        except json.JSONDecodeError:
            return {
                "ok": False,
                "reason": "json_decode_error",
                "payload": {},
                "raw_preview": text[:600],
            }


def json_from_response(raw: str) -> dict[str, Any]:
    payload = json_parse_diagnostics(raw).get("payload")
    return payload if isinstance(payload, dict) else {}
