from __future__ import annotations

from typing import Callable

from .plan import Plan


def stage_progress_percent(start_percent: int | None, end_percent: int | None, step: int, *, steps: int = 3) -> int | None:
    if start_percent is None:
        return None
    start = int(start_percent)
    end = int(end_percent) if end_percent is not None else min(87, start + steps)
    if end <= start:
        return min(87, max(0, start))
    bounded_step = max(0, min(int(steps), int(step)))
    return min(87, max(0, start + round((end - start) * (bounded_step / max(1, int(steps))))))


def writer_progress_message(paragraph_id: str, plan: Plan, *, window: str | None = None) -> str:
    target = f"paragraph {paragraph_id}" + (f" window {window}" if window else "")
    if author_proxy_grounding_required(plan):
        return f"Writing V6 {target} with Author-proxy"
    return f"Writing V6 {target}"


def author_proxy_grounding_required(plan: Plan) -> bool:
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    return isinstance(grounding, dict) and bool(grounding.get("required")) and not plan.paragraph_strategy.get("author_proxy_pack")


def emit_progress(callback: Callable[[int, str], None] | None, percent: int | None, message: str) -> None:
    if callback is not None and percent is not None:
        callback(percent, message)
