"""Single source of truth for AI-essay generators. Adding a new model (gpt-6) is a
one-row edit to models.json — no code change, no hardcode in logic."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = HERE / "models.json"

# provider -> (base_url, api_key_env). OpenRouter is the proven-clean path.
PROVIDER_BASE_URLS: dict[str, tuple[str, str]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

REQUIRED = ("id", "provider", "family", "n_per_topic")

@dataclass(frozen=True)
class Generator:
    id: str
    provider: str
    family: str
    n_per_topic: int

def load_generators(path: Path | None = None) -> list[Generator]:
    data = json.loads((path or DEFAULT_MODELS).read_text())
    out: list[Generator] = []
    for row in data.get("generators", []):
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"generator row {row!r} missing keys: {missing}")
        if row["provider"] not in PROVIDER_BASE_URLS:
            raise ValueError(f"unknown provider {row['provider']!r}; known: {list(PROVIDER_BASE_URLS)}")
        out.append(Generator(row["id"], row["provider"], row["family"], int(row["n_per_topic"])))
    return out
