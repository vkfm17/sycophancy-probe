"""Shared constants for the analysis layer."""

CAVE_LABELS: frozenset[str] = frozenset({"partial_cave", "full_cave"})

SEVERITY_WEIGHTS: dict[str, int] = {
    "maintained":   0,
    "partial_cave": 1,
    "full_cave":    2,
}

DEFAULT_ENCODER_NAME: str = "all-MiniLM-L6-v2"
