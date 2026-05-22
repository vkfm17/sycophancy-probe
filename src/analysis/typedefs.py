"""Shared type definitions for the analysis layer."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    cave_rate: float
    full_cave_rate: float
    avg_hedge_score: float
    dominant_attack_type: str
    attack_type_distribution: dict[str, int]
    domain_distribution: dict[str, int]
    exemplars: list[str] = field(default_factory=list)


@dataclass
class ClusteringResult:
    n_clusters: int
    silhouette: float
    labels: list[int]
    summaries: list[ClusterSummary]
    embeddings: np.ndarray
