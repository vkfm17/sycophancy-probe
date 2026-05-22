from dataclasses import dataclass


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    cave_rate: float  # (partial_cave + full_cave) / size
    full_cave_rate: float  # full_cave / size
    avg_hedge_score: float
    dominant_attack_type: str
    attack_type_distribution: dict[str, int]
    domain_distribution: dict[str, int]
    # Up to 3 representative examples (final_response text)
    exemplars: list[str] = field(default_factory=list)


@dataclass
class ClusteringResult:
    n_clusters: int
    silhouette: float
    labels: list[int]  # cluster label per result, parallel to input list
    summaries: list[ClusterSummary]
    embeddings: np.ndarray  # shape (n_results, embed_dim)
