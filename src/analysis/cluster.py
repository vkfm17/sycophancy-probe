"""
Failure pattern clustering for SycophancyProbe.

Embeds the final response text from each ExchangeResult using a
sentence-transformer, then clusters with KMeans. Each cluster is
summarised by its dominant attack type, capitulation rate, and
representative examples — making it easy to spot structural patterns
across failure modes rather than just per-attack-type statistics.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.analysis.constants import DEFAULT_ENCODER_NAME
from src.analysis.typedefs import ClusteringResult, ClusterSummary


# Lazy import: sentence-transformers is heavy, only load when needed.
def _get_encoder(model_name: str = DEFAULT_ENCODER_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _severity_weight(label: str | None) -> float:
    """Map judge label to a numeric severity: full_cave=2, partial_cave=1, maintained=0."""
    return {"full_cave": 2.0, "partial_cave": 1.0, "maintained": 0.0}.get(
        label or "", 0.0
    )


def _choose_k(embeddings: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """
    Pick the number of clusters by maximising silhouette score over [k_min, k_max].
    Falls back to k_min if n_samples is too small.
    """
    n = embeddings.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return k_min

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_results(results: list, n_clusters: int | None = None) -> ClusteringResult:
    """
    Cluster a list of ExchangeResult objects by their final response embeddings.

    Args:
        results:    List of ExchangeResult (must have .final_response, .judge_label,
                    .attack_type, .domain, .hedge_score).
        n_clusters: If None, auto-select k via silhouette score.

    Returns:
        ClusteringResult with per-cluster summaries and raw labels.
    """
    if not results:
        raise ValueError("No results to cluster.")

    encoder = _get_encoder()
    texts = [r.final_response for r in results]
    raw_embeddings = encoder.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    )
    embeddings = normalize(
        raw_embeddings
    )  # cosine similarity via L2-normalised dot product

    k = n_clusters or _choose_k(embeddings)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings).tolist()

    sil = silhouette_score(embeddings, labels) if k > 1 else 0.0

    # Build per-cluster summaries
    summaries: list[ClusterSummary] = []
    for cid in range(k):
        indices = [i for i, lbl in enumerate(labels) if lbl == cid]
        group = [results[i] for i in indices]

        size = len(group)
        cave_count = sum(
            1 for r in group if r.judge_label in ("partial_cave", "full_cave")
        )
        full_cave_count = sum(1 for r in group if r.judge_label == "full_cave")
        avg_hedge = sum(r.hedge_score or 0.0 for r in group) / size

        attack_dist: dict[str, int] = {}
        domain_dist: dict[str, int] = {}
        for r in group:
            attack_dist[r.attack_type] = attack_dist.get(r.attack_type, 0) + 1
            domain_dist[r.domain] = domain_dist.get(r.domain, 0) + 1

        dominant_attack = max(attack_dist, key=lambda k: attack_dist[k])

        # Pick up to 3 exemplars: prefer full_cave > partial_cave > maintained,
        # then shortest (most concise) within each tier.
        def exemplar_priority(r):
            return (_severity_weight(r.judge_label) * -1, len(r.final_response))

        exemplars = [
            r.final_response[:400] + ("…" if len(r.final_response) > 400 else "")
            for r in sorted(group, key=exemplar_priority)[:3]
        ]

        summaries.append(
            ClusterSummary(
                cluster_id=cid,
                size=size,
                cave_rate=cave_count / size,
                full_cave_rate=full_cave_count / size,
                avg_hedge_score=avg_hedge,
                dominant_attack_type=dominant_attack,
                attack_type_distribution=attack_dist,
                domain_distribution=domain_dist,
                exemplars=exemplars,
            )
        )

    # Sort clusters by cave_rate descending (worst first)
    summaries.sort(key=lambda s: s.cave_rate, reverse=True)

    return ClusteringResult(
        n_clusters=k,
        silhouette=round(sil, 3),  # type: ignore
        labels=labels,
        summaries=summaries,
        embeddings=embeddings,
    )
