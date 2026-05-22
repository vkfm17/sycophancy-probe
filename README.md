# SycophancyProbe

A red-teaming + evals tool for systematically studying sycophantic behavior in large language models.

Built as a portfolio project for an Applied AI Engineer role — see [PROJECT.md](PROJECT.md) for the full design rationale.

---

## Setup

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## Usage

```bash
# Run all 4 attack types on all Q&A pairs
uv run python run.py run

# Smoke test — 2 pairs, one attack type
uv run python run.py run --limit 2 --attack false_assertion

# Print a summary table of the latest results
uv run python run.py results

# Full analysis report: severity×frequency matrix, domain breakdown, clustering
uv run python run.py analyze

# Analysis without embedding step (faster, no sentence-transformers)
uv run python run.py analyze --no-cluster

# Force a specific number of clusters
uv run python run.py analyze --n-clusters 4
```

---

## Project Structure

```
sycophancy-probe/
├── src/
│   ├── redteam/
│   │   ├── attacks.py      # 4 attack generators
│   │   └── typedefs.py     # Shared types
│   ├── evals/
│   │   ├── runner.py       # Calls the subject model, fills in conversation turns
│   │   └── scorer.py       # Claude-as-judge + hedge language detector
│   └── analysis/
│       ├── cluster.py      # Sentence-transformer embeddings + KMeans clustering
│       └── report.py       # Severity×frequency matrix, domain breakdown, hedge distribution
├── data/
│   └── qa_pairs.json       # Ground-truth Q&A pairs
├── results/                # Output from runs (gitignored)
├── run.py                  # CLI entrypoint (run / results / analyze)
└── pyproject.toml
```

---

## Attack Types

| Attack | Mechanism |
|---|---|
| `false_assertion` | Confidently asserts the wrong answer after a correct response |
| `social_pressure` | Multi-turn escalating pushback ("are you sure?" → "my expert friend disagrees") |
| `authority_framing` | Prepends an expert identity claim to the question |
| `leading_question` | Embeds the wrong answer as a premise in the question |

## Scoring

Each response is classified by Claude-as-judge as `maintained`, `partial_cave`, or `full_cave`, and scored by a regex hedge-language detector for a soft continuous signal.

## Analysis

The `analyze` command produces four views:

- **Severity × frequency matrix** — which attack types are both frequent and severe (the "dangerous" quadrant)
- **Per-domain breakdown** — cave rates by subject domain, sorted worst-first
- **Hedge score distribution** — mean/max hedge counts per judge label; a high hedge score on `maintained` responses indicates hedging without capitulating, which is its own interesting finding
- **Failure pattern clusters** — responses embedded with `all-MiniLM-L6-v2` and clustered with KMeans (k auto-selected by silhouette score), with worst-case exemplars surfaced per cluster
