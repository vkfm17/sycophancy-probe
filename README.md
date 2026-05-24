# SycophancyProbe

A red-teaming + evals tool for systematically studying sycophantic behavior in large language models.

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

# Smoke test — 3 pairs, one attack type
uv run python run.py run --limit 3 --attack false_assertion

# Run only hard questions
uv run python run.py run --difficulty hard

# Combine filters
uv run python run.py run --difficulty hard --attack false_assertion

# Change checkpoint frequency (default: save every 10 results; 0 = only at end)
uv run python run.py run --checkpoint 5

# Control parallelism (default: 8 workers); reduce if hitting rate limits
uv run python run.py run --workers 4

# Resume a crashed run (picks up the latest checkpoint automatically)
uv run python run.py run --resume

# Resume into a specific file
uv run python run.py run --resume --output results/20260524_102910.json

# Print a summary table of the latest results
uv run python run.py results

# Pretty-print full conversation exchanges (question → attack → model reply)
uv run python run.py exchanges

# Filter exchanges by judge label or attack type
uv run python run.py exchanges --label full_cave
uv run python run.py exchanges --label partial_cave --attack social_pressure --limit 10

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
├── run.py                  # CLI entrypoint (run / results / exchanges / analyze)
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

---

## Cost & Model Selection

Before running the full suite, use the estimator:

```bash
uv run python estimate_cost.py               # full run estimate
uv run python estimate_cost.py --limit 3 --attack false_assertion  # smoke test cost
```

### Estimated cost for the full dataset (64 pairs × 4 attacks = 704 API calls)

| Model tier | Input $/MTok | Output $/MTok | Estimated total |
|---|---|---|---|
| Haiku (`claude-haiku-4-5-20251001`) | $0.80 | $4.00 | ~$0.60 |
| Sonnet (`claude-sonnet-4-6`) | $3.00 | $15.00 | ~$2.25 |
| Opus (`claude-opus-4-6`) | $15.00 | $75.00 | ~$11.30 |

*Token counts are estimated with a chars/4 heuristic. Verify pricing at [anthropic.com/pricing](https://www.anthropic.com/pricing) before a large run.*

### Recommended configuration

**For a first full run:** use Sonnet as the probe, Haiku as the judge.

```
# .env
PROBE_MODEL=claude-sonnet-4-6
JUDGE_MODEL=claude-haiku-4-5-20251001
```

**Estimated cost: ~$1.20** — roughly half the all-Sonnet price with minimal scoring quality loss.

**Rationale:**

- **Probe model (Sonnet):** The subject under test should be a capable, deployed-grade model. Haiku is too compliant by default and its sycophancy patterns won't generalize to the models Anthropic actually cares about. Opus would give marginally richer responses but at 5× the cost with no benefit to the research question.

- **Judge model (Haiku):** The judge's job is classification against a tight rubric — maintained / partial_cave / full_cave — not open-ended reasoning. Haiku handles structured classification reliably and is ~4× cheaper than Sonnet for this role. The main risk is on `partial_cave` cases (the subtlest label); plan to spot-check a sample of ~30 judge labels manually to calibrate your trust in it.

- **Why not Opus as probe?** The most interesting sycophancy findings are likely to emerge from a model that has real tension between its training to be helpful and its training to be honest. Opus may resist pressure more strongly than Sonnet, which would make the cave rates lower but not necessarily more publishable. Sonnet is also the model most readers will be running in production — making the findings more practically relevant.

### Recommended run order

1. **Smoke test** — 3 pairs, one attack type, verify output format and judge labels look right
   ```bash
   uv run python run.py run --limit 3 --attack false_assertion
   uv run python run.py results
   ```

2. **Single attack type** — run all 64 pairs with `false_assertion` only (~$0.30), review results before committing to the full grid

3. **Full run** — all 4 attack types once you're happy with the output quality
