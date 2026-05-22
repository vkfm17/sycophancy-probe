"""
estimate_cost.py — dry-run cost estimator for SycophancyProbe.

Walks every attack sequence without calling the API and estimates:
  - Number of API calls (probe model + judge model)
  - Input and output token counts (character-based heuristic: chars / 4)
  - Total cost in USD across a range of Claude model tiers

Run with:
  uv run python estimate_cost.py
  uv run python estimate_cost.py --attack false_assertion --limit 5
"""

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from src.redteam.attacks import (
    QAPair, AttackType, build_all_attacks, ATTACK_REGISTRY, AttackSequence
)
from src.evals.scorer import JUDGE_SYSTEM

console = Console()

# ---------------------------------------------------------------------------
# Pricing table (USD per million tokens, as of mid-2025).
# Verify at https://www.anthropic.com/pricing before a large run.
# ---------------------------------------------------------------------------
MODELS = {
    "claude-haiku-4-5-20251001": {
        "input_per_mtok":  0.80,
        "output_per_mtok": 4.00,
    },
    "claude-sonnet-4-6": {
        "input_per_mtok":  3.00,
        "output_per_mtok": 15.00,
    },
    "claude-opus-4-6": {
        "input_per_mtok":  15.00,
        "output_per_mtok": 75.00,
    },
}

# Assumed output token budgets (conservative estimates; adjust if your runs differ)
PROBE_OUTPUT_TOKENS  = 250   # typical factual answer or rebuttal
JUDGE_OUTPUT_TOKENS  = 80    # label + one-sentence reasoning

# Characters per token (rough heuristic for English prose)
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chars_to_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class CallEstimate:
    description: str
    input_tokens: int
    output_tokens: int


def estimate_sequence(seq: AttackSequence) -> list[CallEstimate]:
    """
    Walk an AttackSequence and return one CallEstimate per API call,
    filling in assumed response text for __BASELINE__ / __TURN_N__ placeholders.

    For multi-turn attacks, the conversation grows with each turn, so we
    accumulate context incrementally to get an accurate input token count.
    """
    ASSUMED_RESPONSE = "A" * (PROBE_OUTPUT_TOKENS * CHARS_PER_TOKEN)  # dummy stand-in
    calls: list[CallEstimate] = []
    live_messages: list[dict] = []

    for i, msg in enumerate(seq.messages):
        if msg["role"] == "user":
            live_messages.append(msg)

        elif msg["role"] == "assistant" and msg["content"].startswith("__"):
            # This is a placeholder — estimate a probe call here
            input_text = " ".join(m["content"] for m in live_messages)
            if seq.system_prompt:
                input_text = seq.system_prompt + " " + input_text

            calls.append(CallEstimate(
                description=f"{seq.attack_type} / turn {len(calls) + 1}",
                input_tokens=chars_to_tokens(input_text),
                output_tokens=PROBE_OUTPUT_TOKENS,
            ))
            # Inject the assumed response so subsequent turns see growing context
            live_messages.append({"role": "assistant", "content": ASSUMED_RESPONSE})

        else:
            live_messages.append(msg)

    # Single-turn attacks (no placeholders) — one call at the end
    if not calls:
        input_text = " ".join(m["content"] for m in live_messages)
        if seq.system_prompt:
            input_text = seq.system_prompt + " " + input_text
        calls.append(CallEstimate(
            description=f"{seq.attack_type} / turn 1",
            input_tokens=chars_to_tokens(input_text),
            output_tokens=PROBE_OUTPUT_TOKENS,
        ))

    # Judge call — fixed structure regardless of attack type
    judge_input_tokens = chars_to_tokens(JUDGE_SYSTEM) + chars_to_tokens(
        "Question: X Correct answer: X Wrong answer the user pushed: X AI final response: "
        + ASSUMED_RESPONSE
    )
    calls.append(CallEstimate(
        description=f"{seq.attack_type} / judge",
        input_tokens=judge_input_tokens,
        output_tokens=JUDGE_OUTPUT_TOKENS,
    ))

    return calls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Estimate API cost for a SycophancyProbe run.")
    parser.add_argument("--attack", type=str, default=None,
                        help="Limit to one attack type (e.g. false_assertion)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N Q&A pairs")
    parser.add_argument("--data", type=str, default="data/qa_pairs.json",
                        help="Path to Q&A pairs JSON")
    args = parser.parse_args()

    with open(args.data) as f:
        raw = json.load(f)

    qa_pairs = [QAPair(**item) for item in raw]
    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    attack_filter = None
    if args.attack:
        try:
            attack_filter = AttackType(args.attack)
        except ValueError:
            console.print(f"[red]Unknown attack type: {args.attack}[/red]")
            console.print(f"Valid: {[a.value for a in AttackType]}")
            return

    # Collect all call estimates
    all_calls: list[CallEstimate] = []
    for qa in qa_pairs:
        for seq in build_all_attacks(qa):
            if attack_filter and seq.attack_type != attack_filter:
                continue
            all_calls.extend(estimate_sequence(seq))

    total_input_tokens  = sum(c.input_tokens  for c in all_calls)
    total_output_tokens = sum(c.output_tokens for c in all_calls)
    total_calls         = len(all_calls)

    # Per-attack-type breakdown
    by_attack: dict[str, list[CallEstimate]] = {}
    for c in all_calls:
        key = c.description.split(" / ")[0]
        by_attack.setdefault(key, []).append(c)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    console.print(f"\n[bold]SycophancyProbe — Cost Estimate[/bold]")
    console.print(
        f"  {len(qa_pairs)} Q&A pairs  ×  "
        f"{len([a for a in ATTACK_REGISTRY if not attack_filter or a == attack_filter])} attack type(s)  =  "
        f"[bold]{total_calls}[/bold] API calls total\n"
    )

    # Per-attack-type table
    breakdown = Table(title="Calls & Tokens by Attack Type", box=box.SIMPLE_HEAD)
    breakdown.add_column("Attack Type",    style="bold")
    breakdown.add_column("API Calls",      justify="right")
    breakdown.add_column("Input tokens",   justify="right")
    breakdown.add_column("Output tokens",  justify="right")

    for attack_type, calls in sorted(by_attack.items()):
        breakdown.add_row(
            attack_type,
            str(len(calls)),
            f"{sum(c.input_tokens  for c in calls):,}",
            f"{sum(c.output_tokens for c in calls):,}",
        )

    breakdown.add_section()
    breakdown.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_calls:,}[/bold]",
        f"[bold]{total_input_tokens:,}[/bold]",
        f"[bold]{total_output_tokens:,}[/bold]",
    )
    console.print(breakdown)

    # Cost-by-model table
    cost_table = Table(
        title="Estimated Cost by Model (probe + judge both at same tier)",
        box=box.SIMPLE_HEAD,
        caption="[dim]Verify pricing at anthropic.com/pricing before a large run.[/dim]",
    )
    cost_table.add_column("Model",          style="bold")
    cost_table.add_column("Input $/MTok",   justify="right")
    cost_table.add_column("Output $/MTok",  justify="right")
    cost_table.add_column("Input cost",     justify="right")
    cost_table.add_column("Output cost",    justify="right")
    cost_table.add_column("Total",          justify="right", style="bold green")

    for model, prices in MODELS.items():
        input_cost  = (total_input_tokens  / 1_000_000) * prices["input_per_mtok"]
        output_cost = (total_output_tokens / 1_000_000) * prices["output_per_mtok"]
        total_cost  = input_cost + output_cost
        cost_table.add_row(
            model,
            f"${prices['input_per_mtok']:.2f}",
            f"${prices['output_per_mtok']:.2f}",
            f"${input_cost:.3f}",
            f"${output_cost:.3f}",
            f"${total_cost:.3f}",
        )

    console.print(cost_table)

    # Practical note
    haiku_prices = MODELS["claude-haiku-4-5-20251001"]
    haiku_cost = (
        (total_input_tokens  / 1_000_000) * haiku_prices["input_per_mtok"] +
        (total_output_tokens / 1_000_000) * haiku_prices["output_per_mtok"]
    )
    console.print(
        f"\n[dim]Tip: Run a smoke test first with --limit 3 --attack false_assertion "
        f"(~{3 * 3} API calls). "
        f"Using Haiku as the judge model (JUDGE_MODEL in .env) saves money without "
        f"sacrificing much scoring quality.[/dim]\n"
    )


if __name__ == "__main__":
    main()
