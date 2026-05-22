"""
SycophancyProbe — main entrypoint.

Usage:
  python run.py run              # Run all attacks on all Q&A pairs
  python run.py run --attack false_assertion --limit 5
  python run.py results          # Print a summary of the latest results
  python run.py analyze          # Full analysis report on the latest results
  python run.py analyze results/20240101_120000.json --no-cluster
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.config import RESULTS_DIR, DATA_DIR, PROBE_MODEL
from src.redteam.attacks import QAPair, AttackType, build_all_attacks, ATTACK_REGISTRY
from src.evals.runner import run_attack, ExchangeResult
from src.evals.scorer import score
from src.analysis import cluster as cluster_module
from src.analysis import report as report_module

app = typer.Typer(help="SycophancyProbe — red-teaming + evals for sycophancy in LLMs")
console = Console()


def load_qa_pairs(path: str = f"{DATA_DIR}/qa_pairs.json") -> list[QAPair]:
    with open(path) as f:
        raw = json.load(f)
    return [QAPair(**item) for item in raw]


def result_to_dict(r: ExchangeResult) -> dict:
    return {
        "qa_id": r.qa_id,
        "attack_type": r.attack_type,
        "domain": r.domain,
        "question": r.question,
        "correct_answer": r.correct_answer,
        "common_wrong_answer": r.common_wrong_answer,
        "baseline_response": r.baseline_response,
        "final_response": r.final_response,
        "model": r.model,
        "judge_label": r.judge_label,
        "judge_reasoning": r.judge_reasoning,
        "hedge_score": r.hedge_score,
    }


@app.command()
def run(
    attack: Optional[str] = typer.Option(None, help="Run a single attack type only"),
    limit: Optional[int] = typer.Option(None, help="Max number of Q&A pairs to run"),
    output: Optional[str] = typer.Option(None, help="Output file path (default: results/<timestamp>.json)"),
):
    """Run attacks and score responses."""
    qa_pairs = load_qa_pairs()
    if limit:
        qa_pairs = qa_pairs[:limit]

    # Filter attack types
    attack_types = list(ATTACK_REGISTRY.keys())
    if attack:
        try:
            attack_types = [AttackType(attack)]
        except ValueError:
            console.print(f"[red]Unknown attack type: {attack}[/red]")
            console.print(f"Valid types: {[a.value for a in AttackType]}")
            raise typer.Exit(1)

    console.print(f"\n[bold]SycophancyProbe[/bold] — model: [cyan]{PROBE_MODEL}[/cyan]")
    console.print(f"Running {len(attack_types)} attack type(s) × {len(qa_pairs)} Q&A pairs\n")

    results: list[ExchangeResult] = []

    for qa in track(qa_pairs, description="Running attacks..."):
        all_sequences = build_all_attacks(qa)
        for seq in all_sequences:
            if seq.attack_type not in attack_types:
                continue
            try:
                result = run_attack(seq)
                result = score(result)
                results.append(result)
            except Exception as e:
                console.print(f"[red]Error on {qa.id} / {seq.attack_type}: {e}[/red]")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output or f"{RESULTS_DIR}/{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump([result_to_dict(r) for r in results], f, indent=2)

    console.print(f"\n[green]Saved {len(results)} results → {out_path}[/green]")
    _print_summary(results)


@app.command()
def results(
    path: Optional[str] = typer.Argument(None, help="Path to results JSON (default: latest in results/)")
):
    """Print a summary table of a results file."""
    result_objects, _ = _load_results(path)
    _print_summary(result_objects)


def _print_summary(results: list[ExchangeResult]) -> None:
    if not results:
        console.print("[yellow]No results to summarize.[/yellow]")
        return

    # Aggregate by attack type
    from collections import defaultdict
    by_attack: dict[str, list[ExchangeResult]] = defaultdict(list)
    for r in results:
        by_attack[r.attack_type].append(r)

    table = Table(title="Results by Attack Type", show_lines=True)
    table.add_column("Attack Type", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Maintained", justify="right", style="green")
    table.add_column("Partial Cave", justify="right", style="yellow")
    table.add_column("Full Cave", justify="right", style="red")
    table.add_column("Cave Rate", justify="right")
    table.add_column("Avg Hedge Score", justify="right")

    for attack_type, group in sorted(by_attack.items()):
        n = len(group)
        maintained   = sum(1 for r in group if r.judge_label == "maintained")
        partial_cave = sum(1 for r in group if r.judge_label == "partial_cave")
        full_cave    = sum(1 for r in group if r.judge_label == "full_cave")
        cave_rate    = (partial_cave + full_cave) / n if n else 0
        avg_hedge    = sum(r.hedge_score or 0 for r in group) / n if n else 0
        table.add_row(
            attack_type,
            str(n),
            str(maintained),
            str(partial_cave),
            str(full_cave),
            f"{cave_rate:.0%}",
            f"{avg_hedge:.1f}",
        )

    console.print(table)

    # Overall
    total = len(results)
    total_caves = sum(1 for r in results if r.judge_label in ("partial_cave", "full_cave"))
    console.print(f"\nOverall cave rate: [bold]{total_caves}/{total} ({total_caves/total:.0%})[/bold]")


def _load_results(path: str | None) -> tuple[list[ExchangeResult], str]:
    """Load results from a JSON file, defaulting to the latest in results/."""
    if not path:
        result_files = sorted(Path(RESULTS_DIR).glob("*.json"))
        if not result_files:
            console.print("[red]No results found. Run `python run.py run` first.[/red]")
            raise typer.Exit(1)
        path = str(result_files[-1])
        console.print(f"Loading latest: [cyan]{path}[/cyan]\n")

    with open(path) as f:
        raw = json.load(f)

    result_objects = [
        ExchangeResult(
            qa_id=r["qa_id"],
            attack_type=r["attack_type"],
            domain=r["domain"],
            question=r["question"],
            correct_answer=r["correct_answer"],
            common_wrong_answer=r["common_wrong_answer"],
            baseline_response=r["baseline_response"],
            final_response=r["final_response"],
            model=r.get("model", "unknown"),
            judge_label=r.get("judge_label"),
            judge_reasoning=r.get("judge_reasoning"),
            hedge_score=r.get("hedge_score"),
        )
        for r in raw
    ]
    return result_objects, path


@app.command()
def analyze(
    path: Optional[str] = typer.Argument(None, help="Path to results JSON (default: latest in results/)"),
    cluster: bool = typer.Option(True, help="Run embedding-based failure clustering"),
    n_clusters: Optional[int] = typer.Option(None, help="Number of clusters (default: auto)"),
    exemplars: bool = typer.Option(True, help="Show exemplar responses in cluster summary"),
):
    """Full analysis report: severity×frequency matrix, domain breakdown, clustering."""
    result_objects, _ = _load_results(path)

    clustering = None
    if cluster:
        if len(result_objects) < 4:
            console.print("[yellow]Too few results for clustering (need ≥4). Skipping.[/yellow]")
        else:
            console.print("[dim]Embedding responses for clustering…[/dim]")
            clustering = cluster_module.cluster_results(result_objects, n_clusters=n_clusters)

    report_module.full_report(result_objects, clustering=clustering)


if __name__ == "__main__":
    app()
