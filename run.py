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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn

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
        "live_messages": r.live_messages,
        "model": r.model,
        "judge_label": r.judge_label,
        "judge_reasoning": r.judge_reasoning,
        "hedge_score": r.hedge_score,
    }


def _save_results(results: list[ExchangeResult], out_path: str) -> None:
    """Write results to disk, atomically via a temp file."""
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump([result_to_dict(r) for r in results], f, indent=2)
    os.replace(tmp_path, out_path)


def _run_one(seq) -> ExchangeResult:
    """Run a single attack sequence and score it. Called from worker threads."""
    result = run_attack(seq)
    return score(result)


def _load_checkpoint(path: str) -> tuple[list[ExchangeResult], set[tuple[str, str]]]:
    """Load already-completed results from a checkpoint file.

    Returns the list of ExchangeResult objects and a set of (qa_id, attack_type)
    pairs that are already done, used to skip sequences on resume.
    """
    if not os.path.exists(path):
        return [], set()
    with open(path) as f:
        raw = json.load(f)
    completed = [
        ExchangeResult(
            qa_id=r["qa_id"],
            attack_type=r["attack_type"],
            domain=r["domain"],
            question=r["question"],
            correct_answer=r["correct_answer"],
            common_wrong_answer=r["common_wrong_answer"],
            baseline_response=r["baseline_response"],
            final_response=r["final_response"],
            live_messages=r.get("live_messages", []),
            model=r.get("model", "unknown"),
            judge_label=r.get("judge_label"),
            judge_reasoning=r.get("judge_reasoning"),
            hedge_score=r.get("hedge_score"),
        )
        for r in raw
    ]
    done = {(r.qa_id, r.attack_type) for r in completed}
    return completed, done


@app.command()
def run(
    attack: str | None = typer.Option(None, help="Run a single attack type only"),
    difficulty: str | None = typer.Option(None, help="Filter Q&A pairs by difficulty (easy/medium/hard)"),
    language: str | None = typer.Option(None, help="Filter Q&A pairs by language (en/fr/es)"),
    limit: int | None = typer.Option(None, help="Max number of Q&A pairs to run"),
    output: str | None = typer.Option(None, help="Output file path (default: results/<timestamp>.json)"),
    checkpoint: int = typer.Option(10, help="Save to disk every N results (0 = only at end)"),
    workers: int = typer.Option(8, help="Number of parallel workers"),
    resume: bool = typer.Option(False, help="Resume from an existing checkpoint file"),
):
    """Run attacks and score responses."""
    qa_pairs = load_qa_pairs()
    if difficulty:
        qa_pairs = [qa for qa in qa_pairs if qa.difficulty == difficulty]
        if not qa_pairs:
            console.print(f"[red]No Q&A pairs found with difficulty '{difficulty}'. Valid values: easy, medium, hard.[/red]")
            raise typer.Exit(1)
    if language:
        qa_pairs = [qa for qa in qa_pairs if qa.language == language]
        if not qa_pairs:
            console.print(f"[red]No Q&A pairs found with language '{language}'. Available: en, fr, es.[/red]")
            raise typer.Exit(1)
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

    # Build all sequences upfront so we know the total count
    all_sequences = [
        seq
        for qa in qa_pairs
        for seq in build_all_attacks(qa)
        if seq.attack_type in attack_types
    ]

    # Determine output path up front so checkpoints go to the same file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output or f"{RESULTS_DIR}/{timestamp}.json"

    # Resume: load already-completed results and skip those sequences
    results: list[ExchangeResult] = []
    done: set[tuple[str, str]] = set()
    if resume:
        # If no --output given, try to find the latest checkpoint
        resume_path = output
        if not resume_path:
            existing = sorted(Path(RESULTS_DIR).glob("*.json"))
            if existing:
                resume_path = str(existing[-1])
                out_path = resume_path  # continue writing to the same file
        if resume_path:
            results, done = _load_checkpoint(resume_path)
            if results:
                console.print(f"[cyan]Resuming from {resume_path}: {len(results)} results already done[/cyan]")
            else:
                console.print(f"[yellow]No existing results found at {resume_path}, starting fresh.[/yellow]")
        else:
            console.print("[yellow]No checkpoint file found, starting fresh.[/yellow]")

    sequences = [s for s in all_sequences if (s.qa_pair.id, s.attack_type) not in done]

    console.print(f"\n[bold]SycophancyProbe[/bold] — model: [cyan]{PROBE_MODEL}[/cyan]")
    console.print(f"Total sequences: {len(all_sequences)}  |  Already done: {len(done)}  |  Remaining: {len(sequences)}")
    console.print(f"Workers: {workers}  |  Checkpoint every: {checkpoint or '—'}  |  Output: [cyan]{out_path}[/cyan]\n")

    if not sequences:
        console.print("[green]All sequences already complete![/green]")
        _print_summary(results)
        return

    lock = threading.Lock()

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Running attacks...", total=len(sequences))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, seq): seq for seq in sequences}

            for future in as_completed(futures):
                seq = futures[future]
                try:
                    result = future.result()
                    with lock:
                        results.append(result)
                        if checkpoint and len(results) % checkpoint == 0:
                            _save_results(results, out_path)
                            progress.console.print(f"[dim]  ✓ checkpoint: {len(results)}/{len(all_sequences)} saved[/dim]")
                except Exception as e:
                    progress.console.print(f"[red]Error on {seq.qa_pair.id} / {seq.attack_type}: {e}[/red]")
                finally:
                    progress.advance(task_id)

    _save_results(results, out_path)
    console.print(f"\n[green]Saved {len(results)} results → {out_path}[/green]")
    _print_summary(results)


@app.command()
def results(
    path: str | None = typer.Argument(None, help="Path to results JSON (default: latest in results/)")
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
            live_messages=r.get("live_messages", []),
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
    path: str | None = typer.Argument(None, help="Path to results JSON (default: latest in results/)"),
    cluster: bool = typer.Option(True, help="Run embedding-based failure clustering"),
    n_clusters: int | None = typer.Option(None, help="Number of clusters (default: auto)"),
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


@app.command()
def exchanges(
    path: str | None = typer.Argument(None, help="Path to results JSON (default: latest in results/)"),
    label: str | None = typer.Option(None, help="Filter by judge label (maintained/partial_cave/full_cave)"),
    attack: str | None = typer.Option(None, help="Filter by attack type"),
    limit: int | None = typer.Option(None, help="Max number of exchanges to show"),
):
    """Pretty-print the full conversation for each exchange."""
    result_objects, _ = _load_results(path)

    # Apply filters
    if label:
        result_objects = [r for r in result_objects if r.judge_label == label]
    if attack:
        result_objects = [r for r in result_objects if r.attack_type == attack]
    if limit:
        result_objects = result_objects[:limit]

    if not result_objects:
        console.print("[yellow]No results match your filters.[/yellow]")
        return

    LABEL_STYLE = {
        "maintained": "bold green",
        "partial_cave": "bold yellow",
        "full_cave": "bold red",
    }

    for i, r in enumerate(result_objects, 1):
        style = LABEL_STYLE.get(r.judge_label or "", "white")
        console.rule(
            f"[{style}]{i}/{len(result_objects)} · {r.qa_id} · {r.attack_type} · {r.judge_label}[/{style}]"
        )
        console.print(f"[dim]Domain:[/dim] {r.domain}   [dim]Correct answer:[/dim] {r.correct_answer}   [dim]Wrong answer pushed:[/dim] {r.common_wrong_answer}\n")

        if r.live_messages:
            for msg in r.live_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    console.print(f"[bold cyan]USER:[/bold cyan] {content}\n")
                elif role == "assistant":
                    console.print(f"[bold white]MODEL:[/bold white] {content}\n")
        else:
            # Fallback for older results that only have baseline/final
            console.print(f"[bold cyan]USER:[/bold cyan] {r.question}\n")
            console.print(f"[bold white]MODEL (baseline):[/bold white] {r.baseline_response}\n")
            if r.final_response != r.baseline_response:
                console.print(f"[bold white]MODEL (final):[/bold white] {r.final_response}\n")

        if r.judge_reasoning:
            console.print(f"[dim]Judge reasoning:[/dim] {r.judge_reasoning}")
        console.print()


if __name__ == "__main__":
    app()
