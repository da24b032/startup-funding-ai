"""
CLI entry point for the MLOps template.

Commands:
    mlops generate   — generate synthetic churn dataset
    mlops train      — run training pipeline (all models or one)
    mlops serve      — start the inference server
    mlops drift      — run drift check on a CSV file
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="End-to-End MLOps Template — train, serve, monitor.")
console = Console()


@app.command()
def generate(
    n: int = typer.Option(5000, "--n-samples", "-n"),
    output: str = typer.Option("./data/raw/churn_data.csv", "--output", "-o"),
) -> None:
    """Generate a synthetic churn dataset."""
    from src.data.generate import generate_churn_dataset, save_dataset
    with console.status("Generating dataset…"):
        df = generate_churn_dataset(n_samples=n)
        save_dataset(df, output)
    console.print(f"[green]Saved {len(df)} rows → {output}[/green]")
    churn_rate = df["churn"].mean()
    console.print(f"Churn rate: [bold]{churn_rate:.1%}[/bold]")


@app.command()
def train(
    model: str = typer.Option("random_forest", "--model", "-m",
                               help="random_forest | logistic_regression | gradient_boosting"),
    all_models: bool = typer.Option(False, "--all", help="Train all model types and compare."),
) -> None:
    """Train the churn classifier and log to MLflow."""
    from src.training.train import train as run_train, MODELS

    model_types = list(MODELS.keys()) if all_models else [model]
    results = []

    for mt in model_types:
        console.rule(f"[bold]Training: {mt}[/bold]")
        with console.status(f"Training {mt}…"):
            metrics = run_train(model_type=mt)
        results.append({"model": mt, **metrics})
        console.print(f"  AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    if len(results) > 1:
        table = Table(title="Model Comparison")
        for col in ["model", "accuracy", "roc_auc", "f1", "precision", "recall"]:
            table.add_column(col)
        for r in results:
            table.add_row(*[str(r.get(c, "")) for c in ["model", "accuracy", "roc_auc", "f1", "precision", "recall"]])
        console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn
    uvicorn.run("src.serving.app:app", host=host, port=port, reload=reload)


@app.command()
def drift(
    csv_path: str = typer.Option(..., "--data", "-d", help="CSV file with inference data."),
) -> None:
    """Check for feature drift between baseline and a new CSV file."""
    import pandas as pd
    from src.drift.detector import detect_drift
    from src.features.engineering import prepare_features

    df = pd.read_csv(csv_path)
    X, _ = prepare_features(df)
    results = detect_drift(X)

    table = Table(title="Drift Report")
    table.add_column("Feature")
    table.add_column("KS Stat")
    table.add_column("p-value")
    table.add_column("Drifted?")

    for feature, r in results.items():
        drifted_str = "[red]YES[/red]" if r["drifted"] else "[green]NO[/green]"
        table.add_row(feature, str(r["ks_statistic"]), str(r["p_value"]), drifted_str)

    console.print(table)


if __name__ == "__main__":
    app()
