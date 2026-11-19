"""Experiment tracking.

Thin wrapper over MLflow so the training code never imports MLflow directly.

* Normal operation (and Docker): MLflow is installed, runs are written to the
  configured tracking URI and show up in the MLflow UI.
* Offline fallback: if MLflow cannot be imported (e.g. a locked-down sandbox)
  a ``LocalTracker`` writes the same params / metrics / artifacts as JSON under
  ``mlruns_local/`` and logs a loud warning. This exists only so the pipeline
  can be run and tested without internet; it is NOT a replacement for MLflow.

Both trackers expose the same small API used by ``train.py``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger("tracking")

try:  # pragma: no cover - depends on environment
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None
    MLFLOW_AVAILABLE = False


def git_commit() -> str:
    """Return the current git commit hash, or ``unknown`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - any failure just means "no git info"
        return "unknown"


class LocalTracker:
    """Minimal MLflow-compatible tracker that writes JSON files."""

    backend = "local-json"

    def __init__(self, root: Path, experiment: str):
        self.root = Path(root)
        self.experiment = experiment
        self.root.mkdir(parents=True, exist_ok=True)
        self._stack: list[dict[str, Any]] = []
        logger.warning(
            "MLflow is not installed - using LocalTracker (%s). "
            "Install requirements.txt to get real MLflow tracking.",
            self.root,
        )

    @contextmanager
    def start_run(self, run_name: str, nested: bool = False) -> Iterator[str]:
        run_id = uuid.uuid4().hex
        run_dir = self.root / self.experiment / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "run_name": run_name,
            "start": datetime.now(UTC).isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "dir": run_dir,
            "nested": nested,
        }
        self._stack.append(record)
        try:
            yield run_id
        finally:
            record["end"] = datetime.now(UTC).isoformat()
            serialisable = {k: v for k, v in record.items() if k != "dir"}
            (run_dir / "run.json").write_text(json.dumps(serialisable, indent=2))
            self._stack.pop()

    @property
    def _current(self) -> dict[str, Any]:
        return self._stack[-1]

    def log_params(self, params: dict[str, Any]) -> None:
        self._current["params"].update({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._current["metrics"].update(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._current["tags"].update(tags)

    def log_artifact(self, path: str | Path) -> None:
        target = self._current["dir"] / "artifacts"
        target.mkdir(exist_ok=True)
        src = Path(path)
        (target / src.name).write_bytes(src.read_bytes())

    def log_dict(self, obj: dict, name: str) -> None:
        target = self._current["dir"] / "artifacts"
        target.mkdir(exist_ok=True)
        (target / name).write_text(json.dumps(obj, indent=2, default=str))

    def log_model(self, model: Any, name: str, registered_name: str | None = None) -> None:
        import joblib

        target = self._current["dir"] / "artifacts"
        target.mkdir(exist_ok=True)
        joblib.dump(model, target / f"{name}.joblib")


class MlflowTracker:
    """Real MLflow tracker (used whenever MLflow is installed)."""

    backend = "mlflow"

    def __init__(self, tracking_uri: str, experiment: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def start_run(self, run_name: str, nested: bool = False) -> Iterator[str]:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            yield run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, obj: dict, name: str) -> None:
        mlflow.log_dict(obj, name)

    def log_model(self, model: Any, name: str, registered_name: str | None = None) -> None:
        import mlflow.sklearn

        mlflow.sklearn.log_model(
            model, artifact_path=name, registered_model_name=registered_name
        )


def get_tracker(tracking_uri: str, experiment: str, local_root: Path | None = None):
    """Return an MlflowTracker if MLflow is installed, else a LocalTracker."""
    if MLFLOW_AVAILABLE:
        return MlflowTracker(tracking_uri, experiment)
    return LocalTracker(local_root or PROJECT_ROOT / "mlruns_local", experiment)
