"""Project configuration.

Loads ``params.yaml`` once and exposes it as a plain dict plus a few resolved
paths. Every path can be overridden by an environment variable so the same
code runs unchanged on a laptop and inside Docker.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PARAMS_FILE = Path(os.getenv("PARAMS_FILE", PROJECT_ROOT / "params.yaml"))


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    """Read params.yaml and return it as a nested dict."""
    with open(PARAMS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(relative_path: str, env_var: str | None = None) -> Path:
    """Resolve a project-relative path, letting an env var override it."""
    if env_var and os.getenv(env_var):
        return Path(os.environ[env_var])
    return PROJECT_ROOT / relative_path


def setup_logging(name: str = "startup_outcome") -> logging.Logger:
    """Configure root logging once and return a named logger."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
