"""Reproducibility provenance and W&B artifact logging."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def run_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture run provenance: git SHA, dirty state, Python + library versions.

    Args:
        extra: Optional dict to merge into the provenance record
            (e.g. seed, dataset name/hash).

    Returns:
        Dict with ``git_sha``, ``git_dirty``, ``python_version``,
        ``torch_version``, ``timm_version``, ``hydra_zen_version``,
        ``pydantic_version``, and any keys from ``extra``.
    """
    provenance: dict[str, Any] = {}

    # Git SHA + dirty flag
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        provenance["git_sha"] = sha
        # Check if working tree is dirty
        subprocess.check_output(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        )
        subprocess.check_output(
            ["git", "diff", "--cached", "--quiet"], stderr=subprocess.DEVNULL
        )
        provenance["git_dirty"] = False
    except subprocess.CalledProcessError:
        # Either not a git repo, or working tree is dirty
        if "git_sha" not in provenance:
            provenance["git_sha"] = "unknown"
        provenance["git_dirty"] = True
    except FileNotFoundError:
        provenance["git_sha"] = "unknown"
        provenance["git_dirty"] = False

    # Python + library versions
    provenance["python_version"] = sys.version.split()[0]

    try:
        import torch

        provenance["torch_version"] = torch.__version__
    except ImportError:
        provenance["torch_version"] = "unknown"

    try:
        import timm

        provenance["timm_version"] = timm.__version__
    except ImportError:
        provenance["timm_version"] = "unknown"

    try:
        import hydra_zen

        provenance["hydra_zen_version"] = hydra_zen.__version__
    except ImportError:
        provenance["hydra_zen_version"] = "unknown"

    try:
        import pydantic

        provenance["pydantic_version"] = pydantic.__version__
    except ImportError:
        provenance["pydantic_version"] = "unknown"

    # Merge extra dict if provided
    if extra is not None:
        provenance.update(extra)

    return provenance


def dataset_hash(paths_or_names: list[str] | str) -> str:
    """Compute a deterministic short hash of dataset identifiers.

    Args:
        paths_or_names: List of dataset paths or a single dataset name.
            Order is normalized (sorted) for stability.

    Returns:
        Hex digest (first 16 chars of SHA256) for compact identifiers.
    """
    if isinstance(paths_or_names, str):
        paths_or_names = [paths_or_names]

    # Sort for order-stability
    sorted_paths = sorted(paths_or_names)
    content = "\n".join(sorted_paths).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def log_artifacts(
    run: Any,
    *,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Log config/checkpoint/metrics artifacts to a W&B run.

    Safe no-op when ``run is None`` (for tests or ``use_wandb=False``).

    Args:
        run: W&B run object (e.g. from ``wandb.init()``), or ``None``.
        config_path: Optional path to config file to log as artifact.
        checkpoint_path: Optional path to checkpoint file to log as artifact.
        metrics: Optional dict of final metrics to log as JSON artifact.
    """
    if run is None:
        return

    import wandb

    if config_path is not None:
        config_artifact = wandb.Artifact("config", type="config")
        config_artifact.add_file(config_path)
        run.log_artifact(config_artifact)
        logger.info("Logged config artifact: %s", config_path)

    if checkpoint_path is not None:
        checkpoint_artifact = wandb.Artifact("checkpoint", type="model")
        checkpoint_artifact.add_file(checkpoint_path)
        run.log_artifact(checkpoint_artifact)
        logger.info("Logged checkpoint artifact: %s", checkpoint_path)

    if metrics is not None:
        import json
        import tempfile

        metrics_artifact = wandb.Artifact("metrics", type="result")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(metrics, f, indent=2)
            f.flush()
            metrics_artifact.add_file(f.name)
        run.log_artifact(metrics_artifact)
        logger.info("Logged metrics artifact")
