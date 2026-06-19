"""Ray Tune hyperparameter sweep over the Ray Train path (SCAFFOLD STUB).

Sweeps distillation hyperparameters (learning rate, weight decay, loss type)
with ``ray.tune.Tuner`` + an ``ASHAScheduler`` for early stopping, logging
every trial to W&B via ``WandbLoggerCallback``. Each trial launches the same
distributed ``TorchTrainer`` used by :mod:`phenomica.ray_train`, following the
Ray "Tune over Train" pattern: the Tuner ``param_space`` feeds a nested
``train_loop_config`` consumed by :func:`phenomica.ray_train.train_fn`.

This is an INDEPENDENT launch path from submitit.

All bodies are intentionally unimplemented (scaffold). Build workers fill
them against the fixed signatures below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.tune import ResultGrid

    from phenomica.configs import (
        DataConfig,
        ModelConfig,
        RayDataConfig,
        RayTrainConfig,
        RayTuneConfig,
        TeacherConfig,
        TrainingConfig,
    )

logger = logging.getLogger(__name__)


def build_search_space(tune_cfg: RayTuneConfig) -> dict[str, Any]:
    """Translate :class:`RayTuneSearchSpace` bounds into Ray Tune domains.

    Maps the primitive bounds on ``tune_cfg.search_space`` to Ray Tune
    sampler objects: ``tune.loguniform`` for ``lr`` and ``weight_decay`` and
    ``tune.choice`` for ``loss_type``. The returned dict is nested under the
    ``train_loop_config`` key expected by
    :func:`phenomica.ray_train.train_fn`.

    Args:
        tune_cfg: Ray Tune config holding the search-space bounds.

    Returns:
        A ``param_space`` dict of the form
        ``{"train_loop_config": {"lr": ..., "weight_decay": ...,
        "loss_type": ...}}``.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_tune.build_search_space is a scaffold stub")


def run_ray_tune(
    model_cfg: ModelConfig,
    teacher_cfg: TeacherConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    ray_train_cfg: RayTrainConfig,
    ray_data_cfg: RayDataConfig,
    ray_tune_cfg: RayTuneConfig,
) -> ResultGrid:
    """Run an ASHA-scheduled hyperparameter sweep of the distillation run.

    Constructs the per-trial ``TorchTrainer`` (reusing
    :func:`phenomica.ray_train.train_fn` and the Ray Data shards from
    :func:`phenomica.ray_data.build_ray_dataset`), wraps it in a
    ``ray.tune.Tuner`` with ``param_space`` from :func:`build_search_space`,
    a ``TuneConfig`` (``num_samples``/``metric``/``mode``/
    ``max_concurrent_trials`` + ``ASHAScheduler(grace_period, max_t,
    reduction_factor)``), and a ``RunConfig`` carrying a
    ``WandbLoggerCallback`` gated on ``training_cfg.use_wandb``. Returns the
    fitted result grid.

    Args:
        model_cfg: Student model config.
        teacher_cfg: DINOv2 teacher config.
        data_cfg: Dataset config.
        training_cfg: Base optimizer/loss/wandb hyperparameters (swept fields
            are overridden per-trial by the search space).
        ray_train_cfg: Ray Train scaling config for each trial.
        ray_data_cfg: Ray Data ingest config.
        ray_tune_cfg: Ray Tune sweep config (samples, metric, ASHA params,
            search space).

    Returns:
        The ``ray.tune.ResultGrid`` from ``tuner.fit()``.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_tune.run_ray_tune is a scaffold stub")
