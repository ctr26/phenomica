"""Ray Tune hyperparameter sweep over the Ray Train path.

Sweeps distillation hyperparameters (learning rate, weight decay, loss type)
with ``ray.tune.Tuner`` + an ``ASHAScheduler`` for early stopping, logging the
sweep to W&B via ``WandbLoggerCallback`` (opt-in). Each trial launches the same
distributed ``TorchTrainer`` used by :mod:`phenomica.ray_train`, following the
Ray "Tune over Train" pattern.

This is an INDEPENDENT launch path from submitit; :mod:`phenomica.train` is
untouched. W&B is opt-in via ``training_cfg.use_wandb`` and runs offline by
default, so the sweep needs no network/auth in tests.

Note on Ray Train v2 (active in this environment): the legacy
``Tuner(trainer_instance, ...)`` API was deprecated in Ray 2.43 and does not
work under Train v2. Instead the Tuner trainable is a *driver function*
(:func:`_train_driver_fn`) that builds the per-trial ``TorchTrainer`` itself.
Two callback layers bridge the two libraries:

* ``ray.tune.integration.ray_train.TuneReportCallback`` (a Train
  ``UserCallback``) is attached to the inner ``TorchTrainer`` ``RunConfig`` so
  per-worker ``ray.train.report`` metrics propagate up to Tune (ASHA reads
  ``ray_tune_cfg.metric`` from them);
* ``ray.air.integrations.wandb.WandbLoggerCallback`` (a Tune ``LoggerCallback``)
  is attached to the *Tuner* ``RunConfig`` and logs every trial's metrics.

The base ``train_loop_config`` (serialized configs) and the Ray Data shards are
injected into the driver via ``tune.with_parameters`` so they are not part of
the per-trial search space.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from phenomica.ray_data import build_ray_dataset
from phenomica.ray_train import (
    _BATCH_SIZE_KEY,
    _MODEL_KEY,
    _PROVENANCE_KEY,
    _RAY_TRAIN_KEY,
    _TEACHER_KEY,
    _TRAIN_SHARD,
    _TRAINING_KEY,
    _VAL_SHARD,
    _build_scaling_config,
    train_fn,
)
from phenomica.reproducibility import run_provenance

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

# Tune ``param_space`` key whose dict is forwarded into the inner TorchTrainer's
# ``train_loop_config`` (the "Tune over Train" nesting convention).
_TRAIN_LOOP_CONFIG_KEY = "train_loop_config"


def build_search_space(tune_cfg: RayTuneConfig) -> dict[str, Any]:
    """Translate :class:`RayTuneSearchSpace` bounds into Ray Tune domains.

    Maps the primitive bounds on ``tune_cfg.search_space`` to Ray Tune sampler
    objects (``tune.loguniform`` for ``lr``/``weight_decay``, ``tune.choice``
    for ``loss_type``), nested under the ``train_loop_config`` key so the
    samples reach :func:`phenomica.ray_train.train_fn` as run-time overrides.

    Args:
        tune_cfg: Ray Tune config holding the search-space bounds.

    Returns:
        A ``param_space`` dict of the form
        ``{"train_loop_config": {"lr": ..., "weight_decay": ...,
        "loss_type": ...}}``.
    """
    from ray import tune

    space = tune_cfg.search_space
    return {
        _TRAIN_LOOP_CONFIG_KEY: {
            "lr": tune.loguniform(space.lr_min, space.lr_max),
            "weight_decay": tune.loguniform(space.weight_decay_min, space.weight_decay_max),
            "loss_type": tune.choice(space.loss_types),
        }
    }


def _train_driver_fn(
    config: dict[str, Any],
    *,
    base_train_loop_config: dict[str, Any],
    datasets: dict[str, Any],
) -> None:
    """Per-trial Tune trainable: build and fit the inner ``TorchTrainer``.

    Implements the Ray Train v2 "Tune over Train" pattern. ``config`` carries
    only this trial's sampled hyperparameters under ``train_loop_config``; they
    are merged onto the serialized base configs (injected via
    ``tune.with_parameters``) and handed to :func:`phenomica.ray_train.train_fn`.
    A ``TuneReportCallback`` on the inner Train ``RunConfig`` propagates the
    worker's ``ray.train.report`` metrics back to Tune for ASHA.

    Args:
        config: The Tune trial config; ``config["train_loop_config"]`` holds the
            sampled ``lr``/``weight_decay``/``loss_type`` overrides.
        base_train_loop_config: The shared, non-swept ``train_loop_config``
            (serialized model/teacher/training/ray_train configs).
        datasets: ``{"train": Dataset, "val": Dataset}`` for the inner trainer.
    """
    from ray.train import RunConfig
    from ray.train.torch import TorchTrainer
    from ray.tune.integration.ray_train import TuneReportCallback

    train_loop_config = {**base_train_loop_config, **config[_TRAIN_LOOP_CONFIG_KEY]}
    ray_train_cfg = train_loop_config[_RAY_TRAIN_KEY]

    trainer = TorchTrainer(
        train_fn,
        train_loop_config=train_loop_config,
        scaling_config=_build_scaling_config_from_dict(ray_train_cfg),
        datasets=datasets,
        run_config=RunConfig(callbacks=[TuneReportCallback()]),
    )
    trainer.fit()


def _build_scaling_config_from_dict(ray_train_cfg: dict[str, Any]) -> Any:
    """Build a ``ScalingConfig`` from a serialized ``RayTrainConfig`` dict.

    The driver receives ``ray_train`` as a plain dict (the train loop config is
    serialized for Tune), so :func:`phenomica.ray_train._build_scaling_config`
    (which takes the dataclass) is wrapped via a lightweight shim type.

    Args:
        ray_train_cfg: Serialized ``RayTrainConfig`` (``asdict`` output).

    Returns:
        A ``ray.train.ScalingConfig``.
    """
    from phenomica.configs import RayTrainConfig

    return _build_scaling_config(RayTrainConfig(**ray_train_cfg))


def _build_tune_config(ray_tune_cfg: RayTuneConfig) -> Any:
    """Build a ``TuneConfig`` with an ASHA early-stopping scheduler.

    Args:
        ray_tune_cfg: Sweep config (samples, metric/mode, ASHA knobs).

    Returns:
        A ``ray.tune.TuneConfig`` driving the sweep.
    """
    from ray.tune import TuneConfig
    from ray.tune.schedulers import ASHAScheduler

    scheduler = ASHAScheduler(
        max_t=ray_tune_cfg.max_t,
        grace_period=ray_tune_cfg.grace_period,
        reduction_factor=ray_tune_cfg.reduction_factor,
    )
    return TuneConfig(
        metric=ray_tune_cfg.metric,
        mode=ray_tune_cfg.mode,
        num_samples=ray_tune_cfg.num_samples,
        max_concurrent_trials=ray_tune_cfg.max_concurrent_trials,
        scheduler=scheduler,
    )


def _build_tune_run_config(training_cfg: TrainingConfig) -> Any:
    """Build the Tuner ``RunConfig`` with an optional W&B logger callback.

    The W&B ``WandbLoggerCallback`` (a Tune ``LoggerCallback``) is attached only
    when ``training_cfg.use_wandb`` is set, so the sweep runs offline by default.

    Args:
        training_cfg: Supplies the W&B project and the on/off flag.

    Returns:
        A ``ray.tune.RunConfig``.
    """
    from ray.tune import RunConfig

    callbacks = []
    if training_cfg.use_wandb:
        from ray.air.integrations.wandb import WandbLoggerCallback

        callbacks.append(WandbLoggerCallback(project=training_cfg.wandb_project))

    return RunConfig(name=training_cfg.wandb_run_name, callbacks=callbacks)


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

    Builds the ``train``/``val`` Ray Data shards once
    (:func:`phenomica.ray_data.build_ray_dataset`) and a base
    ``train_loop_config`` of serialized configs, then sweeps ``lr``/
    ``weight_decay``/``loss_type`` with ``ray.tune.Tuner``. Each trial runs the
    same distributed ``TorchTrainer`` as :func:`phenomica.ray_train.run_ray_train`
    via the Train-v2 "Tune over Train" driver (:func:`_train_driver_fn`), with a
    ``TuneReportCallback`` bridging worker metrics to ASHA and an optional
    ``WandbLoggerCallback`` (gated on ``training_cfg.use_wandb``) on the Tuner.

    Args:
        model_cfg: Student model config.
        teacher_cfg: DINOv2 teacher config.
        data_cfg: Dataset config.
        training_cfg: Base optimizer/loss/W&B hyperparameters (swept fields are
            overridden per-trial by the search space).
        ray_train_cfg: Ray Train scaling config for each trial.
        ray_data_cfg: Ray Data ingest config.
        ray_tune_cfg: Ray Tune sweep config (samples, metric, ASHA params,
            search space).

    Returns:
        The ``ray.tune.ResultGrid`` from ``tuner.fit()``.
    """
    from ray import tune
    from ray.tune import Tuner

    train_ds = build_ray_dataset(data_cfg, ray_data_cfg, split=_TRAIN_SHARD)
    val_ds = build_ray_dataset(data_cfg, ray_data_cfg, split=_VAL_SHARD)

    base_train_loop_config: dict[str, Any] = {
        _MODEL_KEY: asdict(model_cfg),
        _TEACHER_KEY: asdict(teacher_cfg),
        _TRAINING_KEY: asdict(training_cfg),
        _RAY_TRAIN_KEY: asdict(ray_train_cfg),
        _BATCH_SIZE_KEY: data_cfg.batch_size,
        _PROVENANCE_KEY: run_provenance(),
    }

    trainable = tune.with_parameters(
        _train_driver_fn,
        base_train_loop_config=base_train_loop_config,
        datasets={_TRAIN_SHARD: train_ds, _VAL_SHARD: val_ds},
    )

    tuner = Tuner(
        trainable,
        param_space=build_search_space(ray_tune_cfg),
        tune_config=_build_tune_config(ray_tune_cfg),
        run_config=_build_tune_run_config(training_cfg),
    )
    logger.info(
        "Launching Ray Tune sweep: num_samples=%d metric=%s mode=%s max_t=%d",
        ray_tune_cfg.num_samples,
        ray_tune_cfg.metric,
        ray_tune_cfg.mode,
        ray_tune_cfg.max_t,
    )
    return tuner.fit()
