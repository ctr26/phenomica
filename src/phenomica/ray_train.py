"""Ray Train (TorchTrainer) distributed distillation path.

Distributed distillation via ``ray.train.torch.TorchTrainer``. The per-worker
loop reuses the existing distillation building blocks unchanged:
:func:`phenomica.models.build_model`, :class:`phenomica.teacher.DINOv2Teacher`,
and the loss classes in :mod:`phenomica.losses`. Data arrives as Ray Data
shards built by :func:`phenomica.ray_data.build_ray_dataset`.

This is an INDEPENDENT launch path from submitit; :mod:`phenomica.train` (the
submitit/Hydra path) is untouched. W&B is opt-in via ``training_cfg.use_wandb``
and attached at the driver as a Ray Train v2 ``UserCallback`` (built by
:func:`_make_wandb_callback`), so workers never touch W&B and the path runs
offline by default.

Note on Ray Train v2 vs the legacy AIR API: ``RunConfig.callbacks`` accepts
only :class:`ray.train.UserCallback` subclasses (the legacy
``ray.air.integrations.wandb.WandbLoggerCallback`` is a Tune ``LoggerCallback``
and is rejected by v2), and ``Result.metrics`` is populated only when a
checkpoint is reported alongside the metrics -- hence rank 0 reports a
checkpoint each epoch.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from phenomica import teacher as teacher_module
from phenomica.losses import DistillationLoss, MultiFunctionDistillationLoss
from phenomica.models import build_model
from phenomica.ray_data import IMAGE_COLUMN, build_ray_dataset
from phenomica.reproducibility import run_provenance

if TYPE_CHECKING:
    from ray.train import Result

    from phenomica.configs import (
        DataConfig,
        ModelConfig,
        RayDataConfig,
        RayTrainConfig,
        TeacherConfig,
        TrainingConfig,
    )
    from phenomica.teacher import DINOv2Teacher

logger = logging.getLogger(__name__)

# Ray Data shard names registered on the TorchTrainer and read back per worker.
_TRAIN_SHARD = "train"
_VAL_SHARD = "val"

# train_loop_config keys carrying the serialized phenomica configs.
_MODEL_KEY = "model"
_TEACHER_KEY = "teacher"
_TRAINING_KEY = "training"
_RAY_TRAIN_KEY = "ray_train"
_BATCH_SIZE_KEY = "batch_size"
_PROVENANCE_KEY = "provenance"

# Checkpoint file name written by rank 0 and reported with the epoch metrics.
_CHECKPOINT_NAME = "student.pt"

# Fallback per-iteration batch size when the driver does not inject one
# (e.g. ``train_fn`` invoked directly in a unit test).
_DEFAULT_BATCH_SIZE = 8


def _sync_head_dims(model_cfg: Any, teacher: DINOv2Teacher, expected_dim: int) -> None:
    """Point the student head dims at the loaded teacher ``embed_dim``.

    Mirrors the dim-sync invariant in
    :class:`phenomica.trainer.DistillationTrainer`: a mismatch between the
    loaded teacher and ``TeacherConfig.embed_dim`` is a configuration error
    surfaced before any compute.

    Args:
        model_cfg: Student model config (mutated in place).
        teacher: Instantiated frozen teacher.
        expected_dim: ``TeacherConfig.embed_dim`` the teacher must match.

    Raises:
        ValueError: If the loaded teacher ``embed_dim`` disagrees with
            ``expected_dim``.
    """
    embed_dim = teacher.embed_dim
    if embed_dim != expected_dim:
        raise ValueError(
            f"teacher_cfg.embed_dim={expected_dim} disagrees with loaded "
            f"DINOv2 embed_dim={embed_dim}"
        )
    model_cfg.projection_dim = embed_dim
    model_cfg.teacher_cls_dim = embed_dim
    model_cfg.teacher_patch_dim = embed_dim


def _build_criterion(model_cfg: Any, training_cfg: Any) -> torch.nn.Module:
    """Select the distillation loss matching the student variant.

    Args:
        model_cfg: Student model config (``variant`` selects the loss).
        training_cfg: Training config supplying ``loss_type`` and weights.

    Returns:
        A loss module mirroring :class:`phenomica.trainer.DistillationTrainer`.
    """
    if model_cfg.variant == "multifunction":
        return MultiFunctionDistillationLoss(loss_type=training_cfg.loss_type)
    return DistillationLoss(
        loss_type=training_cfg.loss_type,
        mse_weight=training_cfg.mse_weight,
        cosine_weight=training_cfg.cosine_weight,
    )


def _run_epoch(
    *,
    model: torch.nn.Module,
    teacher: DINOv2Teacher,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    shard: Any,
    device: torch.device,
    batch_size: int,
    train: bool,
) -> float:
    """Run one pass over a Ray Data shard and return the mean loss.

    Args:
        model: The (DDP-wrapped) student.
        teacher: Frozen teacher producing distillation targets.
        criterion: Distillation loss module.
        optimizer: Optimizer for the train pass; ``None`` for validation.
        shard: A ``ray.data.DataIterator`` from ``get_dataset_shard``.
        device: Worker compute device.
        batch_size: Per-iteration batch size for ``iter_torch_batches``.
        train: Whether to backprop (train) or run under ``no_grad`` (val).

    Returns:
        Sample-weighted mean loss over the shard (0.0 if the shard is empty).
    """
    model.train(train)
    loss_sum = 0.0
    sample_count = 0

    for batch in shard.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32):
        images = batch[IMAGE_COLUMN].to(device, non_blocking=True)
        with torch.no_grad():
            teacher_outputs = teacher(images)

        with torch.set_grad_enabled(train):
            student_output = model(images)
            loss = criterion(student_output, teacher_outputs)

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_n = images.size(0)
        loss_sum += loss.item() * batch_n
        sample_count += batch_n

    return loss_sum / sample_count if sample_count else 0.0


def _report_epoch(model: torch.nn.Module, metrics: dict[str, float]) -> None:
    """Report epoch metrics, attaching a rank-0 checkpoint.

    A checkpoint is required for ``Result.metrics`` to be populated in Ray
    Train v2, so rank 0 writes the unwrapped student state dict and reports it;
    other ranks report metrics only.

    Args:
        model: The (DDP-wrapped) student whose state is checkpointed.
        metrics: Scalar metrics to report (``train_loss``/``val_loss``/...).
    """
    import ray.train
    from ray.train import Checkpoint

    if ray.train.get_context().get_world_rank() != 0:
        ray.train.report(metrics)
        return

    student = model.module if hasattr(model, "module") else model
    with tempfile.TemporaryDirectory() as ckpt_dir:
        torch.save(student.state_dict(), f"{ckpt_dir}/{_CHECKPOINT_NAME}")
        ray.train.report(metrics, checkpoint=Checkpoint.from_directory(ckpt_dir))


def train_fn(config: dict[str, Any]) -> None:
    """Per-worker Ray Train loop (runs on every distributed worker).

    ``config`` is the ``TorchTrainer`` ``train_loop_config``. It carries the
    serialized phenomica configs under ``model``/``teacher``/``training``/
    ``ray_train`` (plain dicts) plus optional Tune-injected overrides
    ``lr``/``weight_decay``/``loss_type`` merged at the top level.

    The loop rebuilds the student via :func:`phenomica.models.build_model`,
    wraps it with ``ray.train.torch.prepare_model`` (DDP), instantiates a
    frozen :class:`phenomica.teacher.DINOv2Teacher`, syncs head dims to the
    teacher ``embed_dim``, selects the loss by ``model.variant``, then iterates
    the ``train`` (and optional ``val``) shards from
    ``ray.train.get_dataset_shard``, calling ``ray.train.report`` with
    ``train_loss``/``val_loss`` each epoch (the metric ASHA/Tune consume).

    Args:
        config: The ``train_loop_config`` dict passed by ``TorchTrainer``.
    """
    import ray.train
    from ray.train.torch import get_device, prepare_model

    from phenomica.configs import ModelConfig, TeacherConfig, TrainingConfig

    model_cfg = ModelConfig(**config[_MODEL_KEY])
    teacher_cfg = TeacherConfig(**config[_TEACHER_KEY])
    training_cfg = TrainingConfig(**config[_TRAINING_KEY])
    ray_train_cfg = config[_RAY_TRAIN_KEY]

    # Tune-injected overrides (also valid as plain run-time tweaks).
    learning_rate = config.get("lr", training_cfg.learning_rate)
    weight_decay = config.get("weight_decay", training_cfg.weight_decay)
    if "loss_type" in config:
        training_cfg.loss_type = config["loss_type"]

    device = get_device()

    extract_layers = model_cfg.teacher_layers if model_cfg.variant == "multifunction" else None
    # Resolve via the module attribute (late binding) so tests can swap in a
    # network-free fake teacher in the worker process.
    teacher = teacher_module.DINOv2Teacher(
        model_name=teacher_cfg.model_name, extract_layers=extract_layers
    ).to(device)

    _sync_head_dims(model_cfg, teacher, teacher_cfg.embed_dim)

    model = prepare_model(build_model(model_cfg))
    criterion = _build_criterion(model_cfg, training_cfg).to(device)

    optimizer_cls = torch.optim.AdamW if training_cfg.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    batch_size = config.get(_BATCH_SIZE_KEY, _DEFAULT_BATCH_SIZE)
    train_shard = ray.train.get_dataset_shard(_TRAIN_SHARD)
    val_shard = ray.train.get_dataset_shard(_VAL_SHARD)

    for epoch in range(ray_train_cfg["max_epochs"]):
        train_loss = _run_epoch(
            model=model,
            teacher=teacher,
            criterion=criterion,
            optimizer=optimizer,
            shard=train_shard,
            device=device,
            batch_size=batch_size,
            train=True,
        )

        metrics: dict[str, float] = {"train_loss": train_loss, "epoch": epoch + 1}
        if val_shard is not None:
            metrics["val_loss"] = _run_epoch(
                model=model,
                teacher=teacher,
                criterion=criterion,
                optimizer=None,
                shard=val_shard,
                device=device,
                batch_size=batch_size,
                train=False,
            )

        _report_epoch(model, metrics)


def _build_scaling_config(ray_train_cfg: RayTrainConfig) -> Any:
    """Translate :class:`RayTrainConfig` into a ``ScalingConfig``.

    Args:
        ray_train_cfg: Ray Train scaling knobs.

    Returns:
        A ``ray.train.ScalingConfig`` with per-worker CPU (and GPU when
        ``use_gpu``) reservations.
    """
    from ray.train import ScalingConfig

    resources = {"CPU": ray_train_cfg.cpus_per_worker}
    if ray_train_cfg.use_gpu:
        resources["GPU"] = ray_train_cfg.gpus_per_worker

    return ScalingConfig(
        num_workers=ray_train_cfg.num_workers,
        use_gpu=ray_train_cfg.use_gpu,
        resources_per_worker=resources,
    )


def _make_wandb_callback(project: str, run_name: str | None) -> Any:
    """Build a driver-side W&B ``UserCallback`` for Ray Train v2.

    Defined lazily so importing :mod:`phenomica` never pulls Ray. The callback
    starts one rank-0 W&B run and logs each reported metrics dict; ``wandb`` is
    imported inside the callback so the path stays offline-safe until enabled.

    Args:
        project: W&B project name.
        run_name: Optional W&B run name.

    Returns:
        An instance of a ``ray.train.UserCallback`` subclass.
    """
    from ray.train import UserCallback

    class WandbUserCallback(UserCallback):
        """Logs Ray Train reported metrics to a single W&B run."""

        def __init__(self) -> None:
            self._run: Any = None

        def after_report(
            self, run_context: Any, metrics: list[dict[str, Any]], checkpoint: Any
        ) -> None:
            import wandb

            if self._run is None:
                self._run = wandb.init(project=project, name=run_name)
            for rank_metrics in metrics:
                self._run.log(rank_metrics)

    return WandbUserCallback()


def _build_run_config(training_cfg: TrainingConfig, ray_train_cfg: RayTrainConfig) -> Any:
    """Build a ``RunConfig`` with an optional W&B callback.

    The W&B ``UserCallback`` (from :func:`_make_wandb_callback`) is attached
    only when ``training_cfg.use_wandb`` is set, so the path runs offline by
    default.

    Args:
        training_cfg: Supplies the W&B project/run-name and the on/off flag.
        ray_train_cfg: Supplies the optional shared ``storage_path``.

    Returns:
        A ``ray.train.RunConfig``.
    """
    from ray.train import RunConfig

    callbacks = []
    if training_cfg.use_wandb:
        callbacks.append(
            _make_wandb_callback(training_cfg.wandb_project, training_cfg.wandb_run_name)
        )

    return RunConfig(
        name=training_cfg.wandb_run_name,
        storage_path=ray_train_cfg.storage_path,
        callbacks=callbacks,
    )


def run_ray_train(
    model_cfg: ModelConfig,
    teacher_cfg: TeacherConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    ray_train_cfg: RayTrainConfig,
    ray_data_cfg: RayDataConfig,
) -> Result:
    """Launch a distributed Ray Train distillation run.

    Builds ``train``/``val`` ``ray.data.Dataset`` objects via
    :func:`phenomica.ray_data.build_ray_dataset`, constructs a ``TorchTrainer``
    around :func:`train_fn` with a ``ScalingConfig`` derived from
    ``ray_train_cfg`` and a ``RunConfig`` carrying a W&B callback gated on
    ``training_cfg.use_wandb``, then calls ``trainer.fit()``. Run provenance is
    captured via :func:`phenomica.reproducibility.run_provenance` and attached
    as trainer ``metadata``.

    Args:
        model_cfg: Student model config.
        teacher_cfg: DINOv2 teacher config.
        data_cfg: Dataset config (root, image_size, batch_size).
        training_cfg: Optimizer/loss/W&B hyperparameters.
        ray_train_cfg: Ray Train scaling config.
        ray_data_cfg: Ray Data ingest config.

    Returns:
        The ``ray.train.Result`` from ``trainer.fit()``.
    """
    from ray.train.torch import TorchTrainer

    train_ds = build_ray_dataset(data_cfg, ray_data_cfg, split=_TRAIN_SHARD)
    val_ds = build_ray_dataset(data_cfg, ray_data_cfg, split=_VAL_SHARD)

    train_loop_config: dict[str, Any] = {
        _MODEL_KEY: asdict(model_cfg),
        _TEACHER_KEY: asdict(teacher_cfg),
        _TRAINING_KEY: asdict(training_cfg),
        _RAY_TRAIN_KEY: asdict(ray_train_cfg),
        _BATCH_SIZE_KEY: data_cfg.batch_size,
        # Ray Train v2 deprecates ``TorchTrainer(metadata=...)`` in favour of
        # passing run metadata through ``train_loop_config``.
        _PROVENANCE_KEY: run_provenance(),
    }

    trainer = TorchTrainer(
        train_fn,
        train_loop_config=train_loop_config,
        scaling_config=_build_scaling_config(ray_train_cfg),
        datasets={_TRAIN_SHARD: train_ds, _VAL_SHARD: val_ds},
        run_config=_build_run_config(training_cfg, ray_train_cfg),
    )
    logger.info(
        "Launching Ray Train: num_workers=%d use_gpu=%s max_epochs=%d",
        ray_train_cfg.num_workers,
        ray_train_cfg.use_gpu,
        ray_train_cfg.max_epochs,
    )
    return trainer.fit()
