"""Ray Train (TorchTrainer) distillation path (SCAFFOLD STUB).

Distributed distillation via ``ray.train.torch.TorchTrainer``. The per-worker
loop reuses the existing distillation building blocks unchanged:
:func:`phenomica.models.build_model`, :class:`phenomica.teacher.DINOv2Teacher`,
and the loss classes in :mod:`phenomica.losses`. Data arrives as a Ray Data
shard built by :func:`phenomica.ray_data.build_ray_dataset`.

This is an INDEPENDENT launch path from submitit; ``phenomica.train`` (the
submitit/Hydra path) is untouched.

All bodies are intentionally unimplemented (scaffold). Build workers fill
them against the fixed signatures below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

logger = logging.getLogger(__name__)


def train_fn(config: dict[str, Any]) -> None:
    """Per-worker Ray Train loop (runs on every distributed worker).

    Expects ``config`` (the TorchTrainer ``train_loop_config``) to carry the
    serialized phenomica configs plus tunable hyperparameters under these
    keys: ``model``, ``teacher``, ``training``, ``ray_train``, and the
    Tune-injected overrides ``lr``/``weight_decay``/``loss_type`` when present.

    Responsibilities (to implement):

    * build the student via ``build_model`` and wrap it with
      ``ray.train.torch.prepare_model`` (DDP);
    * instantiate a frozen ``DINOv2Teacher`` on the worker device;
    * select ``DistillationLoss`` or ``MultiFunctionDistillationLoss`` by
      ``model.variant`` (sync student head dims to the teacher embed_dim,
      mirroring ``DistillationTrainer``);
    * pull the train (and optional val) shard via
      ``ray.train.get_dataset_shard`` and iterate with ``iter_torch_batches``;
    * after each epoch call ``ray.train.report`` with ``train_loss`` and
      ``val_loss`` (the metric ASHA/Tune consumes).

    Args:
        config: The ``train_loop_config`` dict passed by ``TorchTrainer``.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_train.train_fn is a scaffold stub")


def run_ray_train(
    model_cfg: ModelConfig,
    teacher_cfg: TeacherConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    ray_train_cfg: RayTrainConfig,
    ray_data_cfg: RayDataConfig,
) -> Result:
    """Launch a distributed Ray Train distillation run.

    Builds train/val ``ray.data.Dataset`` objects via
    :func:`phenomica.ray_data.build_ray_dataset`, constructs a
    ``TorchTrainer`` with :func:`train_fn`, a ``ScalingConfig`` derived from
    ``ray_train_cfg`` (``num_workers``/``use_gpu``/``resources_per_worker``),
    and a ``RunConfig`` carrying a ``WandbLoggerCallback``
    (``ray.air.integrations.wandb``) gated on ``training_cfg.use_wandb``, then
    calls ``trainer.fit()``.

    Args:
        model_cfg: Student model config.
        teacher_cfg: DINOv2 teacher config.
        data_cfg: Dataset config (root, image_size).
        training_cfg: Optimizer/loss/wandb hyperparameters.
        ray_train_cfg: Ray Train scaling config.
        ray_data_cfg: Ray Data ingest config.

    Returns:
        The ``ray.train.Result`` from ``trainer.fit()``.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_train.run_ray_train is a scaffold stub")
