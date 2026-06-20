"""Hydra structured config schemas and hydra-zen store registration.

Defines typed dataclass schemas for all config groups, then registers
preset instances via hydra-zen store using builds().

Pydantic types (Literal, PositiveInt, PositiveFloat) are used in type
annotations for validation via pydantic_parser at instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

from hydra_zen import builds, store
from omegaconf import MISSING
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


@dataclass
class ModelConfig:
    """Unified model config covering both variants.

    For variant="simple", only backbone/projection_dim/pretrained_backbone
    are used. For variant="multifunction", the teacher_* fields are also used.
    """

    variant: Literal["simple", "multifunction"] = "simple"
    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    # Simple variant fields
    projection_dim: PositiveInt = 768
    # Multi-function variant fields
    teacher_cls_dim: PositiveInt = 768
    teacher_patch_dim: PositiveInt = 768
    teacher_layers: list[int] = field(default_factory=lambda: [3, 6, 9, 11])


@dataclass
class TeacherConfig:
    model_name: str = "dinov2_vitb14"
    embed_dim: PositiveInt = 768


@dataclass
class DataConfig:
    dataset_type: str = "imagefolder"
    root: str = MISSING
    image_size: PositiveInt = 224
    batch_size: PositiveInt = 256
    num_workers: NonNegativeInt = 8
    pin_memory: bool = True
    val_split: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1


@dataclass
class TrainingConfig:
    epochs: PositiveInt = 100
    learning_rate: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-4
    optimizer: Literal["adam", "adamw"] = "adamw"
    lr_scheduler: Optional[Literal["cosine", "step"]] = "cosine"
    warmup_epochs: NonNegativeInt = 5
    warmup_start_factor: PositiveFloat = 1e-4
    gradient_clip: Optional[PositiveFloat] = 1.0
    loss_type: Literal["mse", "cosine", "combined"] = "mse"
    cosine_weight: NonNegativeFloat = 1.0
    mse_weight: NonNegativeFloat = 1.0
    use_wandb: bool = True
    wandb_project: str = "phenomica"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    seed: int = 42
    use_ddp: bool = True
    early_stopping_patience: Optional[int] = 20
    validation_freq: PositiveInt = 1
    eval_freq: Optional[PositiveInt] = 10


@dataclass
class ClusterConfig:
    """Cluster submission config.

    For --multirun sweeps, use hydra/launcher=submitit_slurm and pass these
    params as hydra.launcher.* overrides. For single-job non-multirun, the
    use_submitit flag triggers the manual submitit path in train.py.
    """

    use_submitit: bool = False
    partition: str = "hopper"
    gpus_per_node: PositiveInt = 4
    nodes: PositiveInt = 1
    timeout_min: PositiveInt = 720
    mem_gb: PositiveInt = 64
    cpus_per_task: PositiveInt = 8
    slurm_account: Optional[str] = None
    log_dir: str = "slurm_logs"


# -- Ray launch-path config schemas ------------------------------------------
# Ray is an INDEPENDENT launch path from submitit. These schemas drive the
# ray_data / ray_train / ray_tune modules; they are never consumed by train.py.


@dataclass
class RayDataConfig:
    """Ray Data ingest/preprocessing config.

    Drives :func:`phenomica.ray_data.build_ray_dataset`, which reads images
    from ``DataConfig.root`` into a ``ray.data.Dataset`` and applies the
    distillation preprocessing map (resize/normalize to ImageNet stats).

    Attributes:
        parallelism: Target number of read blocks (-1 lets Ray auto-detect
            based on cluster resources).
        shuffle_buffer_size: Per-worker local shuffle buffer size in rows.
            ``None`` disables streaming shuffle.
        prefetch_batches: Number of batches each worker prefetches while
            iterating torch batches.
        override_num_blocks: Optional explicit read-block count, overriding
            ``parallelism`` when set.
    """

    parallelism: int = -1
    shuffle_buffer_size: Optional[PositiveInt] = None
    prefetch_batches: PositiveInt = 2
    override_num_blocks: Optional[PositiveInt] = None


@dataclass
class RayTrainConfig:
    """Ray Train (TorchTrainer + ScalingConfig) config.

    Drives :func:`phenomica.ray_train.run_ray_train`. Mirrors the scaling
    knobs of ``ray.train.ScalingConfig`` plus a per-run epoch cap; the actual
    optimizer/loss hyperparameters are still sourced from ``TrainingConfig``.

    Attributes:
        num_workers: Number of distributed Ray Train workers (one DDP rank
            each). Use 1 for local CPU smoke tests.
        use_gpu: Request a GPU per worker. Keep ``False`` for CPU/local/tests.
        cpus_per_worker: CPUs reserved per worker (``resources_per_worker``
            CPU entry).
        gpus_per_worker: GPUs reserved per worker (used only when
            ``use_gpu`` is True).
        max_epochs: Epoch cap for the Ray Train run (independent of
            ``TrainingConfig.epochs`` so the Ray path can be smoke-sized).
        storage_path: Optional shared/NFS/S3 path for run artifacts; required
            for multi-node runs, ``None`` for local.
    """

    num_workers: PositiveInt = 1
    use_gpu: bool = False
    cpus_per_worker: PositiveInt = 1
    gpus_per_worker: PositiveInt = 1
    max_epochs: PositiveInt = 10
    storage_path: Optional[str] = None


@dataclass
class RayTuneSearchSpace:
    """Search-space bounds for :class:`RayTuneConfig`.

    Each field is a plain bound that ``ray_tune`` converts into a Ray Tune
    domain (``tune.loguniform`` for ``lr``/``weight_decay``, ``tune.choice``
    for ``loss_type``). Keeping bounds as primitives keeps the dataclass
    hydra/pydantic-serializable (Ray ``Domain`` objects are not).

    Attributes:
        lr_min: Lower bound for the log-uniform learning-rate search.
        lr_max: Upper bound for the log-uniform learning-rate search.
        weight_decay_min: Lower bound for the log-uniform weight-decay search.
        weight_decay_max: Upper bound for the log-uniform weight-decay search.
        loss_types: Categorical choices for ``TrainingConfig.loss_type``.
    """

    lr_min: PositiveFloat = 1e-5
    lr_max: PositiveFloat = 1e-2
    weight_decay_min: PositiveFloat = 1e-6
    weight_decay_max: PositiveFloat = 1e-2
    loss_types: list[str] = field(
        default_factory=lambda: ["mse", "cosine", "combined"]
    )


@dataclass
class RayTuneConfig:
    """Ray Tune (Tuner + ASHAScheduler) config.

    Drives :func:`phenomica.ray_tune.run_ray_tune`, which sweeps over
    :class:`RayTuneSearchSpace` using an ASHA early-stopping scheduler and
    logs each trial to W&B via ``WandbLoggerCallback``.

    Attributes:
        num_samples: Number of hyperparameter samples (trials) to launch.
        metric: Reported metric ASHA optimizes (must match a key passed to
            ``ray.train.report``, e.g. ``"val_loss"``).
        mode: ``"min"`` or ``"max"`` optimization direction for ``metric``.
        grace_period: Minimum training iterations before ASHA may stop a
            trial (``ASHAScheduler.grace_period``).
        max_t: Maximum training iterations per trial (``ASHAScheduler.max_t``).
        reduction_factor: ASHA halving factor between rungs.
        max_concurrent_trials: Cap on simultaneously running trials (bounds
            spawned Train driver processes).
        search_space: Hyperparameter bounds for the sweep.
    """

    num_samples: PositiveInt = 4
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    grace_period: PositiveInt = 1
    max_t: PositiveInt = 10
    reduction_factor: PositiveInt = 2
    max_concurrent_trials: PositiveInt = 2
    search_space: RayTuneSearchSpace = field(default_factory=RayTuneSearchSpace)


# -- hydra-zen store registration --------------------------------------------


def _preset(config_cls: type, group: str, name: str, **overrides: object) -> None:
    """Register one preset config into the hydra-zen store group.

    Builds a structured config from ``config_cls`` (with the full signature
    populated so every field is overridable) and stores it under
    ``group``/``name``.
    """
    store(
        builds(config_cls, populate_full_signature=True, **overrides),
        group=group,
        name=name,
    )


def register_configs() -> None:
    """Register all config presets into the hydra-zen store."""
    _preset(ModelConfig, "model", "simple_resnet18")
    _preset(ModelConfig, "model", "simple_effnet", backbone="efficientnet_b0")
    _preset(ModelConfig, "model", "simple_vit_tiny", backbone="vit_tiny_patch16_224")
    _preset(ModelConfig, "model", "multi_resnet18", variant="multifunction")
    _preset(
        ModelConfig,
        "model",
        "multi_effnet",
        variant="multifunction",
        backbone="efficientnet_b0",
    )

    _preset(TeacherConfig, "teacher", "dinov2_base")
    _preset(
        TeacherConfig,
        "teacher",
        "dinov2_small",
        model_name="dinov2_vits14",
        embed_dim=384,
    )
    _preset(
        TeacherConfig,
        "teacher",
        "dinov2_large",
        model_name="dinov2_vitl14",
        embed_dim=1024,
    )

    _preset(DataConfig, "data", "imagenet", root="data/imagenet")
    _preset(
        DataConfig,
        "data",
        "imagenette",
        root="data/imagenette",
        batch_size=64,
        num_workers=4,
    )
    _preset(
        DataConfig,
        "data",
        "custom",
        root="data/custom",
        batch_size=128,
        num_workers=4,
    )

    _preset(TrainingConfig, "training", "default")
    _preset(
        TrainingConfig,
        "training",
        "debug",
        epochs=10,
        weight_decay=0.0,
        optimizer="adam",
        lr_scheduler=None,
        warmup_epochs=0,
        gradient_clip=None,
        use_wandb=False,
        use_ddp=False,
        early_stopping_patience=None,
        eval_freq=5,
    )

    _preset(ClusterConfig, "cluster", "local")
    _preset(ClusterConfig, "cluster", "biohive", use_submitit=True)

    # -- Ray launch-path presets ---------------------------------------------
    _preset(RayDataConfig, "ray_data", "default")

    _preset(RayTrainConfig, "ray_train", "local_cpu")
    _preset(
        RayTrainConfig,
        "ray_train",
        "biohive_gpu",
        num_workers=4,
        use_gpu=True,
        cpus_per_worker=8,
        max_epochs=100,
    )

    _preset(RayTuneConfig, "ray_tune", "default")
    _preset(
        RayTuneConfig,
        "ray_tune",
        "debug",
        num_samples=2,
        grace_period=1,
        max_t=2,
        max_concurrent_trials=1,
    )


register_configs()
store.add_to_hydra_store()
