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
    gradient_clip: Optional[float] = 1.0
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


@dataclass
class SlurmLauncherConfig:
    """Hydra submitit_slurm launcher config for multirun sweeps."""

    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"
    timeout_min: PositiveInt = 720
    cpus_per_task: PositiveInt = 8
    gpus_per_node: PositiveInt = 4
    tasks_per_node: PositiveInt = 4
    mem_gb: PositiveInt = 64
    nodes: PositiveInt = 1
    partition: str = "hopper"
    account: Optional[str] = None
    name: str = "${hydra.job.name}"


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

    # Launcher presets for --multirun sweeps
    _preset(SlurmLauncherConfig, "hydra/launcher", "submitit_slurm")
    _preset(
        SlurmLauncherConfig,
        "hydra/launcher",
        "submitit_slurm_biohive",
        partition="biohive",
        account="rxrx",
    )


register_configs()
store.add_to_hydra_store()
