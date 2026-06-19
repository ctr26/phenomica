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
    # Enables CLS-attention extraction, required for loss_type="attndistill".
    extract_attention: bool = False


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
    loss_type: Literal["mse", "cosine", "combined", "cospress", "vitkd", "attndistill", "rekd"] = (
        "mse"
    )
    cosine_weight: NonNegativeFloat = 1.0
    mse_weight: NonNegativeFloat = 1.0
    cospress_weight: NonNegativeFloat = 1.0
    cospress_temperature: PositiveFloat = 0.1
    cospress_cosine_weight: NonNegativeFloat = 1.0
    vitkd_weight: NonNegativeFloat = 1.0
    vitkd_student_dim: PositiveInt = 768
    vitkd_teacher_dim: PositiveInt = 768
    vitkd_mask_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    vitkd_gen_weight: NonNegativeFloat = 1.0
    vitkd_num_tokens: PositiveInt = 256
    attndistill_weight: NonNegativeFloat = 1.0
    attndistill_attn_weight: NonNegativeFloat = 1.0
    attndistill_student_dim: PositiveInt = 768
    attndistill_num_heads: PositiveInt = 12
    attndistill_num_tokens: PositiveInt = 256
    rekd_weight: NonNegativeFloat = 1.0
    rekd_temperature: PositiveFloat = 0.1
    rekd_topk: PositiveInt = 5
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
    use_submitit: bool = False
    partition: str = "hopper"
    gpus_per_node: PositiveInt = 4
    nodes: PositiveInt = 1
    timeout_min: PositiveInt = 720
    mem_gb: PositiveInt = 64
    cpus_per_task: PositiveInt = 8
    slurm_account: Optional[str] = None
    log_dir: str = "slurm_logs"


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
        "dinov2_base_attn",
        extract_attention=True,
    )
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
    _preset(TrainingConfig, "training", "cospress", loss_type="cospress")
    _preset(TrainingConfig, "training", "vitkd", loss_type="vitkd")
    _preset(TrainingConfig, "training", "rekd", loss_type="rekd")
    _preset(TrainingConfig, "training", "attndistill", loss_type="attndistill")

    _preset(ClusterConfig, "cluster", "local")
    _preset(ClusterConfig, "cluster", "biohive", use_submitit=True)


register_configs()
store.add_to_hydra_store()
