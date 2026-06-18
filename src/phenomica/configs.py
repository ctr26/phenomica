"""Hydra structured config schemas and hydra-zen store registration.

Defines typed dataclass schemas for all config groups, then registers
preset instances via hydra-zen store using builds().

Pydantic types (Literal, PositiveInt, PositiveFloat) are used in type
annotations for validation via pydantic_parser at instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from hydra_zen import builds, store
from omegaconf import MISSING
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt


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
    val_split: float = 0.1


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
    cosine_weight: float = 1.0
    mse_weight: float = 1.0
    use_wandb: bool = True
    wandb_project: str = "phenomica"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    seed: int = 42
    use_ddp: bool = True
    early_stopping_patience: Optional[int] = 20
    validation_freq: int = 1
    eval_freq: Optional[int] = 10


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


def register_configs() -> None:
    """Register all config schemas and presets into hydra-zen store."""
    # -- Model presets -------------------------------------------------------
    store(
        builds(ModelConfig, populate_full_signature=True),
        group="model",
        name="simple_resnet18",
    )
    store(
        builds(ModelConfig, backbone="efficientnet_b0", populate_full_signature=True),
        group="model",
        name="simple_effnet",
    )
    store(
        builds(
            ModelConfig, backbone="vit_tiny_patch16_224", populate_full_signature=True
        ),
        group="model",
        name="simple_vit_tiny",
    )
    store(
        builds(ModelConfig, variant="multifunction", populate_full_signature=True),
        group="model",
        name="multi_resnet18",
    )
    store(
        builds(
            ModelConfig,
            variant="multifunction",
            backbone="efficientnet_b0",
            populate_full_signature=True,
        ),
        group="model",
        name="multi_effnet",
    )

    # -- Teacher presets -----------------------------------------------------
    store(
        builds(TeacherConfig, populate_full_signature=True),
        group="teacher",
        name="dinov2_base",
    )
    store(
        builds(
            TeacherConfig,
            model_name="dinov2_vits14",
            embed_dim=384,
            populate_full_signature=True,
        ),
        group="teacher",
        name="dinov2_small",
    )
    store(
        builds(
            TeacherConfig,
            model_name="dinov2_vitl14",
            embed_dim=1024,
            populate_full_signature=True,
        ),
        group="teacher",
        name="dinov2_large",
    )

    # -- Data presets --------------------------------------------------------
    store(
        builds(DataConfig, root="data/imagenet", populate_full_signature=True),
        group="data",
        name="imagenet",
    )
    store(
        builds(
            DataConfig,
            root="data/custom",
            batch_size=128,
            num_workers=4,
            populate_full_signature=True,
        ),
        group="data",
        name="custom",
    )

    # -- Training presets ----------------------------------------------------
    store(
        builds(TrainingConfig, populate_full_signature=True),
        group="training",
        name="default",
    )
    store(
        builds(
            TrainingConfig,
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
            populate_full_signature=True,
        ),
        group="training",
        name="debug",
    )

    # -- Cluster presets -----------------------------------------------------
    store(
        builds(ClusterConfig, populate_full_signature=True),
        group="cluster",
        name="local",
    )
    store(
        builds(ClusterConfig, use_submitit=True, populate_full_signature=True),
        group="cluster",
        name="biohive",
    )


register_configs()
store.add_to_hydra_store()
