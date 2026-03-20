"""Hydra structured config schemas and ConfigStore registration.

Defines typed dataclass schemas for all config groups, then registers
preset instances into Hydra's ConfigStore. No YAML files needed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# -- Enums -------------------------------------------------------------------


class ModelVariant(str, Enum):
    SIMPLE = "simple"
    MULTIFUNCTION = "multifunction"


class LossType(str, Enum):
    MSE = "mse"
    COSINE = "cosine"
    COMBINED = "combined"


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class LRScheduler(str, Enum):
    COSINE = "cosine"
    STEP = "step"


# -- Schemas -----------------------------------------------------------------


@dataclass
class ModelConfig:
    """Unified model config covering both variants.

    For variant="simple", only backbone/projection_dim/pretrained_backbone
    are used. For variant="multifunction", the teacher_* fields are also used.
    """

    variant: str = ModelVariant.SIMPLE
    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    # Simple variant fields
    projection_dim: int = 768
    # Multi-function variant fields
    teacher_cls_dim: int = 768
    teacher_patch_dim: int = 768
    teacher_layers: list[int] = field(default_factory=lambda: [3, 6, 9, 11])


@dataclass
class TeacherConfig:
    model_name: str = "dinov2_vitb14"
    embed_dim: int = 768


@dataclass
class DataConfig:
    dataset_type: str = "imagefolder"
    root: str = MISSING
    image_size: int = 224
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    val_split: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = Optimizer.ADAMW
    lr_scheduler: Optional[str] = LRScheduler.COSINE
    warmup_epochs: int = 5
    warmup_start_factor: float = 1e-4
    gradient_clip: Optional[float] = 1.0
    loss_type: str = LossType.MSE
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
    partition: str = "h100"
    gpus_per_node: int = 4
    nodes: int = 1
    timeout_min: int = 720
    mem_gb: int = 64
    cpus_per_task: int = 8
    slurm_account: Optional[str] = None
    log_dir: str = "slurm_logs"


@dataclass
class PhenomicaConfig:
    defaults: list = field(
        default_factory=lambda: [
            "_self_",
            {"model": "simple_resnet18"},
            {"teacher": "dinov2_base"},
            {"data": "imagenet"},
            {"training": "default"},
            {"cluster": "local"},
        ]
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


# -- ConfigStore registration ------------------------------------------------


def register_configs() -> None:
    """Register all config schemas and presets into Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    # Top-level schema
    cs.store(name="phenomica", node=PhenomicaConfig)

    # -- Model presets -------------------------------------------------------
    cs.store(group="model", name="simple_resnet18", node=ModelConfig())
    cs.store(
        group="model", name="simple_effnet",
        node=ModelConfig(backbone="efficientnet_b0"),
    )
    cs.store(
        group="model", name="simple_vit_tiny",
        node=ModelConfig(backbone="vit_tiny_patch16_224"),
    )
    cs.store(
        group="model", name="multi_resnet18",
        node=ModelConfig(variant=ModelVariant.MULTIFUNCTION),
    )
    cs.store(
        group="model", name="multi_effnet",
        node=ModelConfig(
            variant=ModelVariant.MULTIFUNCTION, backbone="efficientnet_b0",
        ),
    )

    # -- Teacher presets -----------------------------------------------------
    cs.store(group="teacher", name="dinov2_base", node=TeacherConfig())
    cs.store(
        group="teacher", name="dinov2_small",
        node=TeacherConfig(model_name="dinov2_vits14", embed_dim=384),
    )
    cs.store(
        group="teacher", name="dinov2_large",
        node=TeacherConfig(model_name="dinov2_vitl14", embed_dim=1024),
    )

    # -- Data presets --------------------------------------------------------
    cs.store(
        group="data", name="imagenet",
        node=DataConfig(root="data/imagenet"),
    )
    cs.store(
        group="data", name="custom",
        node=DataConfig(root="data/custom", batch_size=128, num_workers=4),
    )

    # -- Training presets ----------------------------------------------------
    cs.store(group="training", name="default", node=TrainingConfig())
    cs.store(
        group="training", name="debug",
        node=TrainingConfig(
            epochs=10, weight_decay=0.0, optimizer=Optimizer.ADAM,
            lr_scheduler=None, warmup_epochs=0, gradient_clip=None,
            use_wandb=False, use_ddp=False,
            early_stopping_patience=None, eval_freq=5,
        ),
    )

    # -- Cluster presets -----------------------------------------------------
    cs.store(group="cluster", name="local", node=ClusterConfig())
    cs.store(
        group="cluster", name="biohive",
        node=ClusterConfig(use_submitit=True),
    )


register_configs()
