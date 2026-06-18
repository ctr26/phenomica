"""Phenomica training entry point.

Usage::

    uv run phenomica-train model=simple_resnet18 teacher=dinov2_base
    uv run phenomica-train model=multi_resnet18 teacher=dinov2_large cluster=biohive
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from hydra_zen.third_party.pydantic import pydantic_parser
from omegaconf import DictConfig

import phenomica.configs  # noqa: F401 -- registers groups
from phenomica.configs import (
    DataConfig,
    ModelConfig,
    TeacherConfig,
    TrainingConfig,
)
from phenomica.data import create_dataloaders
from phenomica.distributed import cleanup_distributed, set_seed
from phenomica.trainer import DistillationTrainer

logger = logging.getLogger(__name__)


def _run_training(
    model: ModelConfig,
    teacher: TeacherConfig,
    data: DataConfig,
    training: TrainingConfig,
) -> None:
    """Core training logic: seed, build trainer, create loaders, train."""
    set_seed(training.seed)

    trainer = DistillationTrainer(
        training_cfg=training,
        model_cfg=model,
        teacher_cfg=teacher,
        data_cfg=data,
    )

    train_loader, val_loader = create_dataloaders(
        data, is_distributed=trainer.is_distributed
    )

    try:
        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


@dataclass
class PhenomicaConfig:
    """Top-level config."""

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
    model: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    cluster: dict = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="phenomica", node=PhenomicaConfig)


@hydra.main(config_name="phenomica", version_base="1.3", config_path=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point with pydantic validation."""
    # Instantiate configs with pydantic validation
    model = instantiate(cfg.model, _target_wrapper_=pydantic_parser)
    teacher = instantiate(cfg.teacher, _target_wrapper_=pydantic_parser)
    data = instantiate(cfg.data, _target_wrapper_=pydantic_parser)
    training = instantiate(cfg.training, _target_wrapper_=pydantic_parser)
    cluster = instantiate(cfg.cluster, _target_wrapper_=pydantic_parser)

    if cluster.use_submitit:
        import submitit

        executor = submitit.AutoExecutor(folder=cluster.log_dir)
        executor.update_parameters(
            slurm_partition=cluster.partition,
            gpus_per_node=cluster.gpus_per_node,
            slurm_ntasks_per_node=cluster.gpus_per_node,
            nodes=cluster.nodes,
            timeout_min=cluster.timeout_min,
            mem_gb=cluster.mem_gb,
            cpus_per_task=cluster.cpus_per_task,
            slurm_account=cluster.slurm_account,
            slurm_wckey="default",
        )
        job = executor.submit(_run_training, model, teacher, data, training)
        logger.info("Submitted SLURM job: %s", job.job_id)
    else:
        _run_training(model, teacher, data, training)


if __name__ == "__main__":
    main()
