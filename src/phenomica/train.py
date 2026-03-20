"""Phenomica training entry point.

Usage::

    uv run phenomica-train model=simple_resnet18 teacher=dinov2_base
    uv run phenomica-train model=multi_resnet18 teacher=dinov2_large cluster=biohive
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

import phenomica.configs  # noqa: F401 -- triggers ConfigStore registration
from phenomica.data import create_dataloaders
from phenomica.distributed import cleanup_distributed, set_seed
from phenomica.trainer import DistillationTrainer

logger = logging.getLogger(__name__)


def _run_training(cfg: DictConfig) -> None:
    """Core training logic: seed, build trainer, create loaders, train."""
    set_seed(cfg.training.seed)

    trainer = DistillationTrainer(
        training_cfg=cfg.training,
        model_cfg=cfg.model,
        teacher_cfg=cfg.teacher,
        data_cfg=cfg.data,
    )

    train_loader, val_loader = create_dataloaders(
        cfg.data, is_distributed=trainer.is_distributed
    )

    try:
        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


@hydra.main(config_name="phenomica", version_base="1.3", config_path=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point dispatching to local or SLURM execution."""
    if cfg.cluster.use_submitit:
        import submitit

        executor = submitit.AutoExecutor(folder=cfg.cluster.log_dir)
        executor.update_parameters(
            slurm_partition=cfg.cluster.partition,
            gpus_per_node=cfg.cluster.gpus_per_node,
            slurm_ntasks_per_node=cfg.cluster.gpus_per_node,
            nodes=cfg.cluster.nodes,
            timeout_min=cfg.cluster.timeout_min,
            mem_gb=cfg.cluster.mem_gb,
            cpus_per_task=cfg.cluster.cpus_per_task,
            slurm_account=cfg.cluster.get("slurm_account"),
        )
        job = executor.submit(_run_training, cfg)
        logger.info("Submitted SLURM job: %s", job.job_id)
    else:
        _run_training(cfg)


if __name__ == "__main__":
    main()
