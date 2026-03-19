"""Phenomica training entry point.

Usage::

    uv run phenomica-train model=simple_resnet18 teacher=dinov2_base
    uv run phenomica-train model=multi_resnet18 teacher=dinov2_large cluster=biohive
"""

from __future__ import annotations

import logging
from typing import Any

from hydra_zen import make_config, store, zen

import phenomica.configs  # noqa: F401 -- triggers store registration
from phenomica.data import create_dataloaders
from phenomica.distributed import cleanup_distributed, set_seed
from phenomica.trainer import DistillationTrainer

logger = logging.getLogger(__name__)


def _run_training(
    model: Any,
    teacher: Any,
    data: Any,
    training: Any,
) -> None:
    """Core training logic: seed, build trainer, create loaders, train.

    Args:
        model: Hydra model config.
        teacher: Hydra teacher config.
        data: Hydra data config.
        training: Hydra training config.
    """
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


def task(
    model: Any,
    teacher: Any,
    data: Any,
    training: Any,
    cluster: Any,
) -> None:
    """Main task function dispatching to local or SLURM execution.

    When ``cluster.use_submitit`` is True, the training job is submitted via
    ``submitit.AutoExecutor`` with parameters from the cluster config.
    Otherwise, training runs in the current process.

    Args:
        model: Hydra model config.
        teacher: Hydra teacher config.
        data: Hydra data config.
        training: Hydra training config.
        cluster: Hydra cluster config.
    """
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
            slurm_account=getattr(cluster, "slurm_account", None),
        )
        job = executor.submit(_run_training, model, teacher, data, training)
        logger.info("Submitted SLURM job: %s", job.job_id)
    else:
        _run_training(model, teacher, data, training)


TopConfig = make_config(
    model=None,
    teacher=None,
    data=None,
    training=None,
    cluster=None,
    hydra_defaults=[
        "_self_",
        {"model": "simple_resnet18"},
        {"teacher": "dinov2_base"},
        {"data": "imagenet"},
        {"training": "default"},
        {"cluster": "local"},
    ],
)
store(TopConfig, name="phenomica")


def main() -> None:
    """Hydra entry point for ``phenomica-train``."""
    store.add_to_hydra_store()
    zen(task).hydra_main(config_name="phenomica", version_base="1.2", config_path=None)


if __name__ == "__main__":
    main()
