"""Ray launch entry points (Hydra/hydra-zen console scripts).

Hydra console-script entry points for the two Ray paths, kept INDEPENDENT
from the submitit path in :mod:`phenomica.train`:

* ``ray_train_main`` (console script ``phenomica-ray-train``) -> a single
  distributed :func:`phenomica.ray_train.run_ray_train`;
* ``ray_tune_main`` (console script ``phenomica-ray-tune``) -> an ASHA
  hyperparameter sweep via :func:`phenomica.ray_tune.run_ray_tune`.

Both compose the same hydra-zen config groups as ``train.py`` (``model``,
``teacher``, ``data``, ``training``) plus the Ray-specific groups
(``ray_data``, ``ray_train``, and ``ray_tune`` for the sweep). Each group is
auto-extracted and instantiated by :func:`hydra_zen.zen` from the wrapped
function's parameter names, with ``pydantic_parser`` applied as the
``instantiation_wrapper`` so every group is validated exactly as in
``train.py``. The zen-wrapped task functions are Hydra-agnostic and unit-
testable with plain config dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import zen
from hydra_zen.third_party.pydantic import pydantic_parser
from omegaconf import DictConfig

# Imported at runtime (not under TYPE_CHECKING): the task-function annotations
# below are forward refs that ``pydantic_parser`` resolves against this module's
# globals when validating each instantiated group. ``configs`` is pure
# dataclasses with no heavy deps, and importing it also registers the hydra-zen
# store groups.
from phenomica.configs import (
    DataConfig,
    ModelConfig,
    RayDataConfig,
    RayTrainConfig,
    RayTuneConfig,
    TeacherConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

# Hydra-zen group defaults shared by both Ray entry points. The submitit-only
# ``cluster`` group is intentionally absent: Ray scaling lives in ``ray_train``.
_COMMON_DEFAULTS: list = [
    "_self_",
    {"model": "simple_resnet18"},
    {"teacher": "dinov2_base"},
    {"data": "imagenette"},
    {"training": "debug"},
    {"ray_data": "default"},
    {"ray_train": "local_cpu"},
]


@dataclass
class RayTrainLaunchConfig:
    """Top-level config for the ``phenomica-ray-train`` console script."""

    defaults: list = field(default_factory=lambda: list(_COMMON_DEFAULTS))
    model: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    ray_data: dict = field(default_factory=dict)
    ray_train: dict = field(default_factory=dict)


@dataclass
class RayTuneLaunchConfig:
    """Top-level config for the ``phenomica-ray-tune`` console script."""

    defaults: list = field(default_factory=lambda: [*_COMMON_DEFAULTS, {"ray_tune": "default"}])
    model: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    ray_data: dict = field(default_factory=dict)
    ray_train: dict = field(default_factory=dict)
    ray_tune: dict = field(default_factory=dict)


_cs = ConfigStore.instance()
_cs.store(name="ray_train_launch", node=RayTrainLaunchConfig)
_cs.store(name="ray_tune_launch", node=RayTuneLaunchConfig)


def _ray_train_task(
    model: ModelConfig,
    teacher: TeacherConfig,
    data: DataConfig,
    training: TrainingConfig,
    ray_data: RayDataConfig,
    ray_train: RayTrainConfig,
) -> None:
    """Hydra-agnostic Ray Train task: dispatch to :func:`run_ray_train`.

    ``zen`` populates each parameter by instantiating the matching config group
    (validated through ``pydantic_parser``); this body is plain Python and can
    be called directly with already-instantiated configs in tests.

    Args:
        model: Instantiated student model config.
        teacher: Instantiated DINOv2 teacher config.
        data: Instantiated dataset config.
        training: Instantiated optimizer/loss/W&B config.
        ray_data: Instantiated Ray Data ingest config.
        ray_train: Instantiated Ray Train scaling config.
    """
    from phenomica.ray_train import run_ray_train

    logger.info("Starting Ray Train run (model=%s teacher=%s)", model.backbone, teacher.model_name)
    run_ray_train(model, teacher, data, training, ray_train, ray_data)


def _ray_tune_task(
    model: ModelConfig,
    teacher: TeacherConfig,
    data: DataConfig,
    training: TrainingConfig,
    ray_data: RayDataConfig,
    ray_train: RayTrainConfig,
    ray_tune: RayTuneConfig,
) -> None:
    """Hydra-agnostic Ray Tune task: dispatch to :func:`run_ray_tune`.

    Args:
        model: Instantiated student model config.
        teacher: Instantiated DINOv2 teacher config.
        data: Instantiated dataset config.
        training: Instantiated base optimizer/loss/W&B config.
        ray_data: Instantiated Ray Data ingest config.
        ray_train: Instantiated per-trial Ray Train scaling config.
        ray_tune: Instantiated Ray Tune sweep config.
    """
    from phenomica.ray_tune import run_ray_tune

    logger.info("Starting Ray Tune sweep (num_samples=%d)", ray_tune.num_samples)
    run_ray_tune(model, teacher, data, training, ray_train, ray_data, ray_tune)


# ``zen`` extracts each task parameter from the composed config and instantiates
# it; ``pydantic_parser`` validates every group exactly as ``train.py`` does.
_ray_train_zen = zen(_ray_train_task, instantiation_wrapper=pydantic_parser)
_ray_tune_zen = zen(_ray_tune_task, instantiation_wrapper=pydantic_parser)


@hydra.main(config_name="ray_train_launch", version_base="1.3", config_path=None)
def ray_train_main(cfg: DictConfig) -> None:
    """Console entry point for a single Ray Train distillation run."""
    _ray_train_zen(cfg)


@hydra.main(config_name="ray_tune_launch", version_base="1.3", config_path=None)
def ray_tune_main(cfg: DictConfig) -> None:
    """Console entry point for a Ray Tune hyperparameter sweep."""
    _ray_tune_zen(cfg)


if __name__ == "__main__":
    ray_train_main()
