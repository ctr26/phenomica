"""Phenomica: Vision model distillation from DINOv2.

Public API spans two INDEPENDENT launch paths:

* the submitit/Hydra path driven by :mod:`phenomica.train` (unchanged);
* the Ray path (Ray Data + Ray Train + Ray Tune) driven by the ``ray_*``
  modules and exposed here for programmatic use.

Importing the ``ray_*`` entry points is cheap: they reference Ray only under
``TYPE_CHECKING`` and import it lazily inside their (currently stubbed) bodies.
"""

from __future__ import annotations

from phenomica.configs import (
    RayDataConfig,
    RayTrainConfig,
    RayTuneConfig,
    RayTuneSearchSpace,
)
from phenomica.ray_data import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    build_ray_dataset,
    preprocess_batch,
)
from phenomica.ray_launch import ray_train_main, ray_tune_main
from phenomica.ray_train import run_ray_train
from phenomica.ray_tune import run_ray_tune

__version__ = "0.1.0"

__all__ = [
    "RayDataConfig",
    "RayTrainConfig",
    "RayTuneConfig",
    "RayTuneSearchSpace",
    "IMAGE_COLUMN",
    "LABEL_COLUMN",
    "build_ray_dataset",
    "preprocess_batch",
    "ray_train_main",
    "ray_tune_main",
    "run_ray_train",
    "run_ray_tune",
    "__version__",
]
