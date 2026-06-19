"""Ray launch entry points (SCAFFOLD STUB).

Hydra/hydra-zen console-script entry points for the two Ray paths, kept
INDEPENDENT from the submitit path in :mod:`phenomica.train`:

* ``ray_train_main`` (console script ``phenomica-ray-train``) -> a single
  distributed :func:`phenomica.ray_train.run_ray_train`;
* ``ray_tune_main`` (console script ``phenomica-ray-tune``) -> an ASHA
  hyperparameter sweep via :func:`phenomica.ray_tune.run_ray_tune`.

Both compose the same hydra-zen config groups as ``train.py`` (``model``,
``teacher``, ``data``, ``training``) plus the Ray-specific groups
(``ray_data``, ``ray_train``, ``ray_tune``), instantiate them through the
pydantic parser, and dispatch to the corresponding run function.

All bodies are intentionally unimplemented (scaffold). Build workers wire the
``@hydra.main`` decorator + ``ConfigStore`` top-level config and instantiation
against the fixed signatures below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def ray_train_main(cfg: DictConfig) -> None:
    """Console entry point for a single Ray Train distillation run.

    Instantiates the ``model``/``teacher``/``data``/``training``/``ray_data``/
    ``ray_train`` config groups (via ``hydra_zen`` + ``pydantic_parser``) and
    calls :func:`phenomica.ray_train.run_ray_train`.

    Args:
        cfg: Composed Hydra config with the groups above.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_launch.ray_train_main is a scaffold stub")


def ray_tune_main(cfg: DictConfig) -> None:
    """Console entry point for a Ray Tune hyperparameter sweep.

    Instantiates the ``model``/``teacher``/``data``/``training``/``ray_data``/
    ``ray_train``/``ray_tune`` config groups (via ``hydra_zen`` +
    ``pydantic_parser``) and calls :func:`phenomica.ray_tune.run_ray_tune`.

    Args:
        cfg: Composed Hydra config with the groups above.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_launch.ray_tune_main is a scaffold stub")
