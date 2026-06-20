"""Tests for the Ray Tune sweep path (CPU-only, local, tiny data).

Runs a real :func:`phenomica.ray_tune.run_ray_tune` end to end: two trials, a
single local CPU worker each, one epoch, against a tiny ImageFolder tree, with a
network-free fake DINOv2 teacher installed in every Ray worker via
``worker_process_setup_hook`` and W&B disabled. Asserts the sweep returns a
populated ``ResultGrid`` whose best trial reports the tuned metric.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from PIL import Image

from phenomica.configs import (
    DataConfig,
    ModelConfig,
    RayDataConfig,
    RayTrainConfig,
    RayTuneConfig,
    RayTuneSearchSpace,
    TeacherConfig,
    TrainingConfig,
)
from phenomica.ray_tune import build_search_space

# Repo root: ``<root>/src`` and ``<root>`` are added to the Ray worker
# PYTHONPATH so workers can import this module's setup hook.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_IMAGE_SIZE = 32
_EMBED_DIM = 768
_CLASSES = ("cat", "dog")
_IMAGES_PER_CLASS = 4
_METRIC = "val_loss"


def _install_fake_teacher() -> None:
    """Swap the DINOv2 teacher for a CPU fake in the calling worker process."""
    import torch
    import torch.nn as nn

    import phenomica.teacher as teacher_module

    class FakeTeacher(nn.Module):
        def __init__(self, model_name: str = "dinov2_vitb14", extract_layers=None):
            super().__init__()
            self._embed_dim = _EMBED_DIM
            self._extract_layers = list(extract_layers) if extract_layers else [11]

        @property
        def embed_dim(self) -> int:
            return self._embed_dim

        def forward(self, x):
            b, d = x.size(0), self._embed_dim
            return {
                "cls": torch.randn(b, d),
                "patch_tokens": torch.randn(b, 256, d),
                "patch_stats": torch.randn(b, d * 2),
                "layer_features": [torch.randn(b, d) for _ in self._extract_layers],
            }

    teacher_module.DINOv2Teacher = FakeTeacher


@pytest.fixture
def image_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a tiny ``train``/``val`` ImageFolder tree of random RGB images."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for cls in _CLASSES:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(_IMAGES_PER_CLASS):
                arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(cls_dir / f"{i}.png")
    return tmp_path


def test_build_search_space_nests_train_loop_config() -> None:
    """The search space nests Ray Tune domains under ``train_loop_config``."""
    from ray.tune.search.sample import Categorical, Float

    tune_cfg = RayTuneConfig(search_space=RayTuneSearchSpace(loss_types=["mse", "cosine"]))
    space = build_search_space(tune_cfg)

    assert set(space) == {"train_loop_config"}
    inner = space["train_loop_config"]
    assert isinstance(inner["lr"], Float)
    assert isinstance(inner["weight_decay"], Float)
    assert isinstance(inner["loss_type"], Categorical)
    assert inner["loss_type"].categories == ["mse", "cosine"]


@pytest.mark.slow
def test_run_ray_tune_returns_results_with_metric(image_root: pathlib.Path) -> None:
    """A two-trial CPU sweep completes and reports the tuned metric."""
    ray = pytest.importorskip("ray")

    # Tune-over-Train resources: one concurrent trial needs 1 CPU for the Tune
    # driver fn + 1 for the Train controller + ``num_workers`` worker CPUs.
    # With ``max_concurrent_trials=1`` and a single worker, 4 CPUs avoids the
    # nested placement-group starvation that hangs a 2-CPU cluster.
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={
            "worker_process_setup_hook": _install_fake_teacher,
            "env_vars": {"PYTHONPATH": f"{_REPO_ROOT / 'src'}:{_REPO_ROOT}"},
        },
    )
    try:
        results = _run_short_sweep(image_root)
    finally:
        ray.shutdown()

    assert len(results) == 2
    assert results.num_errors == 0

    best = results.get_best_result(metric=_METRIC, mode="min")
    assert best.metrics is not None
    assert math.isfinite(best.metrics[_METRIC])


def _run_short_sweep(root: pathlib.Path):
    """Drive ``run_ray_tune`` with smoke-sized, CPU-only, offline configs."""
    from phenomica.ray_tune import run_ray_tune

    model_cfg = ModelConfig(variant="simple", backbone="resnet18", pretrained_backbone=False)
    teacher_cfg = TeacherConfig(model_name="dinov2_vitb14", embed_dim=_EMBED_DIM)
    data_cfg = DataConfig(root=str(root), image_size=_IMAGE_SIZE, batch_size=4)
    training_cfg = TrainingConfig(use_wandb=False, loss_type="mse")
    ray_train_cfg = RayTrainConfig(num_workers=1, use_gpu=False, max_epochs=1)
    ray_data_cfg = RayDataConfig()
    ray_tune_cfg = RayTuneConfig(
        num_samples=2,
        metric=_METRIC,
        mode="min",
        grace_period=1,
        max_t=1,
        max_concurrent_trials=1,
        search_space=RayTuneSearchSpace(loss_types=["mse"]),
    )

    return run_ray_tune(
        model_cfg,
        teacher_cfg,
        data_cfg,
        training_cfg,
        ray_train_cfg,
        ray_data_cfg,
        ray_tune_cfg,
    )
