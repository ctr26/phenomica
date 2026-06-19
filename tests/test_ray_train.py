"""Tests for the Ray Train distillation path (CPU-only, local, tiny data).

Runs a real :func:`phenomica.ray_train.run_ray_train` end to end on a single
local CPU worker against a tiny ImageFolder tree, with a network-free fake
DINOv2 teacher installed in the Ray worker via ``worker_process_setup_hook``
and W&B disabled, then asserts a finite ``train_loss`` is reported.
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
    TeacherConfig,
    TrainingConfig,
)

# Repo root: ``<root>/src`` (phenomica) and ``<root>`` (tests) are added to the
# Ray worker PYTHONPATH so the worker can import this module's setup hook.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_IMAGE_SIZE = 32
_EMBED_DIM = 768
_CLASSES = ("cat", "dog")
_IMAGES_PER_CLASS = 4


def _install_fake_teacher() -> None:
    """Swap the DINOv2 teacher for a CPU fake in the calling process.

    Used as a Ray ``worker_process_setup_hook`` so every Train worker builds a
    network-free teacher matching the :class:`DINOv2Teacher` output contract.
    """
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


@pytest.mark.slow
def test_run_ray_train_reports_finite_loss(image_root: pathlib.Path) -> None:
    """A short single-worker CPU run completes and reports a finite loss."""
    ray = pytest.importorskip("ray")

    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={
            "worker_process_setup_hook": _install_fake_teacher,
            "env_vars": {"PYTHONPATH": f"{_REPO_ROOT / 'src'}:{_REPO_ROOT}"},
        },
    )
    try:
        result = run_short_distillation(image_root)
    finally:
        ray.shutdown()

    assert result.metrics is not None
    assert "train_loss" in result.metrics
    assert math.isfinite(result.metrics["train_loss"])
    assert math.isfinite(result.metrics["val_loss"])


def run_short_distillation(root: pathlib.Path):
    """Drive ``run_ray_train`` with smoke-sized, CPU-only, offline configs."""
    from phenomica.ray_train import run_ray_train

    model_cfg = ModelConfig(variant="simple", backbone="resnet18", pretrained_backbone=False)
    teacher_cfg = TeacherConfig(model_name="dinov2_vitb14", embed_dim=_EMBED_DIM)
    data_cfg = DataConfig(root=str(root), image_size=_IMAGE_SIZE, batch_size=4)
    training_cfg = TrainingConfig(use_wandb=False, loss_type="mse")
    ray_train_cfg = RayTrainConfig(num_workers=1, use_gpu=False, max_epochs=1)
    ray_data_cfg = RayDataConfig()

    return run_ray_train(
        model_cfg, teacher_cfg, data_cfg, training_cfg, ray_train_cfg, ray_data_cfg
    )
