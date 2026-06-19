"""Tests for the Ray Data input pipeline (CPU-only, tiny in-memory data)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

from phenomica.configs import DataConfig, RayDataConfig
from phenomica.ray_data import (
    _PATH_COLUMN,
    IMAGE_COLUMN,
    LABEL_COLUMN,
    build_ray_dataset,
    preprocess_batch,
)

_IMAGE_SIZE = 32
_CLASSES = ("cat", "dog")
_IMAGES_PER_CLASS = 3


@pytest.fixture
def image_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a tiny ImageFolder-style tree of random RGB images.

    Layout: ``<root>/<class>/<i>.png`` for two classes, mirroring
    ``torchvision.datasets.ImageFolder`` expectations.
    """
    rng = np.random.default_rng(0)
    for cls in _CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(_IMAGES_PER_CLASS):
            arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{i}.png")
    return tmp_path


def test_preprocess_batch_shape_dtype_and_labels(image_root: pathlib.Path) -> None:
    """preprocess_batch normalizes to CHW float32 and derives int64 labels."""
    raw = np.random.default_rng(1).integers(0, 256, (2, 40, 50, 3), dtype=np.uint8)
    paths = [str(image_root / "cat" / "0.png"), str(image_root / "dog" / "1.png")]
    batch = {IMAGE_COLUMN: raw, _PATH_COLUMN: np.array(paths)}

    out = preprocess_batch(batch, image_size=_IMAGE_SIZE, label_map={"cat": 0, "dog": 1})

    assert out[IMAGE_COLUMN].shape == (2, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    assert out[IMAGE_COLUMN].dtype == np.float32
    assert out[LABEL_COLUMN].tolist() == [0, 1]
    assert out[LABEL_COLUMN].dtype == np.int64


def test_preprocess_batch_without_label_map_omits_labels() -> None:
    """Labels are omitted when no label_map is supplied (inference path)."""
    raw = np.random.default_rng(2).integers(0, 256, (1, 40, 50, 3), dtype=np.uint8)
    out = preprocess_batch({IMAGE_COLUMN: raw}, image_size=_IMAGE_SIZE)

    assert LABEL_COLUMN not in out
    assert out[IMAGE_COLUMN].shape == (1, 3, _IMAGE_SIZE, _IMAGE_SIZE)


@pytest.mark.slow
def test_build_ray_dataset_schema_and_iter(image_root: pathlib.Path) -> None:
    """build_ray_dataset yields a dataset with the right schema and batches."""
    ray = pytest.importorskip("ray")
    import torch

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        data_cfg = DataConfig(root=str(image_root), image_size=_IMAGE_SIZE, batch_size=2)
        dataset = build_ray_dataset(data_cfg, RayDataConfig(), split="train")

        assert dataset.count() == len(_CLASSES) * _IMAGES_PER_CLASS
        columns = set(dataset.schema().names)
        assert {IMAGE_COLUMN, LABEL_COLUMN} <= columns

        shard = dataset.iter_torch_batches(batch_size=2, dtypes=torch.float32)
        batch = next(iter(shard))
        assert batch[IMAGE_COLUMN].shape == (2, 3, _IMAGE_SIZE, _IMAGE_SIZE)
        assert batch[IMAGE_COLUMN].dtype == torch.float32
        assert set(batch[LABEL_COLUMN].tolist()) <= {0, 1}
    finally:
        ray.shutdown()
