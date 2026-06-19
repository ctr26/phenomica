"""Ray Data ingest and preprocessing for distillation (SCAFFOLD STUB).

Builds a ``ray.data.Dataset`` of images from a :class:`~phenomica.configs.DataConfig`
root and applies the same ImageNet-style normalization the torch
``data.create_dataloaders`` path uses, so the Ray Train path consumes
tensors in the teacher's expected input space.

This is the Ray-native INDEPENDENT data path; the torch ``DataLoader`` path
in :mod:`phenomica.data` is untouched and still backs the submitit launch.

All bodies are intentionally unimplemented (scaffold). Build workers fill
them against the fixed signatures below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ray.data

    from phenomica.configs import DataConfig, RayDataConfig

logger = logging.getLogger(__name__)

# Output column names produced by ``preprocess_batch`` and consumed by the
# Ray Train worker's ``iter_torch_batches`` loop.
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"


def preprocess_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Resize, scale, and ImageNet-normalize an image batch in place.

    Designed for ``ray.data.Dataset.map_batches`` with
    ``batch_format="numpy"``. Mirrors the eval transforms in
    :func:`phenomica.data.get_transforms` (resize -> center-crop ->
    to-tensor -> normalize) so Ray-fed images match the torch path.

    Args:
        batch: Mapping with an ``IMAGE_COLUMN`` entry holding a batch of
            HWC uint8 image arrays (as produced by ``ray.data.read_images``)
            and an optional ``LABEL_COLUMN`` entry.

    Returns:
        The batch with ``IMAGE_COLUMN`` replaced by CHW float32 arrays
        normalized to ImageNet mean/std.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_data.preprocess_batch is a scaffold stub")


def build_ray_dataset(
    data_cfg: DataConfig,
    ray_data_cfg: RayDataConfig,
    *,
    split: str = "train",
) -> ray.data.Dataset:
    """Build a preprocessed ``ray.data.Dataset`` from an image directory.

    Reads images under ``data_cfg.root`` (optionally a ``train``/``val``
    subdirectory selected by ``split``) via ``ray.data.read_images`` with
    class subfolders as labels, then applies :func:`preprocess_batch` via
    ``map_batches``. Read parallelism is taken from ``ray_data_cfg``.

    Args:
        data_cfg: Dataset config providing ``root``, ``image_size``, and
            ``dataset_type``.
        ray_data_cfg: Ray Data tuning knobs (parallelism, block override).
        split: Which split subdirectory to load (``"train"`` or ``"val"``);
            falls back to ``data_cfg.root`` when no split subdirs exist.

    Returns:
        A lazily-preprocessed ``ray.data.Dataset`` yielding rows with
        ``IMAGE_COLUMN`` (CHW float32) and ``LABEL_COLUMN`` (int) columns.

    Raises:
        NotImplementedError: Always -- scaffold stub.
    """
    raise NotImplementedError("ray_data.build_ray_dataset is a scaffold stub")
