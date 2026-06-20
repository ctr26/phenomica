"""Ray Data ingest and preprocessing for distillation.

Builds a ``ray.data.Dataset`` of images from a :class:`~phenomica.configs.DataConfig`
root and applies the same ImageNet-style eval transforms the torch
``data.create_dataloaders`` path uses, so the Ray Train path consumes
tensors in the teacher's expected input space.

This is the Ray-native INDEPENDENT data path; the torch ``DataLoader`` path
in :mod:`phenomica.data` is untouched and still backs the submitit launch.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from phenomica.data import get_transforms

if TYPE_CHECKING:
    import ray.data

    from phenomica.configs import DataConfig, RayDataConfig

logger = logging.getLogger(__name__)

# Output column names produced by ``preprocess_batch`` and consumed by the
# Ray Train worker's ``iter_torch_batches`` loop.
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Key ``ray.data.read_images(include_paths=True)`` adds for each row's source
# path; used to derive integer class labels from the parent directory name.
_PATH_COLUMN = "path"

# Standard split subdirectories selected by ``split`` when present under root.
_SPLIT_SUBDIRS = ("train", "val")


def _build_label_map(root: pathlib.Path) -> dict[str, int]:
    """Map sorted class-subfolder names under ``root`` to integer labels.

    Mirrors ``torchvision.datasets.ImageFolder`` semantics (classes are the
    immediate subdirectories, sorted, indexed from 0) so Ray-derived labels
    match the torch path.

    Args:
        root: Directory whose immediate subdirectories name the classes.

    Returns:
        Mapping from class-folder name to its integer label.
    """
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    return {name: idx for idx, name in enumerate(classes)}


def preprocess_batch(
    batch: dict[str, Any],
    *,
    image_size: int = 224,
    label_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Resize, scale, and ImageNet-normalize an image batch.

    Designed for ``ray.data.Dataset.map_batches`` with
    ``batch_format="numpy"``. Reuses the eval transforms from
    :func:`phenomica.data.get_transforms` (resize -> center-crop ->
    to-tensor -> normalize) per image, so Ray-fed images match the torch
    path exactly.

    Args:
        batch: Mapping with an ``IMAGE_COLUMN`` entry holding a batch of
            HWC uint8 image arrays (as produced by ``ray.data.read_images``)
            and, when ``label_map`` is given, a ``_PATH_COLUMN`` entry of
            source paths from which integer labels are derived.
        image_size: Target square crop resolution.
        label_map: Optional class-name -> integer-label mapping. When
            provided, a ``LABEL_COLUMN`` of int64 labels is emitted from the
            parent directory of each ``_PATH_COLUMN`` value.

    Returns:
        A batch dict with ``IMAGE_COLUMN`` as CHW float32 arrays normalized
        to ImageNet mean/std and, when labels were derivable, a
        ``LABEL_COLUMN`` of int64 labels.
    """
    transform = get_transforms(image_size, is_train=False)
    images = [
        transform(Image.fromarray(img).convert("RGB")).numpy() for img in batch[IMAGE_COLUMN]
    ]
    out: dict[str, Any] = {IMAGE_COLUMN: np.stack(images).astype(np.float32)}

    if label_map is not None and _PATH_COLUMN in batch:
        out[LABEL_COLUMN] = np.array(
            [label_map[pathlib.Path(p).parent.name] for p in batch[_PATH_COLUMN]],
            dtype=np.int64,
        )
    return out


def _resolve_split_dir(root: pathlib.Path, split: str) -> pathlib.Path:
    """Return the directory to read for ``split``, falling back to ``root``.

    Uses ``root/<split>`` when both standard split subdirectories exist
    (matching :func:`phenomica.data.create_dataloaders`), otherwise reads
    flat from ``root``.

    Args:
        root: Dataset root directory.
        split: Requested split name (``"train"`` or ``"val"``).

    Returns:
        The resolved directory to read images from.
    """
    has_splits = all((root / sub).is_dir() for sub in _SPLIT_SUBDIRS)
    if has_splits:
        return root / split
    return root


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
            ``batch_size``.
        ray_data_cfg: Ray Data tuning knobs (parallelism, block override).
        split: Which split subdirectory to load (``"train"`` or ``"val"``);
            falls back to ``data_cfg.root`` when no split subdirs exist.

    Returns:
        A lazily-preprocessed ``ray.data.Dataset`` yielding rows with
        ``IMAGE_COLUMN`` (CHW float32) and ``LABEL_COLUMN`` (int64) columns.
    """
    import ray.data

    read_dir = _resolve_split_dir(pathlib.Path(data_cfg.root), split)
    label_map = _build_label_map(read_dir)
    logger.info(
        "Reading images for split=%s from %s (%d classes)",
        split,
        read_dir,
        len(label_map),
    )

    override_num_blocks = ray_data_cfg.override_num_blocks
    if override_num_blocks is None and ray_data_cfg.parallelism > 0:
        override_num_blocks = ray_data_cfg.parallelism

    dataset = ray.data.read_images(
        str(read_dir),
        mode="RGB",
        include_paths=True,
        override_num_blocks=override_num_blocks,
    )
    return dataset.map_batches(
        preprocess_batch,
        batch_format="numpy",
        batch_size=data_cfg.batch_size,
        fn_kwargs={"image_size": data_cfg.image_size, "label_map": label_map},
    )
