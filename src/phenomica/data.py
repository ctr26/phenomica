"""Data loading and transforms for phenomica distillation training."""

import pathlib

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(
    image_size: int = 224, is_train: bool = True
) -> transforms.Compose:
    """Return standard ImageNet-style transforms.

    Args:
        image_size: Target spatial resolution for crops.
        is_train: When True, applies random augmentations suitable for
            training. When False, applies deterministic eval transforms.

    Returns:
        A ``transforms.Compose`` pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_dataloaders(
    cfg, is_distributed: bool = False
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and optional validation dataloaders from a Hydra config.

    Supports ``cfg.dataset_type == "imagefolder"`` backed by
    ``torchvision.datasets.ImageFolder``. If ``cfg.root`` contains ``train/``
    and ``val/`` subdirectories they are used directly; otherwise the dataset
    is loaded from ``cfg.root`` and split according to ``cfg.val_split``.

    Args:
        cfg: Hydra data config with attributes ``root``, ``dataset_type``,
            ``batch_size``, ``num_workers``, ``pin_memory``, ``image_size``,
            and ``val_split``.
        is_distributed: Use ``DistributedSampler`` when True.

    Returns:
        A ``(train_loader, val_loader)`` tuple. ``val_loader`` is ``None``
        when no validation data is available.
    """
    image_size = getattr(cfg, "image_size", 224)
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    root = pathlib.Path(cfg.root)
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.is_dir() and val_dir.is_dir():
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    else:
        full_dataset = datasets.ImageFolder(str(root), transform=train_transform)
        val_split = getattr(cfg, "val_split", 0.1)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        if val_size > 0:
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
        else:
            train_dataset = full_dataset
            val_dataset = None

    # -- Build samplers / loaders ------------------------------------------------
    train_sampler = None
    val_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    return train_loader, val_loader
