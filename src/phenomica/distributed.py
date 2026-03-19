"""Distributed training utilities and metric tracking."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


class AverageMeter:
    """Running average tracker for loss and metric values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int):
    """Set random seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int, int] | tuple[None, None, None]:
    """Initialize DDP from RANK/WORLD_SIZE/LOCAL_RANK env vars.

    Returns:
        (rank, world_size, local_rank) if distributed, else (None, None, None).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None, None, None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Destroy the process group if distributed training is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True if rank 0 or non-distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Current process rank (0 if non-distributed)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def setup_device(use_ddp: bool = True) -> tuple[str, bool, int]:
    """Auto-detect GPU setup and optionally initialize DDP.

    Returns:
        (device, is_distributed, world_size)
    """
    if not torch.cuda.is_available():
        return "cpu", False, 1

    if use_ddp:
        rank, world_size, local_rank = setup_distributed()
        if rank is not None:
            return f"cuda:{local_rank}", True, world_size

    return "cuda", False, 1
