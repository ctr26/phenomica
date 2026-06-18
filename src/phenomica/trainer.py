"""Training orchestrator for distillation."""

from __future__ import annotations

import itertools
import logging
import pathlib
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from phenomica.distributed import AverageMeter, is_main_process, setup_device
from phenomica.eval import evaluate_model
from phenomica.losses import MultiFunctionDistillationLoss, build_loss
from phenomica.models import build_model
from phenomica.teacher import DINOv2Teacher

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """End-to-end training orchestrator for DINOv2 distillation.

    Handles device setup, model construction, optimiser/scheduler creation,
    DDP wrapping, checkpointing, W&B logging, and early stopping.

    Args:
        training_cfg: Hydra training config (epochs, lr, optimizer, etc.).
        model_cfg: Hydra model config (variant, backbone, projection_dim, etc.).
        teacher_cfg: Hydra teacher config (model_name, embed_dim).
        data_cfg: Hydra data config (used only for logging metadata).
    """

    def __init__(
        self,
        training_cfg: Any,
        model_cfg: Any,
        teacher_cfg: Any,
        data_cfg: Any,
    ) -> None:
        # Device and distribution setup.
        self.device, self.is_distributed, self.world_size = setup_device(training_cfg.use_ddp)

        # Teacher -- frozen, never wrapped in DDP.
        extract_layers = model_cfg.teacher_layers if model_cfg.variant == "multifunction" else None
        self.teacher = DINOv2Teacher(
            model_name=teacher_cfg.model_name,
            extract_layers=extract_layers,
        ).to(self.device)

        # Sync student head dims to loaded teacher embed_dim.
        embed_dim = self.teacher.embed_dim
        if embed_dim != teacher_cfg.embed_dim:
            raise ValueError(
                f"teacher_cfg.embed_dim={teacher_cfg.embed_dim} disagrees with loaded "
                f"DINOv2 embed_dim={embed_dim}"
            )
        model_cfg.projection_dim = embed_dim
        model_cfg.teacher_cls_dim = embed_dim
        model_cfg.teacher_patch_dim = embed_dim

        # Student.
        self.model = build_model(model_cfg).to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model)

        # Loss.
        if model_cfg.variant == "multifunction":
            self.criterion = MultiFunctionDistillationLoss(
                loss_type=training_cfg.loss_type,
            )
        else:
            # Extract all training config fields as dict for loss hyperparams
            if hasattr(training_cfg, "model_dump"):
                # Pydantic config
                loss_kwargs = training_cfg.model_dump()
            elif hasattr(training_cfg, "_target_"):
                # OmegaConf structured config
                from omegaconf import OmegaConf

                loss_kwargs = OmegaConf.to_container(training_cfg, resolve=True)
            else:
                # Fallback: dataclass or plain object
                import dataclasses

                if dataclasses.is_dataclass(training_cfg):
                    loss_kwargs = dataclasses.asdict(training_cfg)
                else:
                    loss_kwargs = dict(vars(training_cfg))

            # Remove loss_type from kwargs (passed positionally)
            loss_kwargs.pop("loss_type", None)

            self.criterion = build_loss(training_cfg.loss_type, **loss_kwargs)
        self.criterion = self.criterion.to(self.device)

        # Optimizer.
        self.optimizer = self._build_optimizer(training_cfg)

        # LR scheduler.
        self.scheduler = self._build_scheduler(training_cfg)

        # Config references for checkpointing and logging.
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg
        self.teacher_cfg = teacher_cfg
        self.data_cfg = data_cfg

        self.start_epoch = 0
        self._current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # W&B (rank 0 only).
        self._wandb_run = None
        if training_cfg.use_wandb and is_main_process():
            self._init_wandb()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, cfg: Any) -> torch.optim.Optimizer:
        """Create the optimiser from config.

        Optimizes both student and criterion parameters (for parametric losses).
        """
        student = self._unwrapped_model()
        # Include both student and criterion params (criterion may have learnable params)
        params = itertools.chain(student.parameters(), self.criterion.parameters())

        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.Adam(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def _build_scheduler(self, cfg: Any) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create an LR scheduler with optional linear warmup."""
        if cfg.lr_scheduler is None:
            return None

        warmup_epochs = getattr(cfg, "warmup_epochs", 0)
        total_epochs = cfg.epochs

        if cfg.lr_scheduler == "cosine":
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=cfg.warmup_start_factor,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs - warmup_epochs,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs)

        if cfg.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        logger.warning("Unknown lr_scheduler %r, disabling.", cfg.lr_scheduler)
        return None

    def _init_wandb(self) -> None:
        """Initialise a W&B run on the main process."""
        import wandb

        self._wandb_run = wandb.init(
            project=self.training_cfg.wandb_project,
            name=self.training_cfg.wandb_run_name,
            tags=self.training_cfg.wandb_tags,
            config={
                "model": vars(self.model_cfg)
                if hasattr(self.model_cfg, "__dict__")
                else str(self.model_cfg),
                "teacher": vars(self.teacher_cfg)
                if hasattr(self.teacher_cfg, "__dict__")
                else str(self.teacher_cfg),
                "training": vars(self.training_cfg)
                if hasattr(self.training_cfg, "__dict__")
                else str(self.training_cfg),
            },
        )

    def _unwrapped_model(self) -> nn.Module:
        """Return the underlying student model (unwrap DDP if needed)."""
        if self.is_distributed:
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run a single training epoch.

        Args:
            train_loader: Training data loader yielding ``(images, labels)``.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        loss_meter = AverageMeter()
        grad_clip = getattr(self.training_cfg, "gradient_clip", None)

        progress = (
            tqdm(train_loader, desc="train", leave=False) if is_main_process() else train_loader
        )

        for images, _ in progress:
            images = images.to(self.device, non_blocking=True)

            with torch.no_grad():
                teacher_outputs = self.teacher(images)

            student_output = self.model(images)
            loss = self.criterion(student_output, teacher_outputs)

            self.optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self._unwrapped_model().parameters(), grad_clip)

            self.optimizer.step()
            loss_meter.update(loss.item(), images.size(0))

        # Synchronise average loss across ranks.
        if self.is_distributed:
            loss_tensor = torch.tensor(
                [loss_meter.sum, loss_meter.count],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_loss = loss_meter.avg

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Run a validation pass.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        loss_meter = AverageMeter()

        for images, _ in val_loader:
            images = images.to(self.device, non_blocking=True)
            teacher_outputs = self.teacher(images)
            student_output = self.model(images)
            loss = self.criterion(student_output, teacher_outputs)
            loss_meter.update(loss.item(), images.size(0))

        if self.is_distributed:
            loss_tensor = torch.tensor(
                [loss_meter.sum, loss_meter.count],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            return (loss_tensor[0] / loss_tensor[1]).item()

        return loss_meter.avg

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        save_dir: str = "checkpoints",
    ) -> None:
        """Full training loop with validation, checkpointing, and logging.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            save_dir: Directory for saving model checkpoints.
        """
        save_path = pathlib.Path(save_dir)
        if is_main_process():
            save_path.mkdir(parents=True, exist_ok=True)

        patience = getattr(self.training_cfg, "early_stopping_patience", None)
        val_freq = getattr(self.training_cfg, "validation_freq", 1)
        eval_freq = getattr(self.training_cfg, "eval_freq", None)

        for epoch in range(self.start_epoch, self.training_cfg.epochs):
            self._current_epoch = epoch

            # Set epoch on distributed sampler for proper shuffling.
            if self.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation.
            val_loss = None
            if val_loader is not None and (epoch + 1) % val_freq == 0:
                val_loss = self.validate(val_loader)

            # Feature-quality evaluation.
            eval_metrics = None
            if (
                eval_freq
                and val_loader is not None
                and (epoch + 1) % eval_freq == 0
                and is_main_process()
            ):
                eval_metrics = evaluate_model(
                    self._unwrapped_model(),
                    train_loader,
                    val_loader,
                    device=self.device,
                )

            # Logging.
            current_lr = self.optimizer.param_groups[0]["lr"]
            if is_main_process():
                log_msg = f"Epoch {epoch + 1}/{self.training_cfg.epochs}"
                log_msg += f"  train_loss={train_loss:.6f}  lr={current_lr:.2e}"
                if val_loss is not None:
                    log_msg += f"  val_loss={val_loss:.6f}"
                if eval_metrics:
                    for k, v in eval_metrics.items():
                        log_msg += f"  {k}={v:.4f}"
                logger.info(log_msg)

            if self._wandb_run is not None:
                log_dict: dict[str, float] = {
                    "train/loss": train_loss,
                    "lr": current_lr,
                    "epoch": epoch + 1,
                }
                if val_loss is not None:
                    log_dict["val/loss"] = val_loss

                # Component-level loss metrics.
                for key, val in self.criterion._last_loss_metrics.items():
                    log_dict[f"train/{key}"] = val

                if eval_metrics:
                    for key, val in eval_metrics.items():
                        log_dict[f"eval/{key}"] = val

                self._wandb_run.log(log_dict, step=epoch + 1)

            # Checkpointing and early stopping (rank 0 only).
            if is_main_process():
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(save_path / "best_model.pt")
                elif val_loss is not None:
                    self.patience_counter += 1

                if patience is not None and self.patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch + 1,
                        patience,
                    )
                    break

        # Save final checkpoint.
        if is_main_process():
            self.save_checkpoint(save_path / "final_model.pt")

        if self._wandb_run is not None:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | pathlib.Path) -> None:
        """Save model, optimiser, scheduler, config, and epoch state.

        Args:
            path: File path for the checkpoint.
        """
        state = {
            "model_state_dict": self._unwrapped_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": self._current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_cfg": self.model_cfg,
            "teacher_cfg": self.teacher_cfg,
            "training_cfg": self.training_cfg,
        }
        torch.save(state, path)
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str | pathlib.Path) -> None:
        """Restore training state from a checkpoint.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrapped_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self._current_epoch = self.start_epoch
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("Loaded checkpoint from %s (resuming from epoch %d)", path, self.start_epoch)
