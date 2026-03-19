"""Distillation loss functions for phenomica.

Both loss classes populate self._last_loss_metrics dict for component-level
tracking, following the pattern from txam-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Loss for simple (single-head) distillation.

    Supports MSE, cosine similarity, or weighted combination.
    The student output is compared against teacher_outputs["cls"].

    Args:
        loss_type: One of "mse", "cosine", or "combined".
        mse_weight: Weight for MSE term when using combined mode.
        cosine_weight: Weight for cosine term when using combined mode.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine", "combined"):
            raise ValueError(
                f"loss_type must be 'mse', 'cosine', or 'combined', "
                f"got '{loss_type}'"
            )
        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self._last_loss_metrics: dict[str, float] = {}

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss between student output and teacher CLS token.

        Args:
            student_output: [B, D] tensor from SimpleDistiller.
            teacher_outputs: Dict from DINOv2Teacher; uses the "cls" key.

        Returns:
            Scalar loss tensor.
        """
        teacher_cls = teacher_outputs["cls"]

        mse = F.mse_loss(student_output, teacher_cls)
        cosine = 1.0 - F.cosine_similarity(student_output, teacher_cls).mean()

        if self.loss_type == "mse":
            total = mse
        elif self.loss_type == "cosine":
            total = cosine
        else:
            total = self.mse_weight * mse + self.cosine_weight * cosine

        self._last_loss_metrics = {
            "mse": mse.item(),
            "cosine": cosine.item(),
            "total": total.item(),
        }
        return total


class MultiFunctionDistillationLoss(nn.Module):
    """Loss for multi-function (multi-head) distillation.

    Combines losses from global, spatial, and scale heads with
    configurable weights.

    Args:
        global_weight: Weight for CLS token matching loss.
        spatial_weight: Weight for patch stats matching loss.
        scale_weight: Weight for intermediate layer matching loss.
        loss_type: Base loss function for each component ("mse" or "cosine").
    """

    def __init__(
        self,
        global_weight: float = 1.0,
        spatial_weight: float = 0.5,
        scale_weight: float = 0.25,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine"):
            raise ValueError(
                f"loss_type must be 'mse' or 'cosine', got '{loss_type}'"
            )
        self.global_weight = global_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.loss_type = loss_type
        self._last_loss_metrics: dict[str, float] = {}

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute base loss between prediction and target tensors."""
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return 1.0 - F.cosine_similarity(pred, target).mean()

    def forward(
        self,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined multi-head loss.

        Args:
            student_outputs: Dict with keys "global" [B, D], "spatial" [B, D],
                and "scale" (list of [B, D] tensors).
            teacher_outputs: Dict with keys "cls" [B, D], "patch_stats" [B, D],
                and "layer_features" (list of [B, D] tensors).

        Returns:
            Scalar loss tensor.
        """
        global_loss = self._compute_loss(
            student_outputs["global"], teacher_outputs["cls"]
        )
        spatial_loss = self._compute_loss(
            student_outputs["spatial"], teacher_outputs["patch_stats"]
        )

        scale_losses = [
            self._compute_loss(s, t)
            for s, t in zip(
                student_outputs["scale"],
                teacher_outputs["layer_features"],
                strict=True,
            )
        ]
        scale_loss = torch.stack(scale_losses).mean() if scale_losses else torch.tensor(0.0)

        total = (
            self.global_weight * global_loss
            + self.spatial_weight * spatial_loss
            + self.scale_weight * scale_loss
        )

        self._last_loss_metrics = {
            "global": global_loss.item(),
            "spatial": spatial_loss.item(),
            "scale": scale_loss.item(),
            "total": total.item(),
        }
        return total
