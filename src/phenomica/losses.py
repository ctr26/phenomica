"""Distillation loss functions for phenomica.

Both loss classes populate self._last_loss_metrics dict for component-level
tracking, following the pattern from txam-training.
"""

from __future__ import annotations

import inspect
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Loss registry for extensible loss types
LOSS_REGISTRY: dict[str, type[nn.Module]] = {}


def register_loss(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Decorator to register a loss class in the global registry.

    Args:
        name: Loss type identifier (e.g., "cospress", "vitkd").

    Returns:
        Decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def _filter_kwargs(target_cls: type, kwargs: dict) -> dict:
    """Filter kwargs to only those accepted by target_cls.__init__.

    If the constructor accepts **kwargs (VAR_KEYWORD), pass everything.
    Otherwise, pass only kwargs whose names match constructor parameters.

    Args:
        target_cls: The class whose __init__ to inspect.
        kwargs: Candidate keyword arguments.

    Returns:
        Filtered kwargs dict safe for target_cls(**filtered).
    """
    sig = inspect.signature(target_cls.__init__)
    # Check if __init__ accepts **kwargs
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return kwargs

    # Filter to declared parameters (excluding 'self')
    valid_params = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory to construct a loss module by type name.

    Preserves existing behavior for "mse"/"cosine"/"combined" (mapped to
    DistillationLoss) and constructs registered losses from LOSS_REGISTRY.
    Filters kwargs to only those accepted by the target constructor, allowing
    a superset of hyperparams to be passed safely.

    Args:
        loss_type: Loss identifier (e.g., "mse", "cospress", "vitkd").
        **kwargs: Superset of hyperparams; filtered to target constructor signature.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If loss_type is unknown.
    """
    # Existing builtin types map to DistillationLoss
    if loss_type in ("mse", "cosine", "combined"):
        filtered = _filter_kwargs(DistillationLoss, kwargs)
        return DistillationLoss(loss_type=loss_type, **filtered)

    # Registered custom losses
    if loss_type in LOSS_REGISTRY:
        target_cls = LOSS_REGISTRY[loss_type]
        filtered = _filter_kwargs(target_cls, kwargs)
        return target_cls(**filtered)

    # Unknown type
    known = ["mse", "cosine", "combined"] + list(LOSS_REGISTRY.keys())
    raise ValueError(f"Unknown loss_type='{loss_type}'. Known types: {known}")


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
                f"loss_type must be 'mse', 'cosine', or 'combined', got '{loss_type}'"
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
            raise ValueError(f"loss_type must be 'mse' or 'cosine', got '{loss_type}'")
        self.global_weight = global_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.loss_type = loss_type
        self._last_loss_metrics: dict[str, float] = {}

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        global_loss = self._compute_loss(student_outputs["global"], teacher_outputs["cls"])
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


@register_loss("cospress")
class CosPressLoss(nn.Module):
    """CosPress: Preserves pairwise cosine-similarity structure from teacher.

    Based on "Preserving Angles Improves Feature Distillation of Foundation Models"
    (arxiv 2411.15239). Uses relational KL divergence over normalized similarity
    matrices plus optional direct cosine term.

    Args:
        cospress_weight: Weight for the KL divergence term.
        cospress_temperature: Temperature for softmax over similarity matrices.
        cospress_cosine_weight: Weight for direct cosine similarity term
            (only applied when student and teacher have matching dimensions).
    """

    def __init__(
        self,
        cospress_weight: float = 1.0,
        cospress_temperature: float = 0.1,
        cospress_cosine_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.cospress_weight = cospress_weight
        self.cospress_temperature = cospress_temperature
        self.cospress_cosine_weight = cospress_cosine_weight
        self._last_loss_metrics: dict[str, float] = {}
        self._dim_mismatch_logged = False

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute CosPress loss between student and teacher CLS embeddings.

        Args:
            student_output: [B, D_student] student embeddings.
            teacher_outputs: Dict containing "cls" key with [B, D_teacher] embeddings.

        Returns:
            Scalar loss tensor.
        """
        teacher_cls = teacher_outputs["cls"]
        batch_size = student_output.size(0)

        # Normalize embeddings to unit sphere
        student_norm = F.normalize(student_output, dim=-1)
        teacher_norm = F.normalize(teacher_cls, dim=-1)

        # Compute similarity matrices [B, B]
        sim_student = student_norm @ student_norm.T
        sim_teacher = teacher_norm @ teacher_norm.T

        # Mask diagonal (self-similarity) to -inf for softmax
        mask = torch.eye(batch_size, device=student_output.device, dtype=torch.bool)
        sim_student_masked = sim_student.masked_fill(mask, float("-inf"))
        sim_teacher_masked = sim_teacher.masked_fill(mask, float("-inf"))

        # Convert to distributions with temperature
        # Teacher probabilities (target)
        teacher_probs = F.softmax(sim_teacher_masked / self.cospress_temperature, dim=-1)
        # Student log-probabilities
        student_log_probs = F.log_softmax(sim_student_masked / self.cospress_temperature, dim=-1)

        # KL divergence: KL(teacher || student), averaged over batch
        loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        # Optional direct cosine term (only if dimensions match)
        if student_output.shape[-1] == teacher_cls.shape[-1]:
            cos_sim = F.cosine_similarity(student_output, teacher_cls, dim=-1)
            loss_cosine = (1.0 - cos_sim).mean()
            cosine_val = loss_cosine.item()
        else:
            if not self._dim_mismatch_logged:
                logger.info(
                    f"CosPressLoss: dimension mismatch (student {student_output.shape[-1]} "
                    f"vs teacher {teacher_cls.shape[-1]}), skipping cosine term"
                )
                self._dim_mismatch_logged = True
            loss_cosine = torch.tensor(0.0, device=student_output.device)
            cosine_val = 0.0

        # Total loss
        total = self.cospress_weight * loss_kl + self.cospress_cosine_weight * loss_cosine

        self._last_loss_metrics = {
            "cospress_kl": loss_kl.item(),
            "cospress_cosine": cosine_val,
            "total": total.item(),
        }


@register_loss("vitkd")
class ViTKDLoss(nn.Module):
    """ViTKD loss: shallow-layer direct mimicry + deep-layer generative reconstruction.

    Adapted from arxiv 2209.02432 for pooled student embeddings (not token-level).
    Direct term: shallow teacher layers' patch tokens (mean-pooled) vs projected student.
    Generative term: reconstructs deep teacher layer's masked patch tokens from student.

    Handles variable teacher layer counts robustly. If len(layer_patch_tokens)==1,
    uses it as the deep target; direct term is either skipped or uses the same layer.

    Args:
        vitkd_student_dim: Student embedding dimension (must match student_output.shape[-1]).
        vitkd_teacher_dim: Teacher embedding dimension (typically 768 for DINOv2 base).
        vitkd_num_tokens: Number of patch tokens N (typically 256 for 224x224/14x14).
        vitkd_mask_ratio: Fraction of tokens to mask in generative term [0.0, 1.0].
        vitkd_weight: Weight for direct mimicry term.
        vitkd_gen_weight: Weight for generative reconstruction term.
    """

    def __init__(
        self,
        vitkd_student_dim: int = 768,
        vitkd_teacher_dim: int = 768,
        vitkd_num_tokens: int = 256,
        vitkd_mask_ratio: float = 0.5,
        vitkd_weight: float = 1.0,
        vitkd_gen_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.vitkd_student_dim = vitkd_student_dim
        self.vitkd_teacher_dim = vitkd_teacher_dim
        self.vitkd_num_tokens = vitkd_num_tokens
        self.vitkd_mask_ratio = vitkd_mask_ratio
        self.vitkd_weight = vitkd_weight
        self.vitkd_gen_weight = vitkd_gen_weight

        # Direct term: project student -> teacher dim for shallow layer matching
        self.direct_proj = nn.Linear(vitkd_student_dim, vitkd_teacher_dim)

        # Generative term: reconstruct deep layer's N×D tokens from student embedding
        # Two-stage: project to teacher dim, then generate N token vectors
        self.gen_proj = nn.Linear(vitkd_student_dim, vitkd_teacher_dim)
        self.gen_decoder = nn.Linear(vitkd_teacher_dim, vitkd_num_tokens * vitkd_teacher_dim)

        self._last_loss_metrics: dict[str, float] = {}

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute ViTKD loss.

        Args:
            student_output: [B, vitkd_student_dim] pooled student embedding.
            teacher_outputs: Dict with "layer_patch_tokens" list of [B, N, vitkd_teacher_dim].

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If student_output dim doesn't match vitkd_student_dim.
        """
        B = student_output.size(0)

        # Validate student dim
        if student_output.shape[-1] != self.vitkd_student_dim:
            raise ValueError(
                f"student_output dim {student_output.shape[-1]} "
                f"!= vitkd_student_dim {self.vitkd_student_dim}"
            )

        layer_patch_tokens = teacher_outputs["layer_patch_tokens"]
        num_layers = len(layer_patch_tokens)

        # DIRECT TERM: shallow layers (all but last, or reuse single layer if len==1)
        if num_layers > 1:
            # Use all but the last layer as shallow
            shallow_layers = layer_patch_tokens[:-1]
        else:
            # Single layer: reuse it for direct term (documented limitation)
            shallow_layers = layer_patch_tokens

        # Project student once for direct term
        projected_student = self.direct_proj(student_output)  # [B, D_teacher]

        # Compute direct loss: MSE between projected student and mean-pooled shallow teacher layers
        direct_losses = []
        for shallow_tokens in shallow_layers:  # Each [B, N, D_teacher]
            teacher_shallow_summary = shallow_tokens.mean(dim=1)  # [B, D_teacher]
            direct_losses.append(F.mse_loss(projected_student, teacher_shallow_summary))
        direct_loss = torch.stack(direct_losses).mean() if direct_losses else torch.tensor(0.0)

        # GENERATIVE TERM: deepest layer
        deep_tokens = layer_patch_tokens[-1]  # [B, N, D_teacher]

        # Generate reconstructed tokens from student
        gen_emb = self.gen_proj(student_output)  # [B, D_teacher]
        gen_flat = self.gen_decoder(gen_emb)  # [B, N*D_teacher]
        gen_tokens = gen_flat.view(B, self.vitkd_num_tokens, self.vitkd_teacher_dim)  # [B, N, D]

        # Mask a fraction of tokens and compute MSE only on masked positions
        N = deep_tokens.size(1)
        num_masked = max(1, int(N * self.vitkd_mask_ratio))

        # Random mask per batch element
        mask_indices = torch.rand(B, N, device=deep_tokens.device).argsort(dim=1)[:, :num_masked]

        # Gather masked positions from both generated and target
        batch_idx = torch.arange(B, device=deep_tokens.device).unsqueeze(1).expand(-1, num_masked)
        gen_masked = gen_tokens[batch_idx, mask_indices]  # [B, num_masked, D]
        target_masked = deep_tokens[batch_idx, mask_indices]  # [B, num_masked, D]

        gen_loss = F.mse_loss(gen_masked, target_masked)

        # Total weighted loss
        total = self.vitkd_weight * direct_loss + self.vitkd_gen_weight * gen_loss

        self._last_loss_metrics = {
            "vitkd_direct": direct_loss.item(),
            "vitkd_gen": gen_loss.item(),
            "total": total.item(),
        }
        return total


__all__ = [
    "DistillationLoss",
    "MultiFunctionDistillationLoss",
    "CosPressLoss",
    "ViTKDLoss",
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
]


@register_loss("rekd")
class ReKDLoss(nn.Module):
    """Relation Knowledge Distillation (ReKD) loss.

    Implements in-batch multi-positive contrastive learning guided by teacher
    semantic similarity (arxiv 2112.04174). For each student anchor, the teacher's
    top-k most similar samples become positives in a multi-positive InfoNCE loss,
    pulling the student's relational neighborhoods toward the teacher's.

    Args:
        rekd_temperature: Softmax temperature for student similarities.
        rekd_topk: Number of semantic positives per anchor (clamped to batch size - 1).
        rekd_weight: Global weight for the loss.
    """

    def __init__(
        self,
        rekd_temperature: float = 0.1,
        rekd_topk: int = 5,
        rekd_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.rekd_temperature = rekd_temperature
        self.rekd_topk = rekd_topk
        self.rekd_weight = rekd_weight
        self._last_loss_metrics: dict[str, float] = {}

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute ReKD contrastive loss.

        Args:
            student_output: [B, D_s] student embeddings.
            teacher_outputs: Dict containing "cls" [B, D_t] teacher embeddings.

        Returns:
            Scalar loss tensor.
        """
        teacher_cls = teacher_outputs["cls"]
        B = student_output.size(0)

        # Handle degenerate batch sizes
        if B < 2:
            zero_loss = torch.tensor(0.0, device=student_output.device, dtype=student_output.dtype)
            self._last_loss_metrics = {"rekd_contrastive": 0.0, "total": 0.0}
            return zero_loss

        # Clamp topk to valid range
        effective_topk = min(self.rekd_topk, B - 1)

        # Teacher relation graph: normalize and compute similarity matrix
        teacher_norm = F.normalize(teacher_cls, dim=-1)
        S_t = teacher_norm @ teacher_norm.T  # [B, B]

        # Mine semantic positives: top-k most similar (excluding self)
        # Mask self-similarities with -inf before top-k
        S_t_masked = S_t.clone()
        S_t_masked.fill_diagonal_(-float("inf"))
        _, topk_indices = torch.topk(S_t_masked, k=effective_topk, dim=1)  # [B, k]

        # Student similarities
        student_norm = F.normalize(student_output, dim=-1)
        S_s = student_norm @ student_norm.T  # [B, B]

        # Multi-positive InfoNCE: for each anchor i,
        # loss_i = -log(sum_p exp(s_ip/T) / sum_j exp(s_ij/T))
        # Build positive mask [B, B] from topk_indices
        pos_mask = torch.zeros_like(S_s, dtype=torch.bool)
        batch_indices = torch.arange(B, device=S_s.device).unsqueeze(1)
        pos_mask[batch_indices, topk_indices] = True

        # Logits: mask self with -inf in denominator
        logits = S_s / self.rekd_temperature
        logits_masked = logits.clone()
        logits_masked.fill_diagonal_(-float("inf"))

        # Numerator: logsumexp over positives
        neg_inf = torch.tensor(-float("inf"), device=logits.device)
        pos_logits = torch.where(pos_mask, logits, neg_inf)
        log_numerator = torch.logsumexp(pos_logits, dim=1)  # [B]

        # Denominator: logsumexp over all (excluding self)
        log_denominator = torch.logsumexp(logits_masked, dim=1)  # [B]

        # Loss per anchor
        loss_per_anchor = -(log_numerator - log_denominator)
        contrastive = loss_per_anchor.mean()

        total = self.rekd_weight * contrastive

        self._last_loss_metrics = {
            "rekd_contrastive": contrastive.item(),
            "total": total.item(),
        }
        return total


__all__ = [
    "DistillationLoss",
    "MultiFunctionDistillationLoss",
    "CosPressLoss",
    "ViTKDLoss",
    "ReKDLoss",
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
]
