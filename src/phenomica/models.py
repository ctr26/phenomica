"""Student model variants for DINOv2 distillation.

Provides two distillation architectures:
- SimpleDistiller: single backbone + single projection head
- MultiFunctionDistiller: shared backbone + multiple specialized heads

Both use timm backbones as feature extractors and project into the
teacher's embedding space via learned MLP projection heads.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


def _create_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Create a timm backbone configured as a feature extractor.

    Args:
        name: Model name recognized by ``timm.create_model``.
        pretrained: Whether to load pretrained ImageNet weights.

    Returns:
        A ``(backbone, feature_dim)`` tuple where *backbone* has its
        classification head removed (``num_classes=0``).
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, model.num_features


class ProjectionHead(nn.Module):
    """Two-layer MLP projection: Linear -> LayerNorm -> GELU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleDistiller(nn.Module):
    """Single backbone + single projection head.

    Distills knowledge by matching the DINOv2 CLS token embedding.

    Args:
        backbone: A timm model name (e.g. ``'resnet18'``).
        projection_dim: Output dimensionality, should match the teacher's
            ``embed_dim``.
        pretrained_backbone: Whether to initialise the backbone with
            pretrained ImageNet weights.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 768,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _create_backbone(backbone, pretrained_backbone)
        self.head = ProjectionHead(feat_dim, feat_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images and project into the teacher embedding space.

        Args:
            x: Input images ``[B, C, H, W]``.

        Returns:
            Projected features ``[B, projection_dim]``.
        """
        return self.head(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features without the projection head.

        Args:
            x: Input images ``[B, C, H, W]``.

        Returns:
            Backbone feature vectors ``[B, feat_dim]``.
        """
        return self.backbone(x)


class MultiFunctionDistiller(nn.Module):
    """Shared backbone with multiple specialised projection heads.

    Inspired by CellProfiler's multi-function feature extraction, this
    variant distils several complementary representations from a single
    forward pass through a shared backbone:

    * **global_head** -- matches the DINOv2 CLS token ``[B, teacher_cls_dim]``
    * **spatial_head** -- matches concatenated mean and std of DINOv2 patch
      tokens ``[B, teacher_patch_dim * 2]``
    * **scale_heads** -- each matches an intermediate-layer CLS token
      ``[B, teacher_cls_dim]``

    Args:
        backbone: A timm model name.
        teacher_cls_dim: Dimensionality of the teacher's CLS token.
        teacher_patch_dim: Dimensionality of the teacher's patch tokens.
        teacher_layers: Layer indices whose CLS tokens the scale heads
            should match.  Defaults to an empty list (no scale heads).
        pretrained_backbone: Whether to initialise the backbone with
            pretrained ImageNet weights.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        teacher_cls_dim: int = 768,
        teacher_patch_dim: int = 768,
        teacher_layers: list[int] | None = None,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _create_backbone(backbone, pretrained_backbone)
        self.global_head = ProjectionHead(feat_dim, feat_dim, teacher_cls_dim)
        self.spatial_head = ProjectionHead(feat_dim, feat_dim, teacher_patch_dim * 2)
        self.scale_heads = nn.ModuleList([
            ProjectionHead(feat_dim, feat_dim, teacher_cls_dim)
            for _ in (teacher_layers or [])
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode images through all heads.

        Args:
            x: Input images ``[B, C, H, W]``.

        Returns:
            Dictionary with keys:

            * ``'global'`` -- ``[B, teacher_cls_dim]``
            * ``'spatial'`` -- ``[B, teacher_patch_dim * 2]``
            * ``'scale'`` -- list of ``[B, teacher_cls_dim]`` tensors,
              one per entry in *teacher_layers*
        """
        features = self.backbone(x)
        return {
            "global": self.global_head(features),
            "spatial": self.spatial_head(features),
            "scale": [head(features) for head in self.scale_heads],
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all head outputs into a single feature vector.

        Useful for downstream evaluation where a unified representation
        is needed.  The concatenation order is:
        ``global | spatial | scale_0 | scale_1 | ...``

        Args:
            x: Input images ``[B, C, H, W]``.

        Returns:
            Concatenated features ``[B, D]`` where
            ``D = teacher_cls_dim + teacher_patch_dim * 2
            + len(teacher_layers) * teacher_cls_dim``.
        """
        outputs = self.forward(x)
        parts = [outputs["global"], outputs["spatial"], *outputs["scale"]]
        return torch.cat(parts, dim=1)


def build_model(cfg: Any) -> SimpleDistiller | MultiFunctionDistiller:
    """Factory function: create a student model from a Hydra config.

    The config must have a ``variant`` attribute set to either
    ``'simple'`` or ``'multifunction'``.

    Args:
        cfg: A Hydra/hydra-zen config object with model parameters.

    Returns:
        An initialised student model.

    Raises:
        ValueError: If ``cfg.variant`` is not recognised.
    """
    if cfg.variant == "simple":
        return SimpleDistiller(
            backbone=cfg.backbone,
            projection_dim=cfg.projection_dim,
            pretrained_backbone=cfg.pretrained_backbone,
        )
    if cfg.variant == "multifunction":
        return MultiFunctionDistiller(
            backbone=cfg.backbone,
            teacher_cls_dim=cfg.teacher_cls_dim,
            teacher_patch_dim=cfg.teacher_patch_dim,
            teacher_layers=cfg.teacher_layers,
            pretrained_backbone=cfg.pretrained_backbone,
        )
    raise ValueError(f"Unknown model variant: {cfg.variant!r}")
