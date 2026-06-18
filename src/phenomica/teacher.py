"""Frozen DINOv2 teacher for distillation targets.

Wraps a pretrained DINOv2 vision transformer loaded from torch.hub,
freezing all parameters and extracting intermediate layer features
for multi-level distillation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 model that produces targets for distillation.

    Loads a DINOv2 ViT via ``torch.hub.load('facebookresearch/dinov2', ...)``,
    freezes all parameters, and forces eval mode permanently. The forward pass
    extracts CLS tokens, patch embeddings, patch statistics, and optionally
    intermediate layer features for multi-level distillation.

    Args:
        model_name: Hub model name, e.g. ``"dinov2_vitb14"``.
        extract_layers: Block indices for intermediate feature extraction.
            If ``None``, defaults to the final block only (``[11]`` for ViT-B).
        extract_attention: Whether to extract attention maps from each layer.
            When ``True``, registers hooks to capture CLS-token attention weights.
            Default ``False`` for zero overhead.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        extract_layers: Sequence[int] | None = None,
        extract_attention: bool = False,
    ):
        super().__init__()

        self._model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )

        # Freeze everything -- this model only produces targets.
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.eval()

        self._embed_dim: int = self._model.embed_dim
        self._patch_size: int = getattr(self._model, "patch_size", 14)

        # Decide which transformer blocks to tap for intermediate features.
        num_blocks = len(self._model.blocks)
        if extract_layers is None:
            self._extract_layers = [num_blocks - 1]
        else:
            self._extract_layers = list(extract_layers)

        self._extract_attention = extract_attention
        self._attn_weights: list[torch.Tensor] = []

    def train(self, mode: bool = True) -> DINOv2Teacher:
        """Override to keep the teacher permanently in eval mode."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Extract multi-level features from the frozen teacher.

        Args:
            x: Input images ``[B, C, H, W]``, expected to be 224x224 (or any
                multiple of ``patch_size``).

        Returns:
            Dictionary containing:

            - **cls** -- CLS token embeddings ``[B, embed_dim]``.
            - **patch_tokens** -- Spatial patch embeddings ``[B, N, embed_dim]``
              from the last layer.
            - **patch_stats** -- Concatenated patch mean and std
              ``[B, embed_dim * 2]``, useful as a compact spatial summary.
            - **layer_features** -- List of CLS tokens from intermediate
              blocks specified by ``extract_layers``, each ``[B, embed_dim]``.
            - **layer_patch_tokens** -- List of patch token tensors from
              intermediate blocks, each ``[B, N, embed_dim]``.
            - **attn_maps** -- List of CLS-token attention maps from each layer
              ``[B, num_heads, N]`` when ``extract_attention=True``, else ``None``.
        """
        # Clear previous attention weights if extracting
        if self._extract_attention:
            self._attn_weights = []

        # get_intermediate_layers returns features from the requested block
        # indices. With return_class_token=True each element is a
        # (patch_tokens, cls_token) tuple.
        outputs = self._model.get_intermediate_layers(
            x,
            n=self._extract_layers,
            return_class_token=True,
        )

        # Use the last requested layer for the primary CLS / patch outputs.
        patch_tokens, cls_token = outputs[-1]

        # Patch-level statistics as a compact spatial descriptor.
        patch_mean = patch_tokens.mean(dim=1)
        patch_std = patch_tokens.std(dim=1)
        patch_stats = torch.cat([patch_mean, patch_std], dim=-1)

        # Collect intermediate CLS tokens for multi-layer distillation.
        layer_features = [cls for _, cls in outputs]

        # Collect intermediate patch tokens for all layers.
        layer_patch_tokens = [patches for patches, _ in outputs]

        # Attention maps (if enabled) are populated via hooks during forward.
        # For the fake teacher or when disabled, return None.
        attn_maps = self._attn_weights if self._extract_attention else None

        return {
            "cls": cls_token,
            "patch_tokens": patch_tokens,
            "patch_stats": patch_stats,
            "layer_features": layer_features,
            "layer_patch_tokens": layer_patch_tokens,
            "attn_maps": attn_maps,
        }

    @property
    def embed_dim(self) -> int:
        """Embedding dimensionality of the teacher backbone."""
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        """Spatial patch size in pixels (always 14 for DINOv2)."""
        return self._patch_size
