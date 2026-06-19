"""Frozen DINOv2 teacher for distillation targets.

Wraps a pretrained DINOv2 vision transformer loaded from torch.hub,
freezing all parameters and extracting intermediate layer features
for multi-level distillation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _compute_cls_attention_from_hidden_states(
    x: torch.Tensor,
    attn_module: nn.Module,
) -> torch.Tensor:
    """Recompute CLS-token attention from hidden states entering an attention module.

    DINOv2 attention modules use fused SDPA/xformers that don't return weights,
    so we recompute CLS-token attention from q/k. This is a pure helper function
    for unit testing.

    Args:
        x: Hidden state [B, T, C] entering the attention module.
        attn_module: The attention module (must have `.qkv`, `.num_heads`,
            and optionally `.scale`).

    Returns:
        CLS-token attention weights [B, num_heads, T] with each row summing to ~1.

    Raises:
        AttributeError: If the module lacks required attributes.
    """
    B, T, C = x.shape

    # Defensive attribute access: different DINOv2/timm attention implementations
    # may vary in their exact API.
    if not hasattr(attn_module, "qkv"):
        raise AttributeError(
            f"Attention module {type(attn_module).__name__} lacks 'qkv' attribute; "
            f"cannot recompute attention weights"
        )

    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        raise AttributeError(
            f"Attention module {type(attn_module).__name__} lacks 'num_heads' attribute"
        )

    head_dim = C // num_heads
    scale = getattr(attn_module, "scale", head_dim**-0.5)

    # Compute qkv via the attention module's qkv linear layer.
    # qkv shape: [B, T, 3 * C] -> reshape to [B, T, 3, num_heads, head_dim].
    qkv = attn_module.qkv(x)
    qkv = qkv.reshape(B, T, 3, num_heads, head_dim)

    # Permute to [3, B, num_heads, T, head_dim] and split.
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]  # each [B, num_heads, T, head_dim]

    # Compute attention scores: q @ k^T / sqrt(head_dim).
    attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]

    # Softmax over key dimension to get attention weights.
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, T, T]

    # Extract CLS-query row (token 0) -> [B, num_heads, T].
    cls_attn = attn_weights[:, :, 0, :]

    return cls_attn


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

        # Register hooks to extract CLS-token attention if requested.
        # DINOv2/timm uses fused SDPA/xformers that doesn't return weights,
        # so we recompute them from q/k via a forward hook.
        if extract_attention:
            for layer_idx in self._extract_layers:
                block = self._model.blocks[layer_idx]
                block.attn.register_forward_hook(self._make_attn_hook(layer_idx))

    def _make_attn_hook(self, layer_idx: int):
        """Create a forward hook to extract CLS-token attention from a block.

        DINOv2 attention modules use fused SDPA/xformers that don't return weights,
        so we recompute CLS-token attention from q/k via the helper function
        `_compute_cls_attention_from_hidden_states`.

        Args:
            layer_idx: Block index for debugging/logging.

        Returns:
            Hook function compatible with register_forward_hook.
        """

        def hook(module, inputs, outputs):
            try:
                # inputs[0] is the hidden state [B, T, C] entering the attention module.
                x = inputs[0]
                cls_attn = _compute_cls_attention_from_hidden_states(x, module)
                self._attn_weights.append(cls_attn)

            except Exception as e:
                # Defensive: log warning and skip on any failure.
                logger.warning(
                    f"DINOv2Teacher attention extraction failed at layer {layer_idx}: {e}"
                )

        return hook

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

        # Loud-failure: warn if extraction was enabled but produced nothing.
        if self._extract_attention and (attn_maps is None or len(attn_maps) == 0):
            logger.warning(
                "DINOv2Teacher: extract_attention=True but no attention maps were extracted. "
                "Check that hooks registered successfully and that the model forward succeeded."
            )

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
