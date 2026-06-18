"""Test fixtures for phenomica tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class FakeDINOv2Teacher(nn.Module):
    """CPU-only fake teacher for testing without network downloads."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        extract_layers=None,
        extract_attention: bool = False,
    ):
        super().__init__()
        # Map model names to embed dims
        embed_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
        }
        self._embed_dim = embed_dims.get(model_name, 768)
        self._patch_size = 14
        self._extract_layers = extract_layers or [11]
        self._extract_attention = extract_attention

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Return teacher outputs matching DINOv2Teacher contract."""
        B = x.size(0)
        D = self._embed_dim
        N = 256  # Number of patches for 224x224 / 14x14
        num_heads = 12  # Standard for vitb14
        num_layers = len(self._extract_layers)

        return {
            "cls": torch.randn(B, D),
            "patch_tokens": torch.randn(B, N, D),
            "patch_stats": torch.randn(B, D * 2),
            "layer_features": [torch.randn(B, D) for _ in range(num_layers)],
            "layer_patch_tokens": [torch.randn(B, N, D) for _ in range(num_layers)],
            "attn_maps": (
                [torch.randn(B, num_heads, N) for _ in range(num_layers)]
                if self._extract_attention
                else None
            ),
        }


@pytest.fixture(autouse=True)
def monkeypatch_dinov2_teacher(monkeypatch):
    """Monkeypatch DINOv2Teacher in all modules to avoid network calls."""
    import phenomica.teacher
    import phenomica.trainer

    monkeypatch.setattr(phenomica.teacher, "DINOv2Teacher", FakeDINOv2Teacher)
    monkeypatch.setattr(phenomica.trainer, "DINOv2Teacher", FakeDINOv2Teacher)
