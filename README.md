# Phenomica

General-purpose vision feature extraction via knowledge distillation from DINOv2.

Train lightweight student models (ResNet, EfficientNet, ViT) that learn to reproduce DINOv2's representations — getting strong visual features without the cost of running a large ViT at inference.

## Variants

| Variant | Description | Output |
|---------|-------------|--------|
| **Simple** | Single projection head → CLS token | `[B, projection_dim]` |
| **Multi-function** | Multiple heads (global, spatial, scale) | Dict of feature tensors |

## Quick Start

```bash
# Install
uv sync

# Debug training (no wandb, 10 epochs)
uv run phenomica-train training=debug data.root=path/to/imagefolder

# Full training with multi-function model
uv run phenomica-train model=multi_resnet18 teacher=dinov2_large

# Submit to SLURM cluster
uv run phenomica-train model=multi_resnet18 cluster=biohive
```

## Config Presets

Override any group via CLI:

| Group | Presets |
|-------|---------|
| `model` | `simple_resnet18`, `simple_effnet`, `simple_vit_tiny`, `multi_resnet18`, `multi_effnet` |
| `teacher` | `dinov2_small` (384d), `dinov2_base` (768d), `dinov2_large` (1024d) |
| `data` | `imagenet`, `custom` |
| `training` | `default` (100 epochs, cosine LR, wandb), `debug` (10 epochs, no wandb) |
| `cluster` | `local`, `biohive` (SLURM/H100s via submitit) |

## Docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [GUIDELINES.md](GUIDELINES.md) — Workflows and processes
- [STANDARDS.md](STANDARDS.md) — Code standards
- [HUMANS.md](HUMANS.md) — Documentation system overview
