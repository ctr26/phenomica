# Phenomica

General-purpose vision feature extraction via knowledge distillation from DINOv2.

Train lightweight student models (ResNet, EfficientNet, ViT) that learn to reproduce DINOv2's representations — getting strong visual features without the cost of running a large ViT at inference.

## Variants

| Variant | Description | Output |
|---------|-------------|--------|
| **Simple** | Single projection head → CLS token | `[B, projection_dim]` |
| **Multi-function** | Multiple heads (global, spatial, scale) | Dict of feature tensors |

## Two independent launch paths

Phenomica ships **two non-overlapping ways to run** the same distillation
building blocks (model / loss / teacher / eval / configs). Pick one; they never
import each other.

| | submitit path | Ray path |
|---|---|---|
| Entry module | `phenomica.train` | `phenomica.ray_launch` |
| Console scripts | `phenomica-train` | `phenomica-ray-train`, `phenomica-ray-tune` |
| Data | torch `DataLoader` | Ray Data `Dataset` |
| Distribution | torch DDP + `DistillationTrainer` | Ray Train `TorchTrainer` + `ScalingConfig` |
| Sweeps | Hydra `--multirun` + submitit | Ray Tune `Tuner` + `ASHAScheduler` |
| Scheduler | SLURM (BioHive `hopper`) | Ray cluster (or local) |

See [docs/experiments/scaffold.md](docs/experiments/scaffold.md) for the Ray
path and [docs/experiments/running.md](docs/experiments/running.md) for submitit
sweeps.

## Quick Start

```bash
uv sync   # install (uv-managed; Python deps incl. torch + ray[train,tune,data])
```

### submitit / Hydra path (`phenomica-train`)

```bash
# Debug training (no wandb, 10 epochs)
uv run phenomica-train training=debug data.root=path/to/imagefolder

# Full training with multi-function model
uv run phenomica-train model=multi_resnet18 teacher=dinov2_large

# Submit to SLURM cluster (single job; sweeps use --multirun, see running.md)
uv run phenomica-train model=multi_resnet18 cluster=biohive
```

### Ray path (`phenomica-ray-train` / `phenomica-ray-tune`)

```bash
# Ray Train: single distributed run (CPU-local smoke)
uv run phenomica-ray-train model=simple_resnet18 teacher=dinov2_small \
    data=imagenette training=debug ray_train=local_cpu ray_data=default

# Ray Tune: ASHA hyperparameter sweep (CPU-local)
uv run phenomica-ray-tune model=simple_resnet18 teacher=dinov2_small \
    data=imagenette training=debug ray_train=local_cpu ray_tune=debug ray_data=default
```

### justfile shortcuts

```bash
just ray-train-local    # CPU-local Ray Train smoke
just ray-tune-local     # CPU-local Ray Tune sweep
just smoke              # submit the submitit smoke cell to SLURM
just sweep-objectives   # submitit --multirun loss-type ablation
```

### Tests

```bash
uv run pytest -q                 # full suite (Ray tests run CPU-only/local)
uv run pytest -q -m "not slow"   # skip tests that spin up a local Ray runtime
```

## Config Presets

Override any group via CLI:

| Group | Presets |
|-------|---------|
| `model` | `simple_resnet18`, `simple_effnet`, `simple_vit_tiny`, `multi_resnet18`, `multi_effnet` |
| `teacher` | `dinov2_small` (384d), `dinov2_base` (768d), `dinov2_large` (1024d) |
| `data` | `imagenet`, `imagenette`, `custom` |
| `training` | `default` (100 epochs, cosine LR, wandb), `debug` (10 epochs, no wandb) |
| `cluster` | `local`, `biohive` (SLURM `hopper` partition via submitit) |
| `ray_data` | `default` |
| `ray_train` | `local_cpu` (1 worker, CPU), `biohive_gpu` (4 workers, GPU) |
| `ray_tune` | `default` (4 trials), `debug` (2 trials) |

## Docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [GUIDELINES.md](GUIDELINES.md) — Workflows and processes
- [STANDARDS.md](STANDARDS.md) — Code standards
- [HUMANS.md](HUMANS.md) — Documentation system overview
