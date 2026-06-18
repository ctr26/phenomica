# Running Experiments

## Submission Modes

Phenomica supports two SLURM submission modes:

1. **Multirun sweeps** (recommended): Use `--multirun` with `hydra/launcher=submitit_slurm_biohive` for parameter sweeps. The Hydra submitit launcher plugin handles job submission.

2. **Single-job manual submission**: Use `cluster=biohive` without `--multirun`. The manual `cluster.use_submitit` path in `train.py` submits a single job via `submitit.AutoExecutor`.

## Quick Start

### Local Dry-Run Validation

Validate sweep composition without submitting:

```bash
just smoke-check
```

Or directly:

```bash
PYTHONPATH=src python -m phenomica.train \
    --multirun \
    model=simple_resnet18 \
    teacher=dinov2_small \
    training=debug \
    data=imagenette \
    hydra/launcher=submitit_slurm_biohive \
    --cfg job --resolve
```

### Phase 0: Smoke Test (Real Submission)

```bash
just smoke
```

This submits a single-cell sweep to BioHive SLURM:
- Model: `simple_resnet18`
- Teacher: `dinov2_small`
- Training: `debug` (10 epochs, no wandb)
- Data: `imagenette` (small public dataset)

### Phase 1: Objective Sweep

```bash
just sweep-objectives
```

(TODO: Replace placeholder with actual sweep matrix once objectives are defined.)

## Sweep Syntax

Hydra multirun uses comma-separated values for sweeps:

```bash
python -m phenomica.train --multirun \
    model=simple_resnet18,multi_resnet18 \
    teacher=dinov2_small,dinov2_base \
    training.loss_type=mse,cosine,combined \
    hydra/launcher=submitit_slurm_biohive
```

This generates a Cartesian product (2 models × 2 teachers × 3 losses = 12 jobs).

## SLURM Parameters

The `submitit_slurm_biohive` launcher preset maps to:
- **Partition**: `biohive`
- **Account**: `rxrx`
- **GPUs**: 4 per node
- **Nodes**: 1
- **Timeout**: 720 min (12h)
- **Memory**: 64 GB
- **CPUs per task**: 8

Override via `hydra.launcher.*`:

```bash
python -m phenomica.train --multirun \
    ... \
    hydra/launcher=submitit_slurm_biohive \
    hydra.launcher.timeout_min=1440 \
    hydra.launcher.gpus_per_node=8
```

## Output

Sweep outputs land in:
- **Hydra multidir**: `outputs/<date>/<time>/` (one subdir per sweep cell)
- **SLURM logs**: `outputs/<date>/<time>/.submitit/<job_id>/`
