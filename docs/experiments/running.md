# Running experiments

## Submission modes
1. **Multirun sweeps** (recommended): `--multirun` + the submitit launcher plugin
   (`hydra/launcher=submitit_slurm`) with BioHive params passed as `hydra.launcher.*`
   overrides — one SLURM job per sweep cell. Encapsulated in the `justfile` `biohive` var.
2. **Single-job manual submission**: `cluster=biohive` without `--multirun` — the manual
   `cluster.use_submitit` path in `train.py` submits one job via `submitit.AutoExecutor`.

## BioHive launcher overrides
Pass these with the plugin launcher (the `justfile` `biohive` variable bundles them):

```
hydra/launcher=submitit_slurm
hydra.launcher.partition=hopper          # biohive GPU partition (commit dc5419c)
hydra.launcher.timeout_min=720
hydra.launcher.gpus_per_node=4
hydra.launcher.tasks_per_node=4
hydra.launcher.cpus_per_task=8
hydra.launcher.mem_gb=64
+hydra.launcher.additional_parameters.wckey=default   # required or SLURM rejects the job
# add  hydra.launcher.account=<acct>  if your cluster requires an account
```

`partition` is **`hopper`**, not `biohive` — `biohive` is the cluster name; `hopper` is
the partition (per commit dc5419c, which fixed the non-existent `h100`).

## Quick start
### Local dry-run (NO submission)
Hydra forbids combining `--multirun` with `--cfg`, so validate a single cell:

```bash
just smoke-check
# == uv run phenomica-train model=simple_resnet18 teacher=dinov2_small \
#       training=debug data=imagenette --cfg job --resolve
```

Inspect the resolved launcher (single, no `--multirun`):

```bash
uv run phenomica-train model=simple_resnet18 teacher=dinov2_small training=debug \
    data=imagenette hydra/launcher=submitit_slurm hydra.launcher.partition=hopper \
    +hydra.launcher.additional_parameters.wckey=default --cfg hydra --resolve
```

### Phase 0: smoke (real submission)
```bash
just smoke
```
Submits one cell — `simple_resnet18 × dinov2_small`, `training=debug`, `data=imagenette` —
to BioHive SLURM.

### Phase 1: objective ablation
```bash
just sweep-objectives
```
`simple_resnet18 × dinov2_base` over `training.loss_type=mse,cosine,combined`; Hydra
multirun makes the Cartesian product (one SLURM job per cell).

## Sweep syntax
Comma-separated values expand to a Cartesian product:

```bash
uv run phenomica-train --multirun \
    model=simple_resnet18,multi_resnet18 teacher=dinov2_small,dinov2_base \
    training.loss_type=mse,cosine,combined data=imagenette {{biohive overrides}}
# 2 models × 2 teachers × 3 losses = 12 SLURM jobs
```

## Output
- Hydra multirun dir: `multirun/<date>/<time>/` (one subdir per cell)
- SLURM logs: `<sweep.dir>/.submitit/<job_id>/`
