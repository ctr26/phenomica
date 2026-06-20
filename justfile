# Phenomica experiment sweep commands

# BioHive submitit launcher overrides. partition=hopper is the biohive GPU partition
# (commit dc5419c fixed the non-existent "h100"); wckey=default is required or SLURM
# rejects the job (commits 15a1bc2/4830dd8). Add hydra.launcher.account=<acct> if needed.
biohive := "hydra/launcher=submitit_slurm hydra.launcher.partition=hopper hydra.launcher.timeout_min=720 hydra.launcher.gpus_per_node=4 hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=64 +hydra.launcher.additional_parameters.wckey=default"

# Phase 0: validate the smoke cell composes (local dry-run, NO submission).
# Note: Hydra forbids combining --multirun with --cfg, so the dry-run is single-cell.
smoke-check:
    uv run phenomica-train \
        model=simple_resnet18 teacher=dinov2_small training=debug data=imagenette \
        --cfg job --resolve

# Phase 0: submit the smoke cell to BioHive SLURM.
smoke:
    uv run phenomica-train --multirun \
        model=simple_resnet18 teacher=dinov2_small training=debug data=imagenette \
        {{biohive}}

# Phase 1: objective ablation -- fix resnet18 x dinov2_base, sweep the loss variants.
sweep-objectives:
    uv run phenomica-train --multirun \
        model=simple_resnet18 teacher=dinov2_base data=imagenette \
        training.loss_type=mse,cosine,combined \
        {{biohive}}

# ===========================================================================
# Ray launch path (INDEPENDENT from the submitit targets above). Ray Train +
# Ray Tune run on a Ray cluster, or locally with a small in-process runtime --
# the targets below are CPU-only local smoke runs (use_gpu=False, num_workers=1),
# no SLURM, no submitit. Point data.root at a small ImageFolder first.
# ===========================================================================

# Ray Train: single CPU-local distributed run (1 worker, no GPU, no wandb).
ray-train-local:
    uv run phenomica-ray-train \
        model=simple_resnet18 teacher=dinov2_small \
        data=imagenette training=debug ray_train=local_cpu ray_data=default

# Ray Tune: CPU-local ASHA sweep (debug preset: 2 trials, 1 concurrent, no wandb).
ray-tune-local:
    uv run phenomica-ray-tune \
        model=simple_resnet18 teacher=dinov2_small \
        data=imagenette training=debug ray_train=local_cpu ray_tune=debug ray_data=default
