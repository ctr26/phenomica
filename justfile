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
