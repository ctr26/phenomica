# Phenomica experiment sweep commands

# Phase 0: Quick smoke test on imagenette (local dry-run validation)
smoke-check:
    PYTHONPATH=src python -m phenomica.train \
        --multirun \
        model=simple_resnet18 \
        teacher=dinov2_small \
        training=debug \
        data=imagenette \
        hydra/launcher=submitit_slurm_biohive \
        --cfg job --resolve

# Phase 0: Quick smoke test on imagenette (real SLURM submission)
smoke:
    PYTHONPATH=src python -m phenomica.train \
        --multirun \
        model=simple_resnet18 \
        teacher=dinov2_small \
        training=debug \
        data=imagenette \
        hydra/launcher=submitit_slurm_biohive

# Phase 1: Full objective sweep (placeholder for actual sweep params)
sweep-objectives:
    @echo "TODO: Replace with actual Phase-1 objective sweep parameters"
    @echo "Example: model=simple_resnet18,multi_resnet18 teacher=dinov2_small,dinov2_base,dinov2_large training.loss_type=mse,cosine,combined"
