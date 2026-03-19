"""Hydra-Zen structured configs for phenomica.

All configuration is defined here as Python dataclasses via hydra-zen.
No YAML files needed. Importing this module registers all store entries.
"""

from hydra_zen import make_config, store

# -- Model configs -----------------------------------------------------------

model_store = store(group="model")

model_store(
    make_config(
        variant="simple",
        backbone="resnet18",
        projection_dim=768,
        pretrained_backbone=True,
    ),
    name="simple_resnet18",
)

model_store(
    make_config(
        variant="simple",
        backbone="efficientnet_b0",
        projection_dim=768,
        pretrained_backbone=True,
    ),
    name="simple_effnet",
)

model_store(
    make_config(
        variant="simple",
        backbone="vit_tiny_patch16_224",
        projection_dim=768,
        pretrained_backbone=True,
    ),
    name="simple_vit_tiny",
)

model_store(
    make_config(
        variant="multifunction",
        backbone="resnet18",
        teacher_cls_dim=768,
        teacher_patch_dim=768,
        teacher_layers=[3, 6, 9, 11],
        pretrained_backbone=True,
    ),
    name="multi_resnet18",
)

model_store(
    make_config(
        variant="multifunction",
        backbone="efficientnet_b0",
        teacher_cls_dim=768,
        teacher_patch_dim=768,
        teacher_layers=[3, 6, 9, 11],
        pretrained_backbone=True,
    ),
    name="multi_effnet",
)

# -- Teacher configs ---------------------------------------------------------

teacher_store = store(group="teacher")

teacher_store(
    make_config(model_name="dinov2_vits14", embed_dim=384),
    name="dinov2_small",
)

teacher_store(
    make_config(model_name="dinov2_vitb14", embed_dim=768),
    name="dinov2_base",
)

teacher_store(
    make_config(model_name="dinov2_vitl14", embed_dim=1024),
    name="dinov2_large",
)

# -- Data configs ------------------------------------------------------------

data_store = store(group="data")

data_store(
    make_config(
        dataset_type="imagefolder",
        root="data/imagenet",
        image_size=224,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        val_split=0.1,
    ),
    name="imagenet",
)

data_store(
    make_config(
        dataset_type="imagefolder",
        root="data/custom",
        image_size=224,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        val_split=0.1,
    ),
    name="custom",
)

# -- Training configs --------------------------------------------------------

training_store = store(group="training")

training_store(
    make_config(
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=5,
        warmup_start_factor=1e-4,
        gradient_clip=1.0,
        loss_type="mse",
        cosine_weight=1.0,
        mse_weight=1.0,
        use_wandb=True,
        wandb_project="phenomica",
        wandb_run_name=None,
        wandb_tags=None,
        seed=42,
        use_ddp=True,
        early_stopping_patience=20,
        validation_freq=1,
        eval_freq=10,
    ),
    name="default",
)

training_store(
    make_config(
        epochs=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        lr_scheduler=None,
        warmup_epochs=0,
        warmup_start_factor=1e-4,
        gradient_clip=None,
        loss_type="mse",
        cosine_weight=1.0,
        mse_weight=1.0,
        use_wandb=False,
        wandb_project="phenomica",
        wandb_run_name=None,
        wandb_tags=None,
        seed=42,
        use_ddp=False,
        early_stopping_patience=None,
        validation_freq=1,
        eval_freq=5,
    ),
    name="debug",
)

# -- Cluster configs ---------------------------------------------------------

cluster_store = store(group="cluster")

cluster_store(
    make_config(use_submitit=False),
    name="local",
)

cluster_store(
    make_config(
        use_submitit=True,
        partition="h100",
        gpus_per_node=4,
        nodes=1,
        timeout_min=720,
        mem_gb=64,
        cpus_per_task=8,
        slurm_account=None,
        log_dir="slurm_logs",
    ),
    name="biohive",
)
