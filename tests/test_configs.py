"""Tests for phenomica config system: pydantic validation, enum-bug, dim-sync."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import builds, instantiate
from hydra_zen.third_party.pydantic import pydantic_parser
from pydantic import ValidationError

import phenomica.train  # noqa: F401 -- registers top-level config + groups
from phenomica.configs import ModelConfig, TeacherConfig, TrainingConfig
from phenomica.trainer import DistillationTrainer

_PRESETS = [
    (group, name)
    for group, names in {
        "model": [
            "simple_resnet18",
            "simple_effnet",
            "simple_vit_tiny",
            "multi_resnet18",
            "multi_effnet",
        ],
        "teacher": ["dinov2_base", "dinov2_base_attn", "dinov2_small", "dinov2_large"],
        "data": ["imagenet", "custom"],
        "training": ["default", "debug"],
        "cluster": ["local", "biohive"],
    }.items()
    for name in names
]


def test_enum_bug_regression_guard():
    """Enum-bug regression: variant default serializes as plain str, not Enum repr."""
    # Compose default config and check variant is plain string
    cfg = ModelConfig()
    assert cfg.variant == "simple"
    assert isinstance(cfg.variant, str)
    assert cfg.variant != "ModelVariant.SIMPLE"

    # Instantiate via pydantic_parser and confirm it works
    Conf = builds(ModelConfig, populate_full_signature=True)
    model_cfg = instantiate(Conf, _target_wrapper_=pydantic_parser)
    assert model_cfg.variant == "simple"


def test_pydantic_validation_literal():
    """Pydantic validation rejects invalid Literal values."""
    Conf = builds(ModelConfig, populate_full_signature=True)

    # Valid literal -> OK
    model_cfg = instantiate(Conf(variant="simple"), _target_wrapper_=pydantic_parser)
    assert model_cfg.variant == "simple"

    # Invalid literal -> raises
    # instantiate wraps pydantic errors in InstantiationException
    with pytest.raises((ValidationError, Exception)):
        instantiate(Conf(variant="bogus"), _target_wrapper_=pydantic_parser)


def test_pydantic_validation_positive_int():
    """Pydantic validation accepts positive ints, rejects non-positive."""
    Conf = builds(TrainingConfig, populate_full_signature=True)

    # Valid positive int -> OK
    cfg = instantiate(Conf(epochs=10), _target_wrapper_=pydantic_parser)
    assert cfg.epochs == 10

    # Invalid (zero) -> raises
    with pytest.raises((ValidationError, Exception)):
        instantiate(Conf(epochs=0), _target_wrapper_=pydantic_parser)


def test_enum_bug_trainer_builds():
    """Enum-bug regression: trainer builds with default variant='simple'."""
    # Fake data config (no real data needed)
    from types import SimpleNamespace

    data_cfg = SimpleNamespace(
        dataset_type="imagefolder",
        root="data/dummy",
        image_size=224,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.1,
    )

    trainer = DistillationTrainer(
        training_cfg=TrainingConfig(
            use_wandb=False, use_ddp=False, warmup_epochs=0, lr_scheduler=None
        ),
        model_cfg=ModelConfig(variant="simple", pretrained_backbone=False),
        teacher_cfg=TeacherConfig(),
        data_cfg=data_cfg,
    )

    # Trainer should build without crashing
    assert trainer is not None
    assert trainer.model is not None


def test_dim_coupling_guard_1024():
    """Dim-coupling guard: student heads auto-sync to teacher embed_dim."""
    from types import SimpleNamespace

    data_cfg = SimpleNamespace(
        dataset_type="imagefolder",
        root="data/dummy",
        image_size=224,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.1,
    )

    model_cfg = ModelConfig(variant="simple", pretrained_backbone=False)
    teacher_cfg = TeacherConfig(model_name="dinov2_vitl14", embed_dim=1024)

    trainer = DistillationTrainer(
        training_cfg=TrainingConfig(
            use_wandb=False, use_ddp=False, warmup_epochs=0, lr_scheduler=None
        ),
        model_cfg=model_cfg,
        teacher_cfg=teacher_cfg,
        data_cfg=data_cfg,
    )

    # Student model's projection should be synced to 1024
    x = torch.randn(1, 3, 224, 224)
    out = trainer.model(x)
    assert out.shape == (1, 1024)


def test_dim_mismatch_raises():
    """Dim-coupling guard: mismatch between teacher_cfg.embed_dim and loaded .embed_dim raises."""
    from types import SimpleNamespace

    data_cfg = SimpleNamespace(
        dataset_type="imagefolder",
        root="data/dummy",
        image_size=224,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.1,
    )

    # Fake teacher with embed_dim=1024, but config claims 768
    model_cfg = ModelConfig(variant="simple", pretrained_backbone=False)
    teacher_cfg = TeacherConfig(model_name="dinov2_vitl14", embed_dim=768)  # Wrong!

    with pytest.raises(ValueError, match="disagrees with loaded"):
        DistillationTrainer(
            training_cfg=TrainingConfig(
                use_wandb=False, use_ddp=False, warmup_epochs=0, lr_scheduler=None
            ),
            model_cfg=model_cfg,
            teacher_cfg=teacher_cfg,
            data_cfg=data_cfg,
        )


@pytest.mark.parametrize("group,name", _PRESETS)
def test_all_presets_instantiate(group: str, name: str) -> None:
    """Every registered preset composes and passes pydantic validation.

    Guards against over-strict field constraints (e.g. a ``PositiveInt`` on a
    field a preset legitimately sets to 0, such as ``training=debug`` with
    ``weight_decay=0``/``warmup_epochs=0``) that the default-config tests do
    not exercise.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=None):
        cfg = compose(config_name="phenomica", overrides=[f"{group}={name}"])
    instantiate(getattr(cfg, group), _target_wrapper_=pydantic_parser)


def test_multifunction_variant_trains() -> None:
    """Multifunction variant builds and runs a train step end-to-end.

    Exercises the multi-head model, ``extract_layers`` wiring, and
    ``MultiFunctionDistillationLoss`` path that the schema-only tests skip.
    """
    import math
    from types import SimpleNamespace

    from torch.utils.data import DataLoader, TensorDataset

    data_cfg = SimpleNamespace(
        dataset_type="imagefolder",
        root="data/dummy",
        image_size=224,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.1,
    )
    trainer = DistillationTrainer(
        training_cfg=TrainingConfig(
            use_wandb=False, use_ddp=False, warmup_epochs=0, lr_scheduler=None
        ),
        model_cfg=ModelConfig(variant="multifunction", pretrained_backbone=False),
        teacher_cfg=TeacherConfig(),
        data_cfg=data_cfg,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    avg_loss = trainer.train_epoch(loader)
    assert math.isfinite(avg_loss)


def test_dinov2_base_attn_preset_resolves():
    """Teacher preset dinov2_base_attn sets extract_attention=True."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=None):
        cfg = compose(config_name="phenomica", overrides=["teacher=dinov2_base_attn"])
    teacher_cfg = instantiate(cfg.teacher, _target_wrapper_=pydantic_parser)
    assert teacher_cfg.extract_attention is True


def test_attndistill_receives_real_attention():
    """AttnDistill loss receives non-None attn_maps when extract_attention=True."""
    from types import SimpleNamespace

    from torch.utils.data import DataLoader, TensorDataset

    data_cfg = SimpleNamespace(
        dataset_type="imagefolder",
        root="data/dummy",
        image_size=224,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.1,
    )
    trainer = DistillationTrainer(
        training_cfg=TrainingConfig(
            use_wandb=False,
            use_ddp=False,
            warmup_epochs=0,
            lr_scheduler=None,
            loss_type="attndistill",
        ),
        model_cfg=ModelConfig(variant="simple", pretrained_backbone=False),
        teacher_cfg=TeacherConfig(extract_attention=True),
        data_cfg=data_cfg,
    )

    # Run one forward/criterion call
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    batch = next(iter(loader))
    images, _ = batch
    student_out = trainer.model(images)
    with torch.no_grad():
        teacher_out = trainer.teacher(images)

    loss = trainer.criterion(student_out, teacher_out)

    # Verify attn_maps is non-None and real attention path was used
    assert "attn_maps" in teacher_out
    assert teacher_out["attn_maps"] is not None
    assert trainer.criterion._last_loss_metrics["attndistill_attn"] > 0.0
    assert loss.item() > 0.0
