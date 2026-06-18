"""Tests for phenomica config system: pydantic validation, enum-bug, dim-sync."""

from __future__ import annotations

import pytest
import torch
from hydra_zen import builds, instantiate
from hydra_zen.third_party.pydantic import pydantic_parser
from pydantic import ValidationError

from phenomica.configs import ModelConfig, TeacherConfig, TrainingConfig
from phenomica.trainer import DistillationTrainer


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
    model_cfg = instantiate(
        Conf(variant="simple"), _target_wrapper_=pydantic_parser
    )
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
