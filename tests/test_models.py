"""Smoke tests for phenomica models and losses."""

import torch


def test_simple_distiller_forward():
    from phenomica.models import SimpleDistiller

    model = SimpleDistiller(backbone="resnet18", projection_dim=768, pretrained_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 768)


def test_simple_distiller_extract_features():
    from phenomica.models import SimpleDistiller

    model = SimpleDistiller(backbone="resnet18", projection_dim=768, pretrained_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    feat = model.extract_features(x)
    assert feat.shape == (2, 512)  # resnet18 has 512-dim features


def test_multifunction_distiller_forward():
    from phenomica.models import MultiFunctionDistiller

    model = MultiFunctionDistiller(
        backbone="resnet18",
        teacher_cls_dim=768,
        teacher_patch_dim=768,
        teacher_layers=[3, 6, 9, 11],
        pretrained_backbone=False,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out["global"].shape == (2, 768)
    assert out["spatial"].shape == (2, 768 * 2)
    assert len(out["scale"]) == 4
    assert out["scale"][0].shape == (2, 768)


def test_multifunction_extract_features():
    from phenomica.models import MultiFunctionDistiller

    model = MultiFunctionDistiller(
        backbone="resnet18",
        teacher_cls_dim=768,
        teacher_patch_dim=768,
        teacher_layers=[3, 6, 9, 11],
        pretrained_backbone=False,
    )
    x = torch.randn(2, 3, 224, 224)
    feat = model.extract_features(x)
    # 768 (global) + 768*2 (spatial) + 4*768 (scale) = 768 + 1536 + 3072 = 5376
    assert feat.shape == (2, 5376)


def test_build_model_simple():
    from types import SimpleNamespace

    from phenomica.models import build_model

    cfg = SimpleNamespace(
        variant="simple", backbone="resnet18",
        projection_dim=384, pretrained_backbone=False,
    )
    model = build_model(cfg)
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 384)


def test_build_model_multifunction():
    from types import SimpleNamespace

    from phenomica.models import build_model

    cfg = SimpleNamespace(
        variant="multifunction", backbone="resnet18",
        teacher_cls_dim=384, teacher_patch_dim=384,
        teacher_layers=[3, 6], pretrained_backbone=False,
    )
    model = build_model(cfg)
    out = model(torch.randn(1, 3, 224, 224))
    assert "global" in out and "spatial" in out and "scale" in out


def test_distillation_loss_mse():
    from phenomica.losses import DistillationLoss

    loss_fn = DistillationLoss(loss_type="mse")
    student = torch.randn(4, 768)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)
    assert loss.shape == ()
    assert "mse" in loss_fn._last_loss_metrics


def test_distillation_loss_cosine():
    from phenomica.losses import DistillationLoss

    loss_fn = DistillationLoss(loss_type="cosine")
    student = torch.randn(4, 768)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)
    assert loss.shape == ()
    assert loss_fn._last_loss_metrics["cosine"] >= 0


def test_distillation_loss_combined():
    from phenomica.losses import DistillationLoss

    loss_fn = DistillationLoss(loss_type="combined", mse_weight=0.5, cosine_weight=0.5)
    student = torch.randn(4, 768)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)
    assert loss.shape == ()


def test_multifunction_loss():
    from phenomica.losses import MultiFunctionDistillationLoss

    loss_fn = MultiFunctionDistillationLoss(loss_type="mse")
    student_outputs = {
        "global": torch.randn(4, 768),
        "spatial": torch.randn(4, 768 * 2),
        "scale": [torch.randn(4, 768) for _ in range(4)],
    }
    teacher_outputs = {
        "cls": torch.randn(4, 768),
        "patch_stats": torch.randn(4, 768 * 2),
        "layer_features": [torch.randn(4, 768) for _ in range(4)],
    }
    loss = loss_fn(student_outputs, teacher_outputs)
    assert loss.shape == ()
    assert "global" in loss_fn._last_loss_metrics
    assert "spatial" in loss_fn._last_loss_metrics
    assert "scale" in loss_fn._last_loss_metrics


def test_data_transforms():
    from phenomica.data import get_transforms

    train_t = get_transforms(224, is_train=True)
    val_t = get_transforms(224, is_train=False)
    assert train_t is not None
    assert val_t is not None
