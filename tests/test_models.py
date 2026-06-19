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
        variant="simple",
        backbone="resnet18",
        projection_dim=384,
        pretrained_backbone=False,
    )
    model = build_model(cfg)
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 384)


def test_build_model_multifunction():
    from types import SimpleNamespace

    from phenomica.models import build_model

    cfg = SimpleNamespace(
        variant="multifunction",
        backbone="resnet18",
        teacher_cls_dim=384,
        teacher_patch_dim=384,
        teacher_layers=[3, 6],
        pretrained_backbone=False,
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


def test_loss_registry_decorator():
    """Test loss registry mechanism."""
    import torch.nn as nn

    from phenomica.losses import LOSS_REGISTRY, register_loss

    # Create a dummy loss class
    @register_loss("test_dummy_loss")
    class DummyLoss(nn.Module):
        def forward(self, student_output, teacher_outputs):
            return torch.tensor(0.0)

    assert "test_dummy_loss" in LOSS_REGISTRY
    assert LOSS_REGISTRY["test_dummy_loss"] is DummyLoss


def test_build_loss_existing_types():
    """Test build_loss returns correct classes for existing loss types."""
    from phenomica.losses import DistillationLoss, build_loss

    # Test simple distillation losses
    loss_mse = build_loss("mse")
    assert isinstance(loss_mse, DistillationLoss)
    assert loss_mse.loss_type == "mse"

    loss_cosine = build_loss("cosine")
    assert isinstance(loss_cosine, DistillationLoss)
    assert loss_cosine.loss_type == "cosine"

    loss_combined = build_loss("combined", mse_weight=0.5, cosine_weight=0.5)
    assert isinstance(loss_combined, DistillationLoss)
    assert loss_combined.loss_type == "combined"


def test_build_loss_unknown_type():
    """Test build_loss raises ValueError for unknown loss types."""
    from phenomica.losses import build_loss

    try:
        build_loss("unknown_loss_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown_loss_type" in str(e).lower()
        assert "known types" in str(e).lower() or "valid" in str(e).lower()


def test_build_loss_registered_type():
    """Test build_loss can construct registered losses."""
    import torch.nn as nn

    from phenomica.losses import build_loss, register_loss

    @register_loss("test_registered")
    class TestRegistered(nn.Module):
        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, student_output, teacher_outputs):
            return torch.tensor(0.0)

    loss = build_loss("test_registered", alpha=2.0)
    assert isinstance(loss, TestRegistered)
    assert loss.alpha == 2.0


def test_teacher_layer_patch_tokens():
    """Test teacher outputs include layer_patch_tokens with correct shapes."""
    from phenomica.teacher import DINOv2Teacher

    # Use monkeypatched FakeDINOv2Teacher
    teacher = DINOv2Teacher()
    x = torch.randn(2, 3, 224, 224)
    outputs = teacher(x)

    assert "layer_patch_tokens" in outputs
    layer_patch_tokens = outputs["layer_patch_tokens"]
    assert isinstance(layer_patch_tokens, list)
    assert len(layer_patch_tokens) > 0
    # Each should be [B, N, D]
    for layer_tokens in layer_patch_tokens:
        assert layer_tokens.ndim == 3
        assert layer_tokens.size(0) == 2  # batch
        assert layer_tokens.size(2) == teacher.embed_dim


def test_teacher_attention_maps_disabled():
    """Test teacher doesn't extract attention maps when extract_attention=False."""
    from phenomica.teacher import DINOv2Teacher

    teacher = DINOv2Teacher(extract_attention=False)
    x = torch.randn(2, 3, 224, 224)
    outputs = teacher(x)

    # Should have None or omit attn_maps when disabled
    attn_maps = outputs.get("attn_maps")
    assert attn_maps is None or "attn_maps" not in outputs


def test_fake_teacher_extended_outputs():
    """Test FakeDINOv2Teacher provides layer_patch_tokens and attn_maps."""
    from tests.conftest import FakeDINOv2Teacher

    # Test with extract_attention=True
    teacher = FakeDINOv2Teacher(extract_attention=True)
    x = torch.randn(2, 3, 224, 224)
    outputs = teacher(x)

    # Check layer_patch_tokens
    assert "layer_patch_tokens" in outputs
    layer_patch_tokens = outputs["layer_patch_tokens"]
    assert isinstance(layer_patch_tokens, list)
    assert len(layer_patch_tokens) == len(teacher._extract_layers)
    for layer_tokens in layer_patch_tokens:
        assert layer_tokens.shape[0] == 2  # batch
        assert layer_tokens.shape[1] == 256  # N patches
        assert layer_tokens.shape[2] == teacher.embed_dim

    # Check attn_maps when enabled
    assert "attn_maps" in outputs
    attn_maps = outputs["attn_maps"]
    assert isinstance(attn_maps, list)
    assert len(attn_maps) == len(teacher._extract_layers)
    for attn_map in attn_maps:
        assert attn_map.shape[0] == 2  # batch
        # num_heads (typically 12 for vitb14)
        assert attn_map.ndim == 3
        assert attn_map.shape[2] == 256  # N patches


def test_build_loss_filters_kwargs():
    """Test build_loss filters kwargs to target constructor signature."""
    import torch.nn as nn

    from phenomica.losses import build_loss, register_loss

    # Register a loss with custom params
    @register_loss("test_filtered")
    class FilteredLoss(nn.Module):
        def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha

        def forward(self, student_output, teacher_outputs):
            return torch.tensor(0.0)

    # Pass superset including unrelated config fields
    superset_kwargs = {
        "temperature": 2.0,
        "alpha": 0.8,
        "epochs": 100,  # unrelated
        "mse_weight": 1.0,  # unrelated
        "learning_rate": 0.001,  # unrelated
    }

    loss = build_loss("test_filtered", **superset_kwargs)
    assert isinstance(loss, FilteredLoss)
    assert loss.temperature == 2.0
    assert loss.alpha == 0.8


def test_build_loss_existing_types_with_superset():
    """Test existing loss types still work with superset kwargs."""
    from phenomica.losses import DistillationLoss, build_loss

    # Pass superset including fields for all loss types
    superset_kwargs = {
        "mse_weight": 0.7,
        "cosine_weight": 0.3,
        "cospress_weight": 1.0,  # unrelated to DistillationLoss
        "vitkd_weight": 1.0,  # unrelated
        "epochs": 50,  # unrelated
    }

    loss_mse = build_loss("mse", **superset_kwargs)
    assert isinstance(loss_mse, DistillationLoss)
    assert loss_mse.loss_type == "mse"
    assert loss_mse.mse_weight == 0.7

    loss_combined = build_loss("combined", **superset_kwargs)
    assert isinstance(loss_combined, DistillationLoss)
    assert loss_combined.mse_weight == 0.7
    assert loss_combined.cosine_weight == 0.3


def test_optimizer_includes_criterion_parameters():
    """Test optimizer includes both student and criterion parameters."""
    from types import SimpleNamespace

    import torch.nn as nn

    from phenomica.losses import register_loss
    from phenomica.trainer import DistillationTrainer

    # Register a parametric loss
    @register_loss("test_parametric")
    class ParametricLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(768, 384)  # learnable params

        def forward(self, student_output, teacher_outputs):
            return torch.tensor(0.0)

    # Build trainer with parametric loss
    training_cfg = SimpleNamespace(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler=None,
        gradient_clip=None,
        loss_type="test_parametric",
        use_wandb=False,
        seed=42,
        use_ddp=False,
    )
    model_cfg = SimpleNamespace(
        variant="simple",
        backbone="resnet18",
        projection_dim=768,
        pretrained_backbone=False,
    )
    teacher_cfg = SimpleNamespace(model_name="dinov2_vitb14", embed_dim=768)
    data_cfg = SimpleNamespace(dataset="imagenet", img_size=224)

    trainer = DistillationTrainer(training_cfg, model_cfg, teacher_cfg, data_cfg)

    # Check optimizer includes criterion params
    optimizer = trainer.optimizer
    all_params = []
    for group in optimizer.param_groups:
        all_params.extend(group["params"])

    # Find the projection layer's weight param
    criterion_weight = trainer.criterion.projection.weight
    assert any(p is criterion_weight for p in all_params), (
        "Criterion's learnable parameters must be in optimizer"
    )


def test_optimizer_paramfree_loss_unchanged():
    """Test param-free losses don't change optimizer param count."""
    from types import SimpleNamespace

    from phenomica.trainer import DistillationTrainer

    # Build trainer with param-free loss (mse)
    training_cfg = SimpleNamespace(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_scheduler=None,
        gradient_clip=None,
        loss_type="mse",
        mse_weight=1.0,
        cosine_weight=1.0,
        use_wandb=False,
        seed=42,
        use_ddp=False,
    )
    model_cfg = SimpleNamespace(
        variant="simple",
        backbone="resnet18",
        projection_dim=768,
        pretrained_backbone=False,
    )
    teacher_cfg = SimpleNamespace(model_name="dinov2_vitb14", embed_dim=768)
    data_cfg = SimpleNamespace(dataset="imagenet", img_size=224)

    trainer = DistillationTrainer(training_cfg, model_cfg, teacher_cfg, data_cfg)

    # Optimizer should have only student params (criterion has none)
    optimizer = trainer.optimizer
    all_params = []
    for group in optimizer.param_groups:
        all_params.extend(group["params"])

    # Count should match student params only
    student = trainer._unwrapped_model()
    student_param_count = sum(1 for _ in student.parameters())
    assert len(all_params) == student_param_count


def test_cospress_loss_forward_returns_scalar():
    """Test CosPressLoss forward returns a scalar."""
    from phenomica.losses import CosPressLoss

    loss_fn = CosPressLoss(cospress_weight=1.0, cospress_temperature=0.1)
    student = torch.randn(4, 768)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)
    assert loss.shape == ()
    assert loss.ndim == 0


def test_cospress_loss_metrics():
    """Test CosPressLoss populates _last_loss_metrics with required keys."""
    from phenomica.losses import CosPressLoss

    loss_fn = CosPressLoss(cospress_weight=1.0, cospress_temperature=0.1)
    student = torch.randn(4, 768)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    _ = loss_fn(student, teacher_outputs)

    assert hasattr(loss_fn, "_last_loss_metrics")
    metrics = loss_fn._last_loss_metrics
    assert "cospress_kl" in metrics
    assert "cospress_cosine" in metrics
    assert "total" in metrics
    assert isinstance(metrics["cospress_kl"], float)
    assert isinstance(metrics["cospress_cosine"], float)
    assert isinstance(metrics["total"], float)


def test_cospress_loss_gradient_flow():
    """Test gradients flow through CosPressLoss."""
    from phenomica.losses import CosPressLoss

    loss_fn = CosPressLoss(cospress_weight=1.0, cospress_temperature=0.1)
    student = torch.randn(4, 768, requires_grad=True)
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)
    loss.backward()
    assert student.grad is not None
    assert student.grad.shape == student.shape


def test_cospress_loss_dim_mismatch():
    """Test CosPressLoss handles dimension mismatch (skips cosine term)."""
    from phenomica.losses import CosPressLoss

    loss_fn = CosPressLoss(
        cospress_weight=1.0, cospress_temperature=0.1, cospress_cosine_weight=0.5
    )
    student = torch.randn(4, 128)  # Different dim
    teacher_outputs = {"cls": torch.randn(4, 768)}
    loss = loss_fn(student, teacher_outputs)

    # Should still work (KL term is dim-agnostic)
    assert loss.shape == ()
    metrics = loss_fn._last_loss_metrics
    # Cosine term should be 0.0 due to dim mismatch
    assert metrics["cospress_cosine"] == 0.0


def test_build_loss_cospress():
    """Test build_loss constructs CosPressLoss with filtered kwargs."""
    from phenomica.losses import CosPressLoss, build_loss

    loss = build_loss(
        "cospress",
        cospress_weight=1.0,
        cospress_temperature=0.2,
        cospress_cosine_weight=0.5,
        epochs=100,  # unrelated, should be filtered out
    )
    assert isinstance(loss, CosPressLoss)
    assert loss.cospress_weight == 1.0
    assert loss.cospress_temperature == 0.2
    assert loss.cospress_cosine_weight == 0.5


def test_cospress_preset_exists():
    """Test 'cospress' preset is registered in hydra-zen store."""
    # Import triggers register_configs() via module-level call
    from hydra_zen import store as zen_store

    from phenomica.configs import register_configs  # noqa: F401

    # Check if cospress preset exists in training group
    # Access via the store's public interface
    assert "training" in zen_store, "training group should exist in store"
    training_keys = zen_store["training"].keys()
    assert ("training", "cospress") in training_keys, "cospress preset should be in training group"


def test_vitkd_loss_forward_scalar():
    """Test ViTKD returns scalar loss."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_loss_metrics():
    """Test ViTKD populates _last_loss_metrics."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_loss_gradients_flow():
    """Test gradients flow to student output and loss params."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_loss_single_layer():
    """Test ViTKD handles layer_patch_tokens with length 1."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_loss_multiple_layers():
    """Test ViTKD handles layer_patch_tokens with length > 1."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_loss_mask_ratio():
    """Test ViTKD respects mask_ratio."""
    from phenomica.losses import ViTKDLoss
def test_vitkd_build_loss():
    """Test build_loss constructs ViTKD with filtered kwargs."""
    from phenomica.losses import ViTKDLoss, build_loss
def test_vitkd_preset_exists():
    """Test vitkd preset is registered in hydra-zen store."""
    from hydra_zen import store
def test_vitkd_dim_mismatch_error():
    """Test ViTKD raises ValueError on student dim mismatch."""
    from phenomica.losses import ViTKDLoss
