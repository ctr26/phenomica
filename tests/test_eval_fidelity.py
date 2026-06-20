"""Tests for feature-fidelity evaluation metrics (CKA + cosine)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from phenomica.eval import feature_fidelity, linear_cka, mean_cosine


def test_linear_cka_identical_matrices():
    """CKA of identical matrices should be exactly 1.0."""
    x = torch.randn(100, 64)
    cka = linear_cka(x, x)
    assert abs(cka - 1.0) < 1e-6


def test_linear_cka_orthogonal_invariant():
    """CKA should be invariant to orthogonal rotation."""
    x = torch.randn(100, 64)
    # Random orthogonal matrix via QR decomposition
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    y = x @ Q
    cka = linear_cka(x, y)
    assert abs(cka - 1.0) < 1e-4


def test_linear_cka_isotropic_scaling_invariant():
    """CKA should be invariant to isotropic scaling."""
    x = torch.randn(100, 64)
    y = x * 5.0
    cka = linear_cka(x, y)
    assert abs(cka - 1.0) < 1e-6


def test_mean_cosine_identical():
    """Mean cosine of identical vectors should be 1.0."""
    x = torch.randn(100, 64)
    cosine = mean_cosine(x, x)
    assert abs(cosine - 1.0) < 1e-6


def test_mean_cosine_in_range():
    """Mean cosine should lie in [-1, 1]."""
    x = torch.randn(100, 64)
    y = torch.randn(100, 64)
    cosine = mean_cosine(x, y)
    assert -1.0 <= cosine <= 1.0


def _tiny_loader(n: int = 16, batch: int = 8) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 224, 224), torch.zeros(n, dtype=torch.long)),
        batch_size=batch,
    )


def test_feature_fidelity_simple_student():
    """feature_fidelity runs on a real SimpleDistiller vs the fake teacher.

    Uses the distilled ``forward()`` output (projection dim == teacher embed_dim).
    The backbone-based ``extract_features`` would mismatch dims (e.g. 512 vs 768)
    and crash cosine -- this guards the correct fidelity pairing.
    """
    from phenomica.configs import ModelConfig
    from phenomica.models import build_model
    from tests.conftest import FakeDINOv2Teacher

    teacher = FakeDINOv2Teacher()  # embed_dim 768
    student = build_model(
        ModelConfig(variant="simple", pretrained_backbone=False)
    ).eval()

    metrics = feature_fidelity(student, teacher, _tiny_loader(), device="cpu")
    assert np.isfinite(metrics["cka"]) and np.isfinite(metrics["cosine"])
    assert -1.0 <= metrics["cka"] <= 1.0
    assert -1.0 <= metrics["cosine"] <= 1.0


def test_feature_fidelity_multifunction_student():
    """feature_fidelity handles the multifunction student (dict output -> global head)."""
    from phenomica.configs import ModelConfig
    from phenomica.models import build_model
    from tests.conftest import FakeDINOv2Teacher

    teacher = FakeDINOv2Teacher()
    student = build_model(
        ModelConfig(variant="multifunction", pretrained_backbone=False)
    ).eval()

    metrics = feature_fidelity(student, teacher, _tiny_loader(), device="cpu")
    assert np.isfinite(metrics["cka"]) and np.isfinite(metrics["cosine"])
    assert -1.0 <= metrics["cosine"] <= 1.0


def test_knn_linear_probe_trivial_separable():
    """kNN and linear-probe should achieve perfect accuracy on trivially separable data."""
    from phenomica.eval import knn_accuracy, linear_probe_accuracy

    # Two clearly separated clusters
    train_features = np.vstack(
        [np.random.randn(50, 10) + 5, np.random.randn(50, 10) - 5]
    )
    train_labels = np.array([0] * 50 + [1] * 50)
    test_features = np.vstack(
        [np.random.randn(20, 10) + 5, np.random.randn(20, 10) - 5]
    )
    test_labels = np.array([0] * 20 + [1] * 20)

    knn_acc = knn_accuracy(train_features, train_labels, test_features, test_labels)
    linear_acc = linear_probe_accuracy(
        train_features, train_labels, test_features, test_labels
    )

    assert knn_acc >= 0.95
    assert linear_acc >= 0.95
