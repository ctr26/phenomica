"""Feature quality evaluation via kNN and linear probe."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from all samples using ``model.extract_features``.

    Args:
        model: A student model with an ``extract_features`` method.
        dataloader: Yields ``(images, labels)`` batches.
        device: Device to run inference on.

    Returns:
        ``(features, labels)`` as NumPy arrays with shapes
        ``[N, D]`` and ``[N]`` respectively.
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            features = model.extract_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def knn_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 20,
) -> float:
    """Compute k-NN classification accuracy.

    Args:
        train_features: Training feature matrix ``[N_train, D]``.
        train_labels: Training labels ``[N_train]``.
        test_features: Test feature matrix ``[N_test, D]``.
        test_labels: Test labels ``[N_test]``.
        k: Number of neighbours.

    Returns:
        Classification accuracy as a float in ``[0, 1]``.
    """
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(train_features, train_labels)
    return float(clf.score(test_features, test_labels))


def linear_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Linear probe accuracy via logistic regression.

    Uses L-BFGS solver with up to 1000 iterations and L2 regularisation.

    Args:
        train_features: Training feature matrix ``[N_train, D]``.
        train_labels: Training labels ``[N_train]``.
        test_features: Test feature matrix ``[N_test, D]``.
        test_labels: Test labels ``[N_test]``.

    Returns:
        Classification accuracy as a float in ``[0, 1]``.
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(train_features, train_labels)
    return float(clf.score(test_features, test_labels))


def evaluate_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Run the full evaluation suite (kNN + linear probe).

    Args:
        model: Student model with ``extract_features``.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device string.

    Returns:
        Dict with ``knn_accuracy`` and ``linear_probe_accuracy`` keys.
    """
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, val_loader, device)

    return {
        "knn_accuracy": knn_accuracy(train_features, train_labels, test_features, test_labels),
        "linear_probe_accuracy": linear_probe_accuracy(
            train_features, train_labels, test_features, test_labels
        ),
    }
