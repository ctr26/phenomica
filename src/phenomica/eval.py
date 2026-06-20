"""Feature quality evaluation via kNN and linear probe."""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute linear Centered Kernel Alignment between two feature matrices.

    CKA measures similarity between two representations by comparing their
    Gram matrices after centering. It is invariant to orthogonal transformations
    and isotropic scaling, making it more robust than per-sample cosine similarity
    for detecting dimensional collapse.

    Formula:
        CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
        where K_X = X @ X.T is the linear (dot-product) Gram matrix,
        and HSIC is the Hilbert-Schmidt Independence Criterion (centering + Frobenius).

    Args:
        x: Feature matrix ``[N, D1]``.
        y: Feature matrix ``[N, D2]``.

    Returns:
        CKA score in ``[0, 1]``, where 1.0 = perfectly aligned.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Gram matrices (linear kernel)
    gram_x = x @ x.T
    gram_y = y @ y.T

    # HSIC: center the Gram matrices and compute Frobenius inner product
    n = x.size(0)
    h = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
    gram_x_centered = h @ gram_x @ h
    gram_y_centered = h @ gram_y @ h

    hsic_xy = torch.sum(gram_x_centered * gram_y_centered)
    hsic_xx = torch.sum(gram_x_centered * gram_x_centered)
    hsic_yy = torch.sum(gram_y_centered * gram_y_centered)

    cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    return float(cka.item())


def mean_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean row-wise cosine similarity between two feature matrices.

    Args:
        x: Feature matrix ``[N, D]``.
        y: Feature matrix ``[N, D]``.

    Returns:
        Mean cosine similarity in ``[-1, 1]``.
    """
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
    cosine = (x_norm * y_norm).sum(dim=1)
    return float(cosine.mean().item())


def feature_fidelity(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Measure student vs teacher feature fidelity (CKA + cosine similarity).

    Extracts CLS/feature embeddings from both models over the loader and
    computes linear CKA (primary metric for dimensional collapse) and
    mean cosine similarity.

    Args:
        student: Student model whose ``forward`` (or ``"global"`` head, for the
            multifunction variant) produces the distilled embedding.
        teacher: Teacher model returning dict with ``"cls"`` key.
        loader: Data loader yielding ``(images, labels)`` batches.
        device: Device to run inference on.

    Returns:
        Dict with ``"cka"`` and ``"cosine"`` keys.
    """
    student.eval()
    teacher.eval()

    student_features = []
    teacher_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            # Compare the student's *distilled* output -- forward() for the simple
            # variant, the "global" head for the multifunction variant -- against the
            # teacher CLS, mirroring what the distillation loss optimises. (The raw
            # backbone via extract_features would be a different, dim-mismatched space.)
            student_out = student(images)
            if isinstance(student_out, dict):
                student_out = student_out["global"]
            teacher_out = teacher(images)["cls"]
            student_features.append(student_out.cpu())
            teacher_features.append(teacher_out.cpu())

    student_features = torch.cat(student_features, dim=0)
    teacher_features = torch.cat(teacher_features, dim=0)

    return {
        "cka": linear_cka(student_features, teacher_features),
        "cosine": mean_cosine(student_features, teacher_features),
    }


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
        "knn_accuracy": knn_accuracy(
            train_features, train_labels, test_features, test_labels
        ),
        "linear_probe_accuracy": linear_probe_accuracy(
            train_features, train_labels, test_features, test_labels
        ),
    }
