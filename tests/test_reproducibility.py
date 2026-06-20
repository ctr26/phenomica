"""Tests for reproducibility provenance and W&B artifact logging."""

from __future__ import annotations


def test_run_provenance_returns_dict():
    """run_provenance returns a dict with git_sha and version keys."""
    from phenomica.reproducibility import run_provenance

    prov = run_provenance()
    assert isinstance(prov, dict)
    assert "git_sha" in prov
    assert "git_dirty" in prov
    assert "python_version" in prov
    assert "torch_version" in prov


def test_run_provenance_git_sha_format():
    """git_sha should be 40-hex or 'unknown'."""
    from phenomica.reproducibility import run_provenance

    prov = run_provenance()
    git_sha = prov["git_sha"]
    assert git_sha == "unknown" or (
        isinstance(git_sha, str) and len(git_sha) == 40
    )


def test_run_provenance_extra_merge():
    """run_provenance merges extra dict into output."""
    from phenomica.reproducibility import run_provenance

    prov = run_provenance(extra={"seed": 42, "dataset": "imagenet"})
    assert prov["seed"] == 42
    assert prov["dataset"] == "imagenet"


def test_dataset_hash_deterministic():
    """dataset_hash produces the same hash for the same input."""
    from phenomica.reproducibility import dataset_hash

    paths = ["data/train", "data/val"]
    hash1 = dataset_hash(paths)
    hash2 = dataset_hash(paths)
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) > 0


def test_dataset_hash_order_stable():
    """dataset_hash should be stable across input order (sorted internally)."""
    from phenomica.reproducibility import dataset_hash

    hash1 = dataset_hash(["b", "a", "c"])
    hash2 = dataset_hash(["a", "b", "c"])
    assert hash1 == hash2


def test_log_artifacts_no_op_when_run_none():
    """log_artifacts is a safe no-op when run is None."""
    from phenomica.reproducibility import log_artifacts

    # Should not raise
    log_artifacts(None, config_path="config.yaml", checkpoint_path="model.pt")
    log_artifacts(
        None, metrics={"accuracy": 0.9, "loss": 0.1}
    )
