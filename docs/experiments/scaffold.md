# Ray experiment scaffold

The Ray stack (Ray Data + Ray Train + Ray Tune) is added as an **independent
launch path** alongside the existing submitit/Hydra path. Neither path imports
the other; they share the model/loss/teacher/eval/config building blocks.

This document is the implemented-interface reference for the Ray path: the
config schemas, module signatures, console scripts, and justfile targets below
are all live. Run the two paths via `just` (`ray-train-local` / `ray-tune-local`)
or the console scripts directly.

## Two independent launch paths

| | submitit path | Ray path |
|---|---|---|
| Entry module | `phenomica.train` (`@hydra.main`) | `phenomica.ray_launch` (`ray_train_main` / `ray_tune_main`) |
| Console script | `phenomica-train` | `phenomica-ray-train`, `phenomica-ray-tune` |
| Data | `phenomica.data.create_dataloaders` (torch `DataLoader`) | `phenomica.ray_data.build_ray_dataset` (`ray.data.Dataset`) |
| Distribution | `torch.nn.parallel.DDP` + `DistillationTrainer` | `ray.train.torch.TorchTrainer` + `ScalingConfig` |
| Sweeps | Hydra `--multirun` + submitit launcher | `ray.tune.Tuner` + `ASHAScheduler` |
| Scheduler target | SLURM (BioHive `hopper` partition) | Ray cluster (or local) |

`train.py` is **not touched** by the Ray work. The submitit path remains the
source of truth for the torch flow; the Ray modules add their own CPU-local
tests on top of the original 46-test suite.

## Shared (reused, unchanged) building blocks

The Ray path reuses these as-is — no reimplementation:

- `phenomica.models.build_model(cfg)` -> `SimpleDistiller | MultiFunctionDistiller`
- `phenomica.teacher.DINOv2Teacher` (frozen; `.embed_dim`, dict output with
  `cls`/`patch_stats`/`layer_features`)
- `phenomica.losses.DistillationLoss` / `MultiFunctionDistillationLoss`
- `phenomica.eval` (CKA / cosine / kNN / linear probe)
- `phenomica.reproducibility.run_provenance` / `log_artifacts`
- `phenomica.configs` (`ModelConfig`/`TeacherConfig`/`DataConfig`/`TrainingConfig`)

The dim-sync invariant from `DistillationTrainer.__init__` (student head dims
follow the loaded teacher `embed_dim`, and a mismatch vs `TeacherConfig.embed_dim`
raises) must be preserved inside `ray_train.train_fn`.

## Repo structure (Ray additions)

```
src/phenomica/
  configs.py        # + RayDataConfig, RayTrainConfig, RayTuneConfig, RayTuneSearchSpace
  ray_data.py       # build_ray_dataset, preprocess_batch
  ray_train.py      # train_fn (per-worker), run_ray_train (driver)
  ray_tune.py       # build_search_space, run_ray_tune
  ray_launch.py     # ray_train_main, ray_tune_main (hydra entry points)
docs/experiments/
  running.md        # submitit path (existing)
  scaffold.md       # this file
tests/
  test_ray_*.py     # CPU-only/local Ray tests (build stage)
```

## Module interfaces (fixed signatures)

### `phenomica.ray_data`

```python
IMAGE_COLUMN = "image"   # CHW float32 after preprocess
LABEL_COLUMN = "label"   # int class index

def preprocess_batch(batch: dict[str, Any]) -> dict[str, Any]: ...

def build_ray_dataset(
    data_cfg: DataConfig,
    ray_data_cfg: RayDataConfig,
    *,
    split: str = "train",
) -> ray.data.Dataset: ...
```

- `read_images(root, ...)` reads class subfolders as labels; `map_batches`
  applies `preprocess_batch` (resize -> center-crop -> to-CHW-float32 ->
  ImageNet normalize), mirroring `data.get_transforms(is_train=False)`.
- Read parallelism from `ray_data_cfg.parallelism` /
  `ray_data_cfg.override_num_blocks`.

### `phenomica.ray_train`

```python
def train_fn(config: dict[str, Any]) -> None: ...      # runs per worker

def run_ray_train(
    model_cfg: ModelConfig,
    teacher_cfg: TeacherConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    ray_train_cfg: RayTrainConfig,
    ray_data_cfg: RayDataConfig,
) -> ray.train.Result: ...
```

`train_fn(config)` (= TorchTrainer `train_loop_config`):
1. `build_model` -> `ray.train.torch.prepare_model` (DDP wrap);
2. instantiate `DINOv2Teacher` on the worker device; sync head dims to
   `teacher.embed_dim`; select loss by `model.variant`;
3. `ray.train.get_dataset_shard("train")` (+ optional `"val"`) ->
   `iter_torch_batches(...)`;
4. each epoch -> `ray.train.report({"train_loss": ..., "val_loss": ...})`
   (`val_loss` is the metric ASHA/Tune consume).

`run_ray_train(...)`:
- `train_ds = build_ray_dataset(data_cfg, ray_data_cfg, split="train")`
  (and `"val"`);
- `TorchTrainer(train_fn, train_loop_config={...serialized cfgs...},`
  `scaling_config=ScalingConfig(num_workers, use_gpu, resources_per_worker={"CPU": ..., "GPU": ...}),`
  `datasets={"train": train_ds, "val": val_ds},`
  `run_config=RunConfig(storage_path=..., callbacks=[wandb UserCallback] if training_cfg.use_wandb else []))`;
- return `trainer.fit()`.

W&B note (Ray Train **v2**): `RunConfig.callbacks` accepts only
`ray.train.UserCallback` subclasses, so the run logs via a small driver-side
`UserCallback` (one rank-0 `wandb.init`, then `.log` per reported metrics),
**not** the legacy Tune `WandbLoggerCallback`. `Result.metrics` is populated
only when a checkpoint is reported, so rank 0 reports a checkpoint each epoch.

### `phenomica.ray_tune`

```python
def build_search_space(tune_cfg: RayTuneConfig) -> dict[str, Any]: ...

def run_ray_tune(
    model_cfg, teacher_cfg, data_cfg, training_cfg,
    ray_train_cfg, ray_data_cfg, ray_tune_cfg,
) -> ray.tune.ResultGrid: ...
```

- `build_search_space` -> `{"train_loop_config": {"lr": tune.loguniform(lr_min, lr_max),
  "weight_decay": tune.loguniform(wd_min, wd_max), "loss_type": tune.choice(loss_types)}}`.
- "Tune over Train" pattern (Ray **v2**, 2.55): the legacy
  `Tuner(trainer_instance, ...)` API was deprecated in 2.43 and is rejected by
  Train v2, so the Tuner trainable is a **driver function** that builds the
  per-trial `TorchTrainer` itself (same shape as `run_ray_train`). The shared
  base `train_loop_config` and the Ray Data shards are injected via
  `tune.with_parameters` so they are not part of the search space.
- `Tuner(driver, param_space=build_search_space(...),`
  `tune_config=TuneConfig(metric=ray_tune_cfg.metric, mode=ray_tune_cfg.mode,`
  `num_samples=..., max_concurrent_trials=...,`
  `scheduler=ASHAScheduler(grace_period=..., max_t=..., reduction_factor=...)),`
  `run_config=RunConfig(callbacks=[WandbLoggerCallback(...)] if use_wandb else []))`;
  return `tuner.fit()`.
- Two callback layers bridge the libraries: a Train `TuneReportCallback`
  (`ray.tune.integration.ray_train`) on the **inner** trainer propagates worker
  `ray.train.report` metrics up to ASHA; the Tune `WandbLoggerCallback`
  (`ray.air.integrations.wandb`) on the **Tuner** logs each trial (opt-in).

### `phenomica.ray_launch`

```python
def ray_train_main(cfg: DictConfig) -> None: ...   # phenomica-ray-train
def ray_tune_main(cfg: DictConfig) -> None: ...    # phenomica-ray-tune
```

Mirror `train.py`: `@hydra.main` + a top-level `ConfigStore` dataclass whose
`defaults` compose `model`/`teacher`/`data`/`training` plus `ray_data`/
`ray_train` (and `ray_tune` for the tune entry). The submitit-only `cluster`
group is intentionally absent (Ray scaling lives in `ray_train`).

Each `@hydra.main` callable delegates to a `hydra_zen.zen`-wrapped task
function whose parameters are named after the config groups; `zen` auto-extracts
and instantiates each group, with `pydantic_parser` passed as the
`instantiation_wrapper` so every group is validated exactly as in `train.py`.
The wrapped task functions are Hydra-agnostic (callable with plain config dicts)
and just dispatch to `run_ray_train` / `run_ray_tune`. The config schema classes
are imported at module runtime (not under `TYPE_CHECKING`) so pydantic can
resolve the task-function forward-ref annotations during validation.

## Config schemas (`phenomica.configs`)

Registered as hydra-zen groups via the existing `_preset()` helper.

| Group | Schema | Presets |
|---|---|---|
| `ray_data` | `RayDataConfig` | `default` |
| `ray_train` | `RayTrainConfig` | `local_cpu`, `biohive_gpu` |
| `ray_tune` | `RayTuneConfig` (nests `RayTuneSearchSpace`) | `default`, `debug` |

- **`RayDataConfig`**: `parallelism=-1`, `shuffle_buffer_size=None`,
  `prefetch_batches=2`, `override_num_blocks=None`.
- **`RayTrainConfig`**: `num_workers=1`, `use_gpu=False`, `cpus_per_worker=1`,
  `gpus_per_worker=1`, `max_epochs=10`, `storage_path=None`.
- **`RayTuneSearchSpace`**: `lr_min=1e-5`, `lr_max=1e-2`,
  `weight_decay_min=1e-6`, `weight_decay_max=1e-2`,
  `loss_types=["mse","cosine","combined"]`.
- **`RayTuneConfig`**: `num_samples=4`, `metric="val_loss"`, `mode="min"`,
  `grace_period=1`, `max_t=10`, `reduction_factor=2`,
  `max_concurrent_trials=2`, `search_space=RayTuneSearchSpace()`.

Search-space bounds are stored as primitives (not Ray `Domain` objects) so the
dataclasses stay Hydra/pydantic-serializable; `build_search_space` converts
them to `tune.*` domains at launch.

## Running each path

### submitit (existing, unchanged)
```bash
uv run phenomica-train model=simple_resnet18 teacher=dinov2_base
uv run phenomica-train model=multi_resnet18 teacher=dinov2_large cluster=biohive
```
See [running.md](running.md).

### Ray Train (single distributed run)
```bash
# local CPU smoke (also: just ray-train-local)
uv run phenomica-ray-train model=simple_resnet18 teacher=dinov2_small \
    data=imagenette training=debug ray_train=local_cpu ray_data=default

# multi-GPU
uv run phenomica-ray-train model=simple_resnet18 teacher=dinov2_base \
    data=imagenet ray_train=biohive_gpu ray_data=default
```

### Ray Tune (ASHA sweep)
```bash
# local CPU sweep (also: just ray-tune-local)
uv run phenomica-ray-tune model=simple_resnet18 teacher=dinov2_small \
    data=imagenette training=debug ray_train=local_cpu ray_tune=debug ray_data=default
```

The `just` targets `ray-train-local` / `ray-tune-local` wrap the CPU-local smoke
forms; they sit alongside the submitit `smoke` / `sweep-objectives` targets but
share no SLURM/submitit machinery.

## W&B integration

- **Per-trial / per-run logging** goes through `WandbLoggerCallback`
  (`ray.air.integrations.wandb`) attached to the `RunConfig.callbacks`, gated
  on `training_cfg.use_wandb`. This is the Ray-native equivalent of the manual
  `wandb.init` in `DistillationTrainer`; the Ray path does **not** call
  `wandb.init` directly.
- Tests run offline: `use_wandb=False` so no callback is attached and no
  network/auth is required.

## Testing constraints (build stage)

Ray tests must be **CPU-only and local**: `use_gpu=False`, `num_workers=1`,
tiny synthetic data, `FakeDINOv2Teacher` (from `tests/conftest.py`), a small
local `ray.init(...)`/`ray.shutdown()`, and `use_wandb=False`. No GPU, no
cluster, no real W&B. Mark any test exceeding a few seconds `@pytest.mark.slow`.
