# Phenomica experiment design

**Goal:** establish whether phenomica's distilled students can match DINOv2 feature
quality at a fraction of the compute, and whether *enriched* distillation objectives
beat naive feature-MSE.

**Scope of the current iteration (confirmed):** plan + wire infrastructure with a
**dry-run** only — no GPU spend from this session. The maintainer launches the actual
sweep after review. First data is a **public subset** (Imagenette / ImageNet-100).
Eval is **fidelity + kNN/linear-probe now, phenotypic later**.

## 1 · Dataset
- **Phase 1 (smoke + first real runs):** Imagenette (10-class, ~13k imgs) for the
  pipeline smoke; ImageNet-100 for the first quality numbers. Both public + small +
  reproducible; no cluster data path required (download/cache once, hash-pin).
- **Phase 2 (domain):** in-domain microscopy / Cell-Painting (e.g. RxRx, JUMP-CP) —
  deferred; see `beyond-distillation.md` (intern C) for the recommended benchmark.

## 2 · Experiment matrix
Axes (Hydra config groups make every cell a one-line override):

| Axis | Values |
|------|--------|
| student | `resnet18`, `efficientnet_b0`, `vit_tiny_patch16_224` |
| teacher | `dinov2_small` (384), `dinov2_base` (768), `dinov2_large` (1024) |
| objective | `baseline` (MSE+cosine) · `rkd_angle` · `freqkd` · `multifunction` |

- **Phase 0 — smoke (dry-run target):** 1 cell — `resnet18 × dinov2_small × baseline`,
  Imagenette, 2 epochs, 1 GPU. Proves data → teacher → student → loss → checkpoint →
  W&B → eval end-to-end. **This is what we dry-run via bh-tunnel.**
- **Phase 1 — objective ablation:** fix `resnet18 × dinov2_base`, sweep the 4 objectives
  on ImageNet-100. Answers "does enriched objective beat naive MSE?" (the headline).
- **Phase 2 — capacity/teacher sweep + phenotypic:** best objective × all students ×
  all teachers; add the phenotypic eval suite.

New objectives to wire (from intern A research — see `beyond-distillation.md`):
- **RKD-Angle** — relational/angular preservation; cheap drop-in, strong on transfer.
- **FreqKD** — frequency-decoupled MSE; cheap; reported gains on DINOv2 distillation.
- (stretch) **CRD / cosine+InfoNCE** — contrastive; counters dimensional collapse;
  needs batch ≥ 256.

## 3 · Evaluation suite
- **Phase 1 (wire now):**
  - **Fidelity** — linear CKA + mean cosine of student vs teacher features on a held-out
    split (CKA is the primary; catches dimensional collapse cosine misses).
  - **Downstream** — kNN (k=20) + linear-probe top-1 on the eval set.
  - **Efficiency** — student throughput (img/s) + params vs the teacher (the "speedup
    @ fidelity-retained" axis of the leaderboard).
- **Phase 2 (roadmap):** phenotypic suite — known-relationship recall (CORUM/StringDB),
  perturbation classification, replicate/batch-effect consistency. Spec pending intern C.

## 4 · Compute (BioHive via submitit)
- Launch through the existing `cluster=biohive` config preset (submitit), one SLURM job
  per config via **hydra multirun** + the submitit launcher.
- Phase 0 smoke: 1× GPU, 2 epochs, `training=debug`-style overrides. Phase 1: 1 GPU/cell.
- Dry-run first (`--multirun ... hydra.launcher...` validated, no submission) via
  `bh-tunnel`; maintainer fires the real sweep.

## 5 · Reproducibility contract (W&B + git)
Every run MUST be reconstructable from its W&B lineage:
- **W&B run** per config, name = `{student}-{teacher}-{objective}-{shortsha}`.
- **Stamp** each run's config with: resolved Hydra config, **git SHA** (+ dirty flag),
  dataset name + content hash, seed, library versions.
- **Artifacts:** (a) the resolved config (YAML), (b) best + final checkpoints,
  (c) eval metrics table. Link artifacts so a run → exact code + data + weights.
- **Determinism:** fixed seed; `set_seed(..., deterministic=True)` opt-in for the
  reference runs.

## 6 · Execution plan (this iteration = code + dry-run, no spend)
1. Build the **experiment runner** (TDD) on `feat/craig.russell/experiment-runner`
   (stacked on the config feature): eval module (CKA/cosine/kNN/linear-probe), W&B
   artifact + git-SHA logging, hydra multirun + submitit launcher wiring.
2. Unit-test eval metrics + repro stamping on CPU with the fake teacher.
3. **Dry-run** the Phase 0 smoke via bh-tunnel (validate the submitit launch plan).
4. Hand the maintainer the exact launch command for the real smoke.

## Out of scope (this iteration)
GPU spend / real submission; the Phase 2 phenotypic suite + microscopy data wiring;
merging to main (gated on the feature pre-merge review).
