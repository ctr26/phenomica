# Beyond "just distilling" — research synthesis

Synthesis of three deep-research passes (objectives · data/architecture · microscopy &
evaluation) into actionable directions for phenomica. All claims are from the cited
sources; gains are as-reported (task-dependent — treat as priors, not guarantees).

## Thesis
Naive feature-MSE distillation (phenomica's current baseline) has a specific failure
mode — **dimensional collapse**: the student can drive MSE down while using only a few
effective dimensions, discarding the teacher's representational richness (cosine+InfoNCE
reportedly expands effective rank ~2.4×, 16→38 dims). phenomica can beat "just distilling"
on four levers: **objective**, **data/augmentation**, **architecture-agnostic transfer**,
and **domain + evaluation**.

## 1 · Objective-level losses  (cheap drop-ins unless noted)
| Technique | Mechanism | Why it beats MSE | Reported gain | Cost |
|-----------|-----------|------------------|---------------|------|
| **CRD** (Tian 2020) | contrastive feature alignment | preserves relational structure; counters collapse | +2–4% ImageNet | cheap (batch ≥256) |
| **RKD** (Park 2019) | angular + distance ratios | transfers geometry, not just points | +1.8–3.2% transfer | cheap |
| **CCKD** (Peng 2019) | correlation congruence (2nd-order) | captures inter-sample structure | +1.5–3% dense | cheap |
| **DKD** (Zhao 2022) | decouple target vs non-target | better gradient signal | comparable/better | cheap |
| **FreqKD** (2026) | frequency-decoupled MSE | relaxes high-freq, strict low-freq | +2.4 mAP50 (DINOv2 KD) | cheap |
| cosine+InfoNCE (2026) | angular + contrastive | directly expands effective rank | rank 2.4× | cheap (thin evidence) |
| SSKD / token-level (DeiT) | SSL aux task / ViT tokens | richer supervisory signal | varies | moderate |

**Try first:** **CRD** (or **RKD-angle**) as the first enriched objective vs the MSE+cosine baseline.
Sources: CRD arxiv.org/abs/1910.10699 · RKD arxiv.org/abs/1904.05068 · CCKD (Peng 2019) · DKD arxiv.org/abs/2203.08679.

## 2 · Data / augmentation / architecture
- **RSD — Redundancy Suppression Distillation** (2025): drives the penultimate correlation
  matrix toward identity to suppress architecture-specific patterns → **built for ViT→CNN**,
  which is *exactly* phenomica (DINOv2→ResNet/EffNet). +2.34% ImageNet (Swin-T→RN18). CHEAP.
- **Cosine embedding loss**: magnitude-insensitive drop-in for MSE; preserves orientation. CHEAP.
- **SwAV multi-crop** (ICML 2020): 2 global + N small crops → multi-scale signal (DINOv2 itself
  is multi-crop). CHEAP, augmentation-only. arxiv.org/abs/2006.09882.
- **Scale-KD** multi-scale attention: adaptable but weak accuracy evidence — deprioritize.
- Multi-teacher: **avoided** — 2× teacher inference cost, mostly video-domain evidence.

**Try first:** RSD + cosine-embedding loss + multi-crop augmentation.

## 3 · Microscopy domain — KEY STRATEGIC FINDING
- **A natural-image DINOv2 teacher will NOT preserve phenotypic signal.** Microscopy-pretrained
  teachers (OpenPhenom-S/16, SubCell, CA-MAE) dominate; "even the smallest CA-MAE-S/16 trained on
  microscopy outperforms the largest ViTs trained on natural images" (ViTally Consistent 2024,
  arxiv.org/html/2411.02572). Linear-probe acc correlates ρ=0.97 with StringDB recall.
  → **If microscopy is the target, distill from a domain FM (OpenPhenom/SubCell), not vanilla DINOv2.**
- Self-distillation (DINO-style) *preserves and enriches* phenotypic signal — Cell-DINO +20% avg,
  +70% at 1% labels (PLOS Comp Bio 2025).
- **Channel-agnostic** patch-embed (4–5 Cell-Painting channels, any order); microscopy augmentation
  (drop blur/solarize/greyscale; add per-channel jitter, channel zeroing, intensity rescaling);
  self-normalization per channel.
- **Batch effects need explicit post-hoc correction** (Harmony / Seurat RPCA) — augmentation alone
  does not fix them (Nat Comm 2024).
- **Open gap = phenomica's opportunity:** no published evidence on whether a *compressed* student
  (foundation teacher → tiny student) **retains phenotypic signal**. A systematic study of that is a
  publishable contribution, well beyond generic distillation. Sources: rxrx.ai/phenom ·
  biorxiv 2024.12.06.627299 (SubCell) · PLOS pcbi.1013828 (Cell-DINO).

## 4 · Evaluation — what "full result" means
ImageNet linear-probe is **uncorrelated** with phenotypic signal — skip it as the headline.
Tiered suite (intern C):
- **Tier 1 (cheap, wire now):** CKA fidelity to teacher · kNN + linear-probe on RxRx3-core ·
  replicate consistency (KS / Cramér–von Mises vs random-pair null).
- **Tier 2 (medium):** biological-relationship recall (StringDB/CORUM) · perturbation/MoA
  classification (JUMP-CP) · batch-effect metrics (kBET / silhouette / ARI).
- **Tier 3 (heavy):** RxRx1 1139-class probe · zero-shot compound–gene activity · cross-dataset
  (JUMP-CP external) generalization.
- **"Full result" = Tier 1 + bio-relationship recall + one MoA benchmark.**
Datasets: RxRx3-core (github.com/recursionpharma/rxrx3-core) · JUMP-CP · RxRx1 (rxrx.ai/rxrx1).

## How phenomica beats "just distilling" — prioritized
1. Swap MSE → **CRD/RKD** (counter dimensional collapse) — Phase 1 objective ablation.
2. Add **multi-crop aug + RSD** (ViT→CNN) — Phase 1.
3. For microscopy: distill from a **domain FM**, channel-agnostic student, domain aug — Phase 2.
4. Evaluate on **CKA + biological recall**, not ImageNet — built into the eval suite.

**Novelty hook:** first systematic test of whether a *compressed* student retains *phenotypic*
signal from a foundation teacher — the question the field hasn't answered.
