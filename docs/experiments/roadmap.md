# Phenomica master roadmap — code → full result

End-to-end map from the codebase to a publishable distillation result, with a
reproducibility lane (W&B + git) running underneath every stage.

```mermaid
flowchart TD
    subgraph CODE["1 · Code (phenomica)"]
        CFG["pydantic + hydra-zen configs<br/>validated, typed, reproducible"]
        DATA["Data: Imagenette / ImageNet-100<br/>(public smoke) → in-domain microscopy later"]
        TEACH["Teacher: DINOv2 frozen<br/>vits14 / vitb14 / vitl14"]
        STU["Student: timm backbone<br/>resnet18 / effnet_b0 / vit_tiny"]
        LOSS["Distillation loss<br/>baseline MSE+cosine → +RKD-angle / FreqKD / CRD"]
        CFG --> DATA
        CFG --> STU
        DATA --> TEACH
        TEACH --> LOSS
        STU --> LOSS
    end

    subgraph TRAIN["2 · Train @ BioHive (submitit)"]
        SUB["submitit launcher<br/>hydra multirun sweep"]
        TR["DistillationTrainer<br/>DDP + AMP"]
        LOSS --> SUB --> TR
    end

    subgraph REPRO["Reproducibility lane (W&B + git)"]
        WB["W&B run per config<br/>+ git SHA + dataset hash + seed"]
        ART["Artifacts:<br/>config • checkpoint • metrics"]
        TR --> WB --> ART
    end

    subgraph EVAL["3 · Evaluate"]
        FID["Fidelity: CKA + cosine vs teacher"]
        KNN["kNN + linear-probe accuracy"]
        PHENO["Phenotypic (Phase 2):<br/>CORUM recall • perturbation • batch-effect"]
        ART --> FID --> KNN --> PHENO
    end

    subgraph RESULT["4 · Full result"]
        LB["Leaderboard:<br/>feature quality vs throughput"]
        DEC{"Beats naive<br/>distillation?"}
        PHENO --> LB --> DEC
        DEC -->|yes| SHIP["Ship distilled student<br/>+ W&B model artifact"]
        DEC -->|no| LOSS
    end
```

## Reading the diagram
- **Stage 1 (Code)** is the merged `pydantic + hydra-zen` config feature: every run
  is a typed, validated, CLI-overridable config — the foundation for reproducible sweeps.
- **Stage 2 (Train)** fans the config matrix out to BioHive via submitit; one SLURM
  job per config (hydra multirun).
- **Reproducibility lane** is not a stage but a cross-cut: every training run logs a
  W&B run stamped with git SHA + dataset hash + seed, and uploads config + checkpoint
  + metrics as W&B artifacts, so any result is re-runnable from its lineage.
- **Stage 3 (Eval)** measures *feature quality*, not just loss: CKA/cosine fidelity to
  the teacher, kNN + linear-probe accuracy now; phenotypic metrics (Phase 2).
- **Stage 4 (Result)** ranks students by quality-vs-throughput and asks the core
  question — does an enriched objective beat naive MSE distillation? If not, the loop
  feeds back to the loss-design stage.

See `experiment-design.md` for the matrix, eval suite, compute, and reproducibility
contract; `beyond-distillation.md` for the research synthesis behind the loss choices.
