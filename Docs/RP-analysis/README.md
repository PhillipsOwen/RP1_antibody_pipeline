# MISM — Research Project Overview

This repository spans three interconnected research projects (RPs) linking within-host molecular immunology to population-level disease dynamics. Together they form a pipeline from antibody-scale molecular interactions to epidemic transmission modeling.

---

## Research Projects

### RP1: Antibody Discovery Pipeline

**Goal:** Predict and discover broadly neutralizing antibodies against viral escape mutants.

**Approach:** 16-stage computational pipeline integrating viral escape panel generation, BCR repertoire analysis, antibody language models (ESM2), molecular dynamics simulations, VAE/GAN structural modeling, Markov state modeling, synthetic evolution, and a lab-in-the-loop experimental feedback cycle.

**Analysis:** [RP1_analysis.md](RP1_analysis.md)

**Codebase:** [RP1_antibody_pipeline/](RP1_antibody_pipeline/)

---

### RP2: Spatiotemporal Dynamics of Cell-Virus Interactions

**Goal:** Model how viruses (EBV, HIV, SARS-CoV-2) spread within lymph node tissue at single-cell resolution.

**Approach:** Four-stage abstraction pipeline — experimental scRNA-seq data → ETL → agent-based model (ABM) of synthetic lymph node tissue → physics-informed neural network (PINN) for parameter extraction → ODE/PDE/SDE forward simulation with MongoDB checkpointing.

**Analysis:** [RP2_Analysis.md](RP2_Analysis.md)

---

### RP3: Link Host Immune Response to Population Disease Transmission

**Goal:** Connect within-host immunological dynamics to population-scale epidemic spread.

**Approach:** Three-stage pipeline — ODE/differential equations modeling individual immune responses → agent-based model (ABM) of synthetic dynamic contact networks (R + Julia) → stochastic epidemic model (CTMC) for Bayesian inference on population-level outbreak data.

**Analysis:** [RP3_analysis.md](RP3_analysis.md)

---

## Cross-RP Integration

| From | To | Data Passed | Status |
|---|---|---|---|
| RP1 (antibody fitness scores, escape panel) | RP2 (ABM viral dynamics parameters) | Viral escape profiles, binding affinities | Conceptual — interface not yet defined |
| RP2 (within-host viral dynamics) | RP3 ODE stage | Individual-level infectiousness trajectories ("Bruce's outputs") | Planned — schema not yet defined |
| RP3 ODE → ABM → SEM | Population-scale epidemic inference | Immune state → contact network → outbreak estimates | In development |

---

## Repository Structure

```
MISM/
├── README.md                        # This file
├── RP1_analysis.md                  # RP1 data model and pipeline analysis
├── RP2_Analysis.md                  # RP2 data model and pipeline analysis
├── RP3_analysis.md                  # RP3 data model and pipeline analysis
│
├── RP1_antibody_pipeline/           # RP1 codebase (Python, 16-stage pipeline)
│   ├── README.md
│   ├── main.py
│   ├── config.py
│   ├── data/                        # Input sequences (FASTA, OAS CSVs)
│   ├── models/                      # ALM, VAE, GAN
│   ├── md_simulations/              # OpenMM/GROMACS/CHARMM runners
│   ├── msm_analysis/                # PyEMMA MSM
│   ├── synthetic_evolution/         # Genetic algorithms
│   ├── viral_escape/                # Escape panel generation
│   ├── experiments/                 # Validation, lab loop, checkpoints
│   └── Docs/                        # RP1 documentation
│
└── Docs/
    └── RP1 data pipeline AI analysi.txt   # Background notes on RP1
```

---

## Key File Formats Across All RPs

| Format | Used In | Purpose |
|---|---|---|
| `.fasta` | RP1, RP3 | Biological sequence input (viral genomes, antigen sequences) |
| `.csv` | RP1, RP2, RP3 | Tabular data — BCR repertoires, contact networks, surveillance counts |
| `.npy` | RP1, RP2 | NumPy arrays — matrices, embeddings, trajectories |
| `.pkl` | RP1, RP2 | Python pickled objects — models, atlases, MSMs |
| `.pt` | RP1 | PyTorch model weights |
| `.h5` / `.h5ad` | RP2 | HDF5 / AnnData — scRNA-seq data, high-dimensional arrays |
| `.xtc` / `.dcd` / `.nc` | RP1, RP2 | MD simulation trajectories |
| `.xml` (SBML) | RP2, RP3 | Systems Biology Markup Language — ODE model definitions |
| `.json` | RP1, RP2, RP3 | Reports, configs, parameter estimates, checkpoint metadata |
| `.rds` / `.RData` | RP3 | R serialized objects — ABM outputs, posterior samples |
| `.graphml` / `.gexf` | RP3 | Network graph formats — dynamic contact network evolution |
| MongoDB BSON | RP2 | Semi-fault-tolerant simulation state checkpoints |
