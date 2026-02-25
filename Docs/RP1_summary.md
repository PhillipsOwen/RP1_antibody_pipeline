# RP1 Antibody Pipeline - Comprehensive Summary

## Overview

The **RP1_antibody_pipeline** is a sophisticated computational biology framework for **predicting antibody responses to viral escape mutants**. This end-to-end pipeline integrates molecular dynamics simulations, machine learning (language models, VAEs, GANs), Markov State Modeling, and immunological analysis to discover broadly neutralizing antibodies against virus variants.

**Location:** `C:\Users\powen\PycharmProjects\helx\RP1_antibody_pipeline`
**Main Entry Point:** `main.py` (1,016 lines)
**Latest Update:** February 24, 2026

---

## Directory Structure

```
RP1_antibody_pipeline/
├── config.py                           # Global configuration (PipelineConfig + all sub-configs)
├── main.py                             # Pipeline orchestrator & 11-stage runner
├── requirements.txt                    # Dependencies + external tool annotations
├── __init__.py                         # Package initialization
│
├── data/                               # Data loading & genomic sequences
│   ├── bcr_loader.py                  # B-cell receptor repertoire loading
│   ├── get-virus-fasta.py             # NCBI Entrez viral sequence downloader
│   ├── SARS-CoV-2_sequences.fasta     # 10 complete SARS-CoV-2 genomes (2026)
│   ├── SARS-CoV-2_sequences.zip       # Compressed archive
│   └── oas/                           # [User-provided] OAS bulk-download CSVs
│
├── models/                             # Machine learning models
│   ├── antibody_lm.py                 # ESM2 language model wrapper
│   ├── vae.py                         # Variational Autoencoder (conformations)
│   ├── gan.py                         # Generative Adversarial Network
│   └── alm_finetuner.py               # ALM fine-tuning with MD-guided loss
│
├── md_simulations/                     # Molecular dynamics & structural modeling
│   ├── md_runner.py                   # OpenMM/GROMACS/CHARMM runners + MDTraj
│   ├── binding_md.py                  # Binding score prediction (MM/PBSA + proxy)
│   ├── structural_pathways.py         # Ag-Ab complex builder & pathway simulator
│   └── output/                        # MD trajectory outputs
│
├── msm_analysis/                       # Markov State Modeling
│   ├── msm_builder.py                 # MSM construction (PyEMMA or NumPy)
│   └── output/                        # MSM results
│
├── synthetic_evolution/                # Evolutionary simulation
│   ├── evolution.py                   # Repertoire evolution & affinity maturation
│   └── output/                        # Evolution results
│
├── viral_escape/                       # Viral mutation & immune evasion analysis
│   ├── escape_mutant.py               # Escape mutant generation
│   ├── binding_predictor.py           # Cross-reactivity scoring
│   ├── antigen_profile.py             # Epitope-ALM profiling
│   └── blind_spot.py                  # Immune blind spot detection
│
├── experiments/                        # Validation & iterative refinement
│   ├── validation.py                  # Experimental validation metrics
│   ├── lab_loop.py                    # Lab-in-the-loop active learning
│   └── output/                        # Results & reports
│       ├── blind_spot_report.json
│       ├── escape_report.json
│       ├── validation_report.json
│       ├── escape_coverage.csv
│       ├── validation_data.csv
│       ├── correlation_plot.png
│       ├── cross_reactivity_heatmap.png
│       ├── score_distribution.png
│       ├── OUTPUT_SUMMARY.md
│       └── lab_loop/
│
├── utils/                              # Shared utilities
│   └── helpers.py                      # Parallel evaluation, I/O, embeddings, logging
│
└── Docs/                               # Documentation
    ├── MEMORY.md                       # Project context & recent changes
    ├── RP1_conversation.md             # Design notes
    └── session_2026-02-24_rp1-gap-closure.md  # Session log
```

---

## Key Components

### Core Orchestration

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 315 | Centralized configuration dataclasses for all pipeline components (LM, VAE, GAN, MD, MSM, evolution, viral escape, vaccine design, lab-loop, etc.) |
| `main.py` | 1016 | Pipeline orchestrator with 11 sequential stages + mock mode for testing |

### Data Handling

| File | Purpose |
|------|---------|
| `data/bcr_loader.py` | Loads B-cell receptor sequences from OAS (public) or private CSV/FASTA files; builds `BCRSequence` objects with metadata |
| `data/get-virus-fasta.py` | Downloads viral sequences from NCBI Entrez API |
| `data/SARS-CoV-2_sequences.fasta` | 10 complete SARS-CoV-2 genomes (2026, Los Angeles County) |
| `utils/helpers.py` | Parallel map, sequence I/O, FASTA parsing, spike protein extraction, embedding utilities |

### Machine Learning Models

| File | Key Classes | Purpose |
|------|------------|---------|
| `models/antibody_lm.py` | `AntibodyLM` | ESM2 wrapper for scoring, generating, and embedding antibody sequences |
| `models/vae.py` | `AntibodyVAE` | Learns latent space of conformational states; supports checkpoint save/load |
| `models/gan.py` | `AntibodyGAN` | Generates structurally realistic antibody variants |
| `models/alm_finetuner.py` | `ALMFineTuner` | Fine-tunes LM pseudo-log-likelihood with MD-predicted binding scores |

### Molecular Dynamics & Structural Modeling

| File | Key Classes | Purpose |
|------|------------|---------|
| `md_simulations/md_runner.py` | `TrajectoryAnalyzer`, `OpenMMRunner`, `GROMACSRunner`, `CHARMMRunner` | MD backends + trajectory feature extraction |
| `md_simulations/binding_md.py` | `BindingMDPredictor`, `MMPBSACalculator` | Binding score prediction via MM/PBSA or embedding proxy |
| `md_simulations/structural_pathways.py` | `AgAbComplexBuilder`, `BindingPathwaySimulator` | Mock Ag-Ab complexes + 3-phase binding pathway MSM |

### Viral Escape & Immunology

| File | Key Classes | Purpose |
|------|------------|---------|
| `viral_escape/escape_mutant.py` | `EscapeMutant`, `EscapeMutantGenerator` | Generates panels of viral escape mutants at epitope residues |
| `viral_escape/binding_predictor.py` | `CrossReactivityScorer` | Scores antibody coverage against escape panel |
| `viral_escape/antigen_profile.py` | `AntigenBindingSiteProfiler` | Compares antigen epitopes vs ALM binding sites |
| `viral_escape/blind_spot.py` | `BlindSpotAnalyzer` | Identifies immune blind spots (poorly covered epitope regions) |

### Evolutionary & Repertoire Modeling

| File | Key Classes | Purpose |
|------|------------|---------|
| `synthetic_evolution/evolution.py` | `RepertoireEvolver`, `Antibody`, `Generation` | Simulates affinity maturation with escape awareness |
| `msm_analysis/msm_builder.py` | `MSMBuilder` | Markov State Models from MD trajectories (PyEMMA or NumPy) |

### Validation & Lab Integration

| File | Key Classes | Purpose |
|------|------------|---------|
| `experiments/validation.py` | `ValidationDataset`, `generate_report()`, `generate_escape_report()` | Experimental validation metrics & plotting |
| `experiments/lab_loop.py` | `LabInTheLoop` | Ingests wet-lab results; refines ALM; suggests next round candidates |

---

## Configuration System

The pipeline uses **dataclasses** for composable configuration (`config.py`):

- **LMConfig:** ESM2 model parameters (model name, max sequence length, GPU device, sampling parameters)
- **VAEConfig:** Variational autoencoder architecture (input/hidden/latent dims, learning rate, epochs)
- **GANConfig:** Generative adversarial network parameters
- **BCRConfig:** B-cell receptor data sources (OAS directory, private data path, max sequences, atlas output path)
- **MDConfig:** Molecular dynamics (backend selection: openmm|gromacs|charmm, PDB source, forcefield, simulation parameters)
- **MSMConfig:** Markov State Model settings (lag time, number of states, parallel jobs)
- **EvolutionConfig:** Synthetic evolution (generations, population size, mutation rate, CDR selection)
- **ViralEscapeConfig:** Escape mutant generation (antigen sequence, epitope residues, panel size, mutation limits)
- **BlindSpotConfig:** Immune blind spot analysis thresholds
- **LabLoopConfig:** Lab-in-the-loop active learning settings
- **PipelineConfig:** Master config aggregating all sub-configs

---

## Pipeline Workflow (11 Stages)

The main pipeline (`main.py`) runs sequentially through 11 stages:

```
Stage 0:  Viral Escape Panel
          └─ Generate 50 escape mutants from wildtype spike RBD

Stage 1:  BCR Repertoire
          └─ Load OAS (public) + private BCR data; build ESM2-based atlas

Stage 2:  Antibody Language Model
          └─ Generate/score candidates from seed sequences

Stage 2a: Antigen-ALM Profile
          └─ Score binding compatibility (epitope-weighted similarity)

Stage 2b: MD Binding Prediction
          └─ Physics-based (MM/PBSA) or embedding-proxy binding scores

Stage 2c: ALM Fine-tuning
          └─ Adjust LM weights using MD binding scores (pairwise margin ranking loss)

Stage 2d: Immune Blind Spot Analysis
          └─ Identify poorly covered epitope regions

Stage 3:  Structural Modelling
          └─ VAE learns conformational space; GAN generates variants

Stage 4:  MD + MSM
          └─ Run/load MD trajectories; build Markov State Models

Stage 5:  Synthetic Evolution
          └─ Simulate affinity maturation with escape awareness

Stage 6:  Repertoire Screening
          └─ Parallel LM scoring of all candidates; select top 100

Stage 7:  Cross-Reactivity
          └─ Score every candidate against escape panel (breadth of coverage)

Stage 8:  Vaccine Design
          └─ Select broadly neutralizing candidates (high coverage, high binding)

Stage 9:  Experimental Validation
          └─ Compare predictions vs. measured binding data (correlation, enrichment)

Stage 10: Lab-in-the-Loop
          └─ Ingest wet-lab results; refine ALM; recommend next candidates
```

---

## Dependency Stack

### Core Scientific Packages

- `numpy, scipy, scikit-learn` — numerical computing & machine learning
- `torch, transformers` — deep learning + ESM2 protein language model
- `mdtraj` — MD trajectory analysis
- `pandas, matplotlib, seaborn` — data handling & visualization
- `biopython` — sequence I/O, FASTA parsing, bioinformatics

### Optional External Tools

Install via conda/package manager:

- `openmm` — OpenMM molecular dynamics (default backend)
- `gromacs` — GROMACS MD engine (alternative backend)
- `charmm` — CHARMM MD engine (alternative, academic license required)
- `pyemma` — Markov State Modeling (fallback to NumPy if absent)
- `ray` — distributed parallel evaluation (optional)
- `pdbfixer` — PDB structure preparation

### Data Sources

- **OAS (Observed Antibody Space):** https://opig.stats.ox.ac.uk/webapps/oas/ — public BCR repertoires
- **RCSB PDB:** https://www.rcsb.org/ — experimental protein structures
- **AlphaFold DB:** https://alphafold.ebi.ac.uk/ — predicted structures
- **NCBI Entrez:** viral sequences via API

---

## Output Structure

Outputs are saved to `experiments/output/`:

| File | Content |
|------|---------|
| `OUTPUT_SUMMARY.md` | Comprehensive run report (inputs, antigens, epitopes, binding data) |
| `validation_report.json` | Correlation metrics (Pearson, Spearman, Kendall) |
| `escape_report.json` | Escape mutant panel coverage statistics |
| `blind_spot_report.json` | Immune blind spot analysis (per-position coverage, hard blind spots) |
| `escape_coverage.csv` | Per-candidate escape coverage (binary matrix) |
| `validation_data.csv` | Paired predicted vs. experimental binding |
| `correlation_plot.png` | Scatter plot of predictions vs. measurements |
| `cross_reactivity_heatmap.png` | Candidate × escape mutant binding matrix |
| `score_distribution.png` | Histogram of predicted binding scores |
| `lab_loop/` | Per-iteration results from lab-in-the-loop refinement |

---

## Key Design Decisions

1. **BCR Atlas:** Mean-pooled ESM2 embeddings from OAS repertoire → disease-specific centroid + standard deviation reference for coverage analysis

2. **Antigen Profiling:** Epitope-weighted cosine similarity between antigen and antibody embedding space

3. **MD Binding Proxy:** L2 distance on unit sphere maps to [0,1] binding score; falls back from MM/PBSA → embedding proxy if OpenMM unavailable

4. **ALM Fine-tuning:** Pairwise margin ranking loss on pseudo-log-likelihoods; only FFN layers updated (attention frozen)

5. **Evolution Mechanism:** CDR-focused mutation (higher mutation rate in CDR regions); selection based on binding + escape evasion

6. **MSM Integration:** Structural pathways (3-phase approach: approach → bound → separation) as separate MSM; main MSM built from MD trajectory features

7. **Checkpointing:** VAE supports save/load for reproducibility; models persisted to `MODELS_DIR`

---

## Execution Modes

### Mock Mode (Fast Testing)

```bash
python -m RP1_antibody_pipeline.main --mock
```

- Uses synthetic BCR repertoire (20 random sequences)
- Uses mock MD trajectories (Gaussian + sinusoidal)
- Runs in ~5 seconds
- No GPU, no large model downloads

### Full Mode (Production)

```bash
python -m RP1_antibody_pipeline.main
```

- Loads real OAS data (if CSV files present in `data/oas/`)
- Runs ESM2 (650M parameters) on GPU
- Executes MD simulations (OpenMM, GROMACS, or CHARMM)
- Builds true Markov State Models
- Requires ~30+ minutes on GPU-accelerated hardware

---

## Summary

The **RP1_antibody_pipeline** is a production-grade computational immunology platform that combines:

- **Sequence-based ML:** ESM2 language models for antibody generation & scoring
- **Physics-based MD:** OpenMM/GROMACS/CHARMM for binding simulation
- **Generative models:** VAEs & GANs for conformational space exploration
- **Statistical modeling:** Markov State Models for kinetic analysis
- **Evolutionary algorithms:** Synthetic affinity maturation with escape awareness
- **Immunological analysis:** Viral escape prediction, blind spot detection, vaccine design
- **Experimental integration:** Lab-in-the-loop active learning for iterative refinement

It is designed to predict which antibodies from a human BCR repertoire will effectively counter viral escape variants, with applications to pandemic preparedness and vaccine design.

---

## Documentation Files

| File | Purpose |
|------|---------|
| `Docs/MEMORY.md` | Project context, recent changes, design decisions |
| `Docs/RP1_conversation.md` | Design notes and conversation history |
| `Docs/session_2026-02-24_rp1-gap-closure.md` | Session log detailing gap fixes |
| `Docs/RP1_summary.md` | This comprehensive summary document |
