# RP1: Predicting Antibody Responses to Viral Escape Mutants
**Conversation Log** — 2026-02-24

---

## Research Project Overview

**RP1: Predicting Antibody Responses to Viral Escape Mutants**

Pipeline: Molecular Dynamics (MD) ⮕ Antibody LMs ⮕ Experimental validation

- Predict how immune system might adapt (or struggle) when viruses mutate to escape detection
- Faster assessment of new viruses and potential impact
- Guide vaccine design to cover both current and future viral variants
- Predict how well vaccines might work against seasonal or new threats

---

## ChatGPT Exchange: Acronyms and Topics in Molecular Dynamics Context

### Question
> RP1: explain the following acronyms and topics within the context of molecular dynamics:
> MSM, VAE, GAN, synthetic ag-ab structures and pathways, repertoire scale, synthetic repertoire evolution, antibody LMs and experimental validation

---

### 1. MSM — Markov State Models

A statistical framework used to model long-timescale molecular kinetics from many short MD simulations.

**In molecular dynamics:**
- MD simulations generate time-ordered atomic coordinates
- MSMs discretize conformational space into states
- Transition probabilities between states are estimated
- Assumes the Markov property (future depends only on current state, not history)

**Why it matters for antibodies:**
- Antibody binding involves rare conformational transitions (e.g., CDR loop rearrangements)
- MSMs allow reconstruction of: binding pathways, conformational switching, on/off rates, metastable states
- Enables prediction of binding kinetics, not just structures

**In Ag–Ab systems:**
- Can identify pre-binding states
- Capture induced-fit vs conformational selection mechanisms
- Estimate free energy landscapes

---

### 2. VAE — Variational Autoencoder

A probabilistic generative model that learns a continuous latent representation of data.

**In molecular dynamics:**
- Used to: learn low-dimensional representations of protein conformations, identify slow collective variables, generate new conformations

**In antibody research:**
- Learn latent embeddings of: antibody sequences, CDR loop structures, binding conformations
- Can generate: novel antibody structures, synthetic conformational states, binding intermediates

**Why useful:**
- Smooth latent space enables interpolation between conformations
- Helps explore conformational landscapes beyond MD sampling limits

---

### 3. GAN — Generative Adversarial Network

A generative model with: Generator (creates samples) + Discriminator (evaluates realism). They compete during training.

**In molecular dynamics / structural biology:**
- Generate: protein conformations, contact maps, antibody structures
- Learn realistic structural distributions

**In Ag–Ab systems:**
- Generate synthetic antibody-antigen complexes
- Produce candidate binding poses
- Augment structural datasets

---

### 4. Synthetic Ag–Ab Structures and Pathways

Artificially generated: antibody–antigen complex structures, binding/unbinding pathways, conformational trajectories.

**Generated via:** MD simulations, enhanced sampling (metadynamics, umbrella sampling), VAEs/GANs, diffusion models.

**Why important:**
- Experimental structures (PDB) are sparse
- Synthetic generation allows exploration of hypothetical mutations, mapping binding landscapes, simulating immune evolution, training ML models on expanded datasets

---

### 5. Repertoire Scale

Refers to modeling at the scale of the entire antibody repertoire (~10⁸–10¹² possible antibodies).

**In computational MD context:**
- Instead of studying one antibody–antigen pair, you study thousands to millions of sequences, structural ensembles, fitness landscapes

**Involves:**
- Sequence embeddings (LMs), structural prediction (AlphaFold-like), high-throughput docking, reduced-order MD approximations, MSMs across many variants

---

### 6. Synthetic Repertoire Evolution

Computational simulation of antibody evolution processes: somatic hypermutation, affinity maturation, selection pressure.

**In MD context:**
- Mutate antibody sequences
- Predict structure changes
- Simulate binding energetics
- Evaluate selection fitness

**May involve:** MD-based free energy calculations, ML-predicted binding affinities, evolutionary algorithms.

**Purpose:** Design antibodies in silico, predict escape mutations, model immune response dynamics.

---

### 7. Antibody LMs — Antibody Language Models

Transformer-based models trained on large antibody sequence datasets.

**Examples:** AbLang, AntiBERTa, ESM variants, proprietary foundation models.

**They learn:** sequence grammar, structural constraints, evolutionary relationships, CDR variability patterns.

**In MD and structural context:**
- Predict mutation effects
- Guide MD simulations
- Provide embeddings for MSM state clustering
- Score synthetic sequences
- Generate novel antibody sequences

---

### 8. Experimental Validation

After computational modeling:

| Type | Methods |
|------|---------|
| Structural | X-ray crystallography, Cryo-EM, NMR |
| Binding | SPR, BLI, ITC |
| Functional | Neutralization assays, ELISA, Cell-based assays |
| Stability | DSC, Aggregation assays |

---

### Conceptual Stack (Big Picture)

```
Sequence space modeling   → Antibody LMs
Structure space modeling  → MD, VAE, GAN
Kinetics modeling         → MSM
Evolution modeling        → Synthetic repertoire evolution
Population scale          → Repertoire scale
Reality check             → Experimental validation
```

---

## Computational Pipeline Diagram

*(Generated by ChatGPT — Infographic version)*

Stages:
```
[Seed Sequences]
      ↓
[Antibody LM] ──────────────────────────────────────── Score & generate mutations
      ↓
[Structure Prediction: VAE + GAN] ───────────────────── Latent conformational space
      ↓
[MD Simulations + MSM] ──────────────────────────────── Kinetics & binding pathways
      ↓
[Synthetic Repertoire Evolution] ────────────────────── Affinity maturation simulation
      ↓
[Repertoire-Scale Screening] ────────────────────────── Parallel evaluation
      ↓
[Experimental Validation] ───────────────────────────── SPR, BLI, ELISA, Cryo-EM
```

---

## Python Framework (High-Level)

### Project Structure
```
RP1_antibody_pipeline/
├── data/
├── models/
├── md_simulations/
├── msm_analysis/
├── synthetic_evolution/
├── viral_escape/          ← Added for RP1
├── experiments/
├── utils/
├── config.py
└── main.py
```

### Key Libraries
- `transformers`, `torch` — ESM2 antibody language model
- `mdtraj`, `openmm` — molecular dynamics
- `pyemma` — Markov State Models
- `numpy`, `scipy`, `scikit-learn` — numerics
- `matplotlib`, `seaborn` — visualization
- `pandas` — data I/O

---

## Claude Code Implementation Session

### Gap Analysis Against RP1

The existing codebase was a general antibody discovery pipeline missing the core RP1 capability:
the **antigen (virus) side**. Specifically absent:

| Missing Component | Description |
|---|---|
| Viral escape mutant generation | No model of antigen mutation to evade antibodies |
| Cross-reactivity scoring | No antibody-vs-escape-variant binding prediction |
| Immune adaptation prediction | No answer to "how will immune system cope?" |
| Vaccine breadth analysis | No broadly-neutralising candidate selection |
| Coverage metrics | No escape panel coverage / breadth scoring |

### Changes Made

#### New: `viral_escape/escape_mutant.py`
- `EscapeMutant` dataclass — wildtype/mutant sequences, mutation list, escape score
- `EscapeMutantGenerator` — generates panels of escape variants
  - Single-residue sweep at epitope positions
  - Random double/triple mutants to fill panel
  - `score_escape()` — fraction of epitope disrupted (0–1)
  - `mutation_hotspots()` — physicochemical entropy ranking
  - `generate_known_variants()` — plug in Alpha/Delta/Omicron mutation tables

#### New: `viral_escape/binding_predictor.py`
- `CrossReactivityScorer` — predicts binding across escape panels
  - `build_coverage_matrix()` — efficient outer-product (n_ab × n_var) matrix
  - `epitope_similarity()` — fraction of antibody-contact residues unchanged
  - `delta_affinity()` — ΔΔG proxy when antigen mutates
  - `vaccine_candidates()` — selects broadly neutralising candidates
  - `predict_immune_adaptation()` — answers the RP1 question directly

#### Updated: `config.py`
```python
ViralEscapeConfig:
  antigen_sequence    # wildtype viral antigen (replace with real FASTA)
  epitope_residues    # 0-indexed antibody-contact positions
  panel_size          # number of escape mutants (default 50)
  max_mutations       # max simultaneous mutations per variant (default 3)
  binding_threshold   # score cutoff for "covered" (default 0.5)

VaccineDesignConfig:
  min_coverage_fraction  # must cover ≥60% of panel
  top_candidates         # max candidates to report (default 20)
```

#### Updated: `main.py` — Pipeline now has 9 stages

| Stage | Name | Purpose |
|---|---|---|
| 0 | `stage_escape_panel` | Generate viral escape mutant panel |
| 1 | `stage_lm` | Antibody LM scoring/generation |
| 2 | `stage_structure` | VAE + GAN conformational space |
| 3 | `stage_msm` | MD trajectory + MSM kinetics |
| 4 | `stage_evolution` | **Escape-aware** affinity maturation |
| 5 | `stage_repertoire_screen` | Parallel screening |
| 6 | `stage_cross_reactivity` | Score every candidate vs. escape panel |
| 7 | `stage_vaccine_design` | Select broadly neutralising candidates |
| 8 | `stage_validation` | Experimental comparison + escape report |

**Key design: escape-aware scorer in evolution**
```python
blended_score = alpha * LM_fitness + (1 - alpha) * coverage_fraction
# alpha=0.7 — evolution pressure toward broadly cross-reactive antibodies
```

#### Updated: `experiments/validation.py`
- `escape_coverage_score()` — per-antibody panel coverage fraction
- `breadth_score()` — mean coverage across all antibodies
- `plot_cross_reactivity_heatmap()` — seaborn heatmap (antibody × variant)
- `generate_escape_report()` — outputs:
  - `cross_reactivity_heatmap.png`
  - `escape_coverage.csv`
  - `escape_report.json` (breadth score, broadly neutralising %, most vulnerable variants)

### Run Commands
```bash
# Fast mock run (no GPU, no large model downloads)
python -m RP1_antibody_pipeline.main --mock

# Full run
python -m RP1_antibody_pipeline.main

# Custom seeds from CSV
python -m RP1_antibody_pipeline.main --seeds-csv my_antibodies.csv
```

### Output Files
```
experiments/output/
  validation_report.json         # standard binding correlation metrics
  correlation_plot.png
  score_distribution.png
  escape_report.json             # RP1: immune adaptation summary
  cross_reactivity_heatmap.png   # RP1: antibody × variant binding heatmap
  escape_coverage.csv            # RP1: per-antibody coverage fractions
```

---

---

## Claude Code Session — 2026-02-24 (Continued)

### Session Goals

Extend the RP1 pipeline with five new capabilities:

1. Compare pathogen antigen sequences against ALM binding site representations
2. Use MD to predict antibody–antigen binding scores
3. Feed MD results into the ALM to fine-tune the model
4. Predict immune blind spots in typical human BCR repertoires
5. Laboratory-in-the-loop validation for iterative model refinement

Also: use real SARS-CoV-2 genomic FASTA data (`data/SARS-CoV-2_sequences.fasta`) as the antigen source.

---

### FASTA Data

`data/SARS-CoV-2_sequences.fasta` contains **10 complete SARS-CoV-2 genomes** (2026, USA/California submissions). These are used as the antigen panel — spike protein sequences are extracted from each record.

**Spike extraction challenge:** The 2026 genomes are 184–431 nt shorter than the Wuhan reference (due to N-gap regions and partial assemblies), so fixed reference coordinates (nt 21562–25384) fail.

**Solution (two-stage extraction in `utils/helpers.py`):**
1. Try reference slice (nt 21562–25384) — fast path for clean genomes
2. If that fails, run `_find_spike_orf()` — scans all 3 reading frames for an ORF > 800 aa containing the conserved RBD motif `GVYYPDKVFR`
3. If that also fails, fall back to a hardcoded RBD stub

---

### New Files Created

#### `data/bcr_loader.py`
BCR repertoire loading from OAS (Observed Antibody Space) and private sources.

| Class | Role |
|---|---|
| `BCRSequence` | Dataclass: sequence, CDR3, V-gene, isotype, disease label |
| `BCRRepertoire` | Collection with `get_sequences()`, `get_cdr3s()`, `filter_by_length()`, `summary()` |
| `OASLoader` | Reads AIRR-Community standard CSV columns; `load()`, `load_directory()` |
| `PrivateBCRLoader` | Reads CSV or FASTA (biopython SeqIO) |
| `load_repertoire()` | Factory: dispatches on `source='oas'\|'csv'\|'fasta'` |

#### `viral_escape/antigen_profile.py`
Compares pathogen antigen sequences against ALM binding site representations.

| Class/Method | Role |
|---|---|
| `AntigenBindingSiteProfiler` | Core profiler |
| `profile_antigen_epitope()` | Epitope-weighted antigen embedding (400 aa window around epitope centre) |
| `profile_antibody_cdr()` | CDR-weighted antibody embedding (CDR-H1/H2/H3: positions 24–34, 52–56, 93–102) |
| `binding_compatibility()` | Cosine similarity scalar → [0,1] affinity |
| `build_affinity_matrix()` | Full (n_antibody × n_antigen) similarity matrix via batched outer product |

#### `md_simulations/binding_md.py`
MD-based antibody–antigen binding prediction.

| Class/Method | Role |
|---|---|
| `BindingMDPredictor` | Binding predictor using embedding geometry as MD proxy |
| `_embedding_proxy()` | L2 on unit sphere → `1 − dist/2` → [0,1] |
| `_md_interface_energy()` | Proxy + LJ-inspired repulsion penalty for near-perfect alignment |
| `predict_binding_scores()` | Full (n_antibody × n_antigen) binding score matrix |
| `top_pairs()` | Ranked list of best antibody–antigen pairs |

#### `models/alm_finetuner.py`
Feeds MD binding results back into the ALM via fine-tuning.

| Class/Method | Role |
|---|---|
| `ALMFineTuner` | Manages mock and real fine-tuning |
| `_mock_finetune()` | Pairwise hinge loss updating `_score_offsets` dict (no GPU) |
| `_real_finetune()` | Freezes attention weights; updates FFN via AdamW + `MarginRankingLoss` |
| `_pll_tensor()` | Simultaneous masked forward pass for differentiable pseudo-log-likelihood |
| `score_with_finetuning()` | Returns PLL scores offset by fine-tuning adjustments |

#### `viral_escape/blind_spot.py`
Predicts immune blind spots in typical human BCR repertoires.

| Class/Method | Role |
|---|---|
| `BlindSpotAnalyzer` | Core analyzer; pre-normalises atlas embeddings once for fast cosine lookups |
| `coverage_score()` | Max cosine similarity between antigen epitope embedding and any antibody in BCR atlas → [0,1] |
| `blind_spot_score()` | `1 − coverage_score` |
| `per_position_coverage()` | Mean coverage across all 19 single-AA substitutions at each epitope position |
| `hard_blind_spot_positions()` | Positions where ALL 20 AA substitutions remain below threshold |
| `analyze()` | Full panel analysis; saves JSON report; sets `repertoire_at_risk` if mean blind-spot > 0.5 |
| `summary_string()` | Human-readable one-paragraph summary |

#### `experiments/lab_loop.py`
Laboratory-in-the-loop iterative model refinement.

| Class/Method | Role |
|---|---|
| `ExperimentalResult` | Dataclass: sequence, measured_binding, assay_type, notes |
| `LabInTheLoop` | Manages iteration loop |
| `run_iteration()` | Full cycle: ingest → correlate → refine → re-score → suggest → update escape panel → save report |
| `ingest_experimental_data()` | Reads CSV with `sequence` + `measured_binding` columns |
| `refine_model()` | Normalises experimental scores → re-runs `ALMFineTuner.finetune()` |
| `suggest_next_experiments()` | Active learning: 70% uncertainty sampling + 30% length diversity |
| `_generate_mock_data()` | Correlated Gaussian noise on predicted scores (for mock mode) |
| `_update_escape_panel()` | Sequences with binding < `escape_threshold` added as confirmed `EscapeMutant` objects |

---

### Files Updated

#### `config.py`
New configuration dataclasses added:

```python
BCRConfig          # OAS data dir, private data path, disease label, atlas output
AntigenALMConfig   # n_antigen_sequences, similarity_metric
ALMFinetuneConfig  # learning_rate, n_epochs, margin, min_binding_score
BlindSpotConfig    # blind_spot_threshold, output_path
LabLoopConfig      # experimental_csv, n_suggestions, escape_threshold, output_dir
MDConfig           # added: backend (openmm|gromacs|charmm), pdb_source, forcefield
```

All added to `PipelineConfig`.

#### `models/antibody_lm.py`
- `build_atlas()` — batched embedding → `{disease, centroid, std, embeddings, n_sequences}` dict
- `atlas_similarity()` — cosine sim between query and atlas centroid → [0,1]
- Mock equivalents added to `RandomAntibodyLM`

#### `md_simulations/md_runner.py`
- `GROMACSRunner` — full gmx workflow: pdb2gmx → editconf → solvate → grompp → mdrun; NVT MDP template
- `CHARMMRunner` — subprocess-based; generates CHARMM input script; academic licence required
- `get_md_runner(backend, ...)` — factory returning correct runner

#### `utils/helpers.py`
- Added `_CODON_TABLE`, `_translate()`, `_parse_fasta_first()`, `_find_spike_orf()`
- Updated `load_spike_from_fasta()`: biopython SeqIO → reference coords → ORF scan → hardcoded fallback
- Added `load_all_spike_sequences_from_fasta()`: extracts spike from all records in FASTA
- Added `_iter_fasta_records()`: pure-Python fallback FASTA iterator (no biopython dependency)
- **Bug fix:** `parallel_map()` now checks if `fn` is picklable before spawning a process pool — falls back to serial execution for local closures (fixes `AttributeError: Can't get local object '..._eval'`)

#### `main.py`
Expanded from 9 stages to **11 stages**:

| Stage | Name | New? | Purpose |
|---|---|---|---|
| 0 | `stage_escape_panel` | | Viral escape mutant generation |
| 1 | `stage_bcr_repertoire` | **NEW** | Load OAS/private BCR data → build disease atlas |
| 2 | `stage_lm` | | Antibody LM scoring/generation |
| 2a | `stage_antigen_alm_profile` | **NEW** | Antigen vs ALM binding site affinity matrix |
| 2b | `stage_md_binding_prediction` | **NEW** | MD proxy binding score matrix |
| 2c | `stage_alm_finetune` | **NEW** | Re-fine-tune ALM with MD binding scores |
| 2d | `stage_blind_spot_analysis` | **NEW** | Identify immune blind spots in BCR atlas |
| 3 | `stage_structure` | | VAE + GAN conformational space |
| 4 | `stage_msm` | | MD trajectory + MSM kinetics |
| 5 | `stage_evolution` | | Escape-aware affinity maturation |
| 6 | `stage_repertoire_screen` | | Parallel screening |
| 7 | `stage_cross_reactivity` | | Coverage matrix (n_ab × n_var) |
| 8 | `stage_vaccine_design` | | Broadly neutralising candidate selection |
| 9 | `stage_validation` | | Experimental comparison + escape report |
| 10 | `stage_lab_loop` | **NEW** | Lab-in-the-loop iterative refinement |

Fine-tuned scores from stage 2c are propagated downstream — `score_map` is updated so stages 5–9 use the ALM-refined scores rather than raw LM scores.

---

### Mock Run Results

```
Stage 0  (Escape panel)        10 escape mutants generated
Stage 1  (BCR repertoire)      20 mock sequences; atlas built (64-dim)
Stage 2  (LM scoring)          873 unique sequences
Stage 2a (Antigen-ALM)         873×2 affinity matrix; mean=0.499
Stage 2b (MD binding)          873×2 binding matrix; mean=0.293
Stage 2c (ALM fine-tune)       352 sequences; 3 epochs; loss=0.100
Stage 2d (Blind spot)          mean_score=0.373; at_risk=False
Stage 3  (VAE/GAN)             latent shape (100,64)
Stage 4  (MD+MSM)              timescales [313, 66.7, 47.4, 34.2, 30.2]
Stage 5  (Evolution)           38 candidates; diversity=0.227
Stage 6  (Repertoire screen)   [FIXED THIS SESSION — was crashing on closure pickling]
```

---

### Bug Fixed This Session

**`AttributeError: Can't get local object 'stage_repertoire_screen.<locals>._eval'`**

Python's `multiprocessing.Pool` cannot pickle locally-defined closure functions. Stage 6 defined `_eval` as a local closure inside `stage_repertoire_screen` and passed it to `parallel_map`.

**Fix (`utils/helpers.py`):** Added a `pickle.dumps(fn)` probe before spawning the pool. If the function is not picklable, `parallel_map` silently falls back to serial execution.

```python
try:
    pickle.dumps(fn)
except Exception:
    logger.debug("fn is not picklable — falling back to serial execution.")
    return [fn(item) for item in items]
```

This is a general solution — any future caller passing a closure will automatically fall back to serial mode without crashing.

---

*End of session — 2026-02-24*

---

## Spec vs. Implementation Review — 2026-02-24

Full review of the RP1 specification against what has been built.

---

### Core Concepts

| Spec Item | Status | Notes |
|---|---|---|
| MSM | Complete | `msm_builder.py` — raw trajectories → TICA → KMeans microstates → macrostates; PyEMMA or NumPy fallback |
| VAE | Complete | `models/vae.py` — PyTorch encoder/decoder, latent space sampling |
| GAN | Complete | `models/gan.py` — generator + discriminator, structural augmentation |
| Synthetic ag-ab structures & pathways | Partial | Embeddings and binding scores computed; no 3D complex generation or binding pathway simulation (needs AlphaFold-Multimer + enhanced MD sampling) |
| Physics-based fine-tuning | Gap | ALM fine-tuned with pairwise ranking loss using embedding distances — not true physics. Real physics-based fine-tuning would use ΔG from MM-PBSA or FEP as the training label |
| Repertoire scale | Complete | `stage_repertoire_screen` — parallel scoring of full candidate pool |
| ALM embedding | Complete | ESM2 mean-pool embeddings; mock fallback; atlas construction |
| Synthetic repertoire evolution | Complete | `evolution.py` — CDR-focused mutation, escape-aware fitness, generational selection |

---

### Use Case: Assessing a New Virus

| Step | Status | Module | Notes |
|---|---|---|---|
| Get pathogen antigen sequences | Complete | `utils/helpers.py` | FASTA extraction with ORF-scan fallback for 2026 genomes |
| Compare against ALM binding sites | Complete | `viral_escape/antigen_profile.py` | Epitope-weighted antigen embedding vs CDR-weighted antibody embedding; full affinity matrix |
| MD to predict binding | Partial | `md_simulations/binding_md.py` | Embedding L2 proxy works end-to-end; `_md_interface_energy()` is a stub — real MM-PBSA not implemented |
| Feed MD results into ALM fine-tuning | Complete | `models/alm_finetuner.py` | MD binding scores → normalised ranking signal → AdamW on FFN layers (attention frozen) |
| Predict immune blind spots | Complete | `viral_escape/blind_spot.py` | Per-position substitution coverage; hard blind spots; `repertoire_at_risk` flag |
| Lab-in-the-loop validation | Complete | `experiments/lab_loop.py` | Ingest CSV → re-fine-tune → uncertainty sampling → escape panel update |

---

### Molecular Dynamics Block

| Spec Item | Status | Notes |
|---|---|---|
| Input: PDBs | Configured, not automated | `MDConfig.pdb_source` points to a local file; no automatic fetch from RCSB or AlphaFold DB |
| Where to get PDBs | Not documented | RCSB PDB, AlphaFold DB, ESMFold — not referenced in code or docs |
| Output: MD traces | Complete | `.dcd` (OpenMM/CHARMM) or `.xtc` (GROMACS) written to `md_simulations/output/` |
| Microstates → Macrostates (MSM) | Complete | TICA/PCA → KMeans → NumPy or PyEMMA MSM |
| Energy functions & MD codebase | Partial | OpenMM/GROMACS/CHARMM wrappers implemented; custom energy functions absent; forcefield config exposed |
| VAE PyTorch checkpoints | Gap | VAE trains each run from scratch; `torch.save`/`torch.load` not implemented — models are not persistent across runs |
| MD software: CHARMM, GROMACS, PyTorch | Complete | All three wrappers in `md_runner.py`; documented in `requirements.txt` |

---

### Antibody LLMs Block

| Spec Item | Status | Notes |
|---|---|---|
| Input: public BCR repertoires (OAS) | Complete | `data/bcr_loader.py` — OASLoader reads AIRR-Community CSV |
| Where to get OAS data | Partially documented | URL in `requirements.txt` comment only; not surfaced in config or startup logging |
| Input: private BCR repertoire data | Complete | `BCRConfig.private_data_path` — accepts CSV or FASTA |
| Output: disease-specific atlas | Complete | `build_atlas()` → centroid, std, full embeddings dict; saved to `data/atlas.pkl` |
| PLM PyTorch checkpoints | Complete | ESM2 via HuggingFace `transformers`; model cached locally after first download |

---

### Experimental Validation Block

| Spec Item | Status | Notes |
|---|---|---|
| Output: experimental readouts | Complete | Pearson r, Spearman ρ, ROC-AUC, top-N recall, RMSE; correlation + distribution plots |
| RP1 escape report | Complete | Breadth score, cross-reactivity heatmap, escape coverage CSV, escape report JSON |

---

### Identified Gaps (Priority Order)

**1. Physics-based fine-tuning signal (High)**
The spec calls out physics-based fine-tuning. Currently the signal is embedding distance — a geometric proxy. To be physics-based, ΔG estimates from MD (MM-PBSA or FEP) need to replace or augment this as the ranking label fed to `alm_finetuner.py`.

**2. Synthetic ag-ab structural pathways (High)**
The spec lists "synthetic ag-ab structures and pathways" as a core concept. Currently only sequence-level embeddings exist. Missing: AlphaFold-Multimer complex generation → MD binding/unbinding pathway simulation (metadynamics or steered MD) → pathway featurisation into MSM.

**3. VAE checkpoint persistence (Medium)**
VAE trains from scratch each run. `torch.save` / `torch.load` checkpoint support needs to be added so trained conformational models are reusable.

**4. PDB data acquisition guidance (Medium)**
The spec notes "(where can other users get this data)" for PDBs. Entirely absent. `MDConfig` docstring and/or a README section should reference RCSB PDB (`rcsb.org`), AlphaFold DB, and ESMFold.

**5. OAS data acquisition guidance (Low)**
OAS URL exists only in a `requirements.txt` comment. Should be surfaced in `BCRConfig` docstring and pipeline startup logging.

---

*End of review — 2026-02-24*

---

## Claude Code Session — 2026-02-24 (Gap Closure)

### Session Goal

Implement all five gaps identified in the spec-vs-implementation review above.

---

### Gap 1 (High) — Physics-Based Fine-Tuning Signal

**Problem:** `_md_interface_energy()` was a stub using heuristic embedding repulsion. The fine-tuning signal fed to `alm_finetuner.py` was not physics-based.

**Solution:** Added `MMPBSACalculator` class to `md_simulations/binding_md.py`.

| Component | Detail |
|---|---|
| `compute_interaction_energy(pdb_path)` | Loads complex PDB → OpenMM energy minimisation (AMBER14, in-vacuo, `NoCutoff`) → ΔE = E_complex − E_receptor − E_ligand using `Modeller.delete()` to isolate each chain |
| `energy_to_score(ΔE)` | Sigmoid `1 / (1 + exp(ΔE / kT))` → [0,1]; negative ΔE (favourable) maps to score > 0.5 |
| `score_pair(ab_seq, ag_seq, pdb_path, ab_emb, ag_emb)` | Physics path when PDB exists; embedding L2-distance proxy fallback otherwise |
| `_md_interface_energy()` rewrite | Checks OpenMM availability + `pdb_source`; logs clear reason for each fallback path; calls `MMPBSACalculator` in physics path |
| `BindingMDPredictor.__init__` | Gained optional `pdb_source: Optional[str] = None` kwarg |

Auto-fallback hierarchy: OpenMM unavailable → no pdb_source → PDB file missing → embedding proxy.

**Files changed:** `md_simulations/binding_md.py`

---

### Gap 2 (High) — Synthetic Ag-Ab Structural Pathways

**Problem:** The spec listed "synthetic ag-ab structures and pathways" as a core concept. Only sequence embeddings existed — no 3D complexes, no pathway simulation.

**Solution:** New file `md_simulations/structural_pathways.py` + new pipeline stage 2.5.

#### `AgAbComplexBuilder`

| Method | Detail |
|---|---|
| `build_mock_complex(ab_seq, ag_seq)` | Returns dict: `inter_chain_distances` (n,), `contact_map` (n_ab×n_ag), `backbone_angles` (n,2), `n_contacts`. Seeded RNG from sequence hash for reproducibility. |
| `build_from_pdb(pdb_path, ab_chain=0, ag_chain=1)` | Same dict from real MDTraj coordinates. Uses `chainid {n}` DSL selector for Cα atoms. |
| `batch_build(ab_seqs, ag_seqs, mock=True, pdb_paths=None)` | Cross-calls mock or real builder per pair; PDB paths override mock flag when valid. |

#### `BindingPathwaySimulator`

| Method | Detail |
|---|---|
| `simulate_mock_pathway(complex_features, n_frames=200)` | Three-phase Gaussian envelope: approach (d×1.5) → bound (d×0.7) → separation. Adds thermal noise + slow sinusoidal mode for MSM detection. Returns `(n_frames, n_features)`. |
| `simulate_steered_md(pdb_path, pull_residues, n_steps, output_dir)` | OpenMM `CustomExternalForce` with linearly increasing pull fraction along Ab–Ag separation vector; runs in 20 segments; outputs DCD trajectory. |
| `featurize_pathway(trajectory)` | Appends per-frame mean_dist, std_dist, mean_vel to raw features → `(n_frames, n_raw+3)` for TICA/MSM input. |

#### `stage_structural_pathways(cfg, top_seqs, antigen_seqs, lm, mock)`

- Inserted in `run_pipeline()` between stages 2b and 2c (labelled **stage 2.5**)
- Builds n_ab×n_ag cross-product pairs (5×2 mock, 20×5 real)
- Runs mock pathway simulation (100 frames mock, 200 real)
- Featurizes + stacks → combined trajectory → separate pathway-level `MSMBuilder`
  (lag = `cfg.msm.lag_time // 2`, n_states ≤ 10)
- Returns `pathway_msm`, `complex_features`, `pathway_timescales`, `free_energy`
- Added to `run_pipeline()` return dict as `"pathway_result"`

**Files changed:** `md_simulations/structural_pathways.py` (new), `main.py`

---

### Gap 3 (Medium) — VAE Checkpoint Persistence

**Problem:** `AntibodyVAE.fit()` trained from scratch every run. No model persistence.

**Solution:** Added checkpoint methods to `AntibodyVAE`; updated `stage_structure()`.

| Component | Detail |
|---|---|
| `save_checkpoint(path)` | `torch.save({model_state_dict, input_dim, hidden_dim, latent_dim, beta})`. `input_dim` recovered from `encoder.net[0].in_features`. |
| `load_checkpoint(path, device)` (classmethod) | Reconstructs architecture from saved config dict, loads state dict, moves to device. |
| `stage_structure()` update | Checks `MODELS_DIR/vae_checkpoint.pt` on entry (non-mock only); loads if present; trains from scratch otherwise; saves checkpoint after training (non-mock only). |

**Files changed:** `models/vae.py`, `main.py`

---

### Gap 4 (Medium) — PDB Data Acquisition Guidance

**Problem:** `MDConfig` docstring was brief. `stage_msm()` gave no guidance when falling back to mock trajectory.

**Solution:**

- **`MDConfig` docstring** expanded with:
  - RCSB PDB — experimental structures (`https://www.rcsb.org/`)
  - AlphaFold DB — pre-computed structures for >200M proteins (`https://alphafold.ebi.ac.uk/`)
  - ESMFold — fast single-sequence de-novo prediction (API + `pip install fair-esm`)
  - AlphaFold2-Multimer — Ag-Ab complex prediction
  - PDBFixer — structure preparation/protonation (`pip install pdbfixer`)
- **`stage_msm()`** — added `logger.info` with RCSB + AlphaFold URLs and PDBFixer advice when `not mock` and MD files are absent

**Files changed:** `config.py`, `main.py`

---

### Gap 5 (Low) — OAS Data Acquisition Guidance

**Problem:** OAS URL existed only in a `requirements.txt` comment. Not surfaced in config or startup logging.

**Solution:**

- **`BCRConfig` docstring** expanded with step-by-step OAS bulk download instructions: URL, species/study/isotype selection, gzip decompression command, expected CSV columns
- **`stage_bcr_repertoire()`** — added `logger.info` pointing to OAS URL when OAS directory is missing (non-mock only)

**Files changed:** `config.py`, `main.py`

---

### Updated Pipeline Stage Table

| Stage | Name | Purpose |
|---|---|---|
| 0 | `stage_escape_panel` | Viral escape mutant panel from wildtype antigen |
| 1 | `stage_bcr_repertoire` | OAS + private BCR loading; build disease atlas |
| 2 | `stage_lm` | Antibody LM scoring/generation (ESM2) |
| 2a | `stage_antigen_alm_profile` | Spike variants vs ALM binding sites |
| 2b | `stage_md_binding_prediction` | Embedding proxy or MM/PBSA binding scores |
| **2.5** | **`stage_structural_pathways`** | **Mock Ag-Ab complexes + pathway MSM** *(new)* |
| 2c | `stage_alm_finetune` | Fine-tune LM FFN layers on MD binding scores |
| 2d | `stage_blind_spot_analysis` | Immune blind spots: atlas vs antigen coverage |
| 3 | `stage_structure` | VAE + GAN conformational space *(checkpoint-aware)* |
| 4 | `stage_msm` | MD trajectory loading + conformational MSM |
| 5 | `stage_evolution` | Escape-aware affinity maturation |
| 6 | `stage_repertoire_screen` | Parallel LM scoring, top candidates |
| 7 | `stage_cross_reactivity` | Coverage matrix (n_ab × n_variants) |
| 8 | `stage_vaccine_design` | Broadly neutralising candidate selection |
| 9 | `stage_validation` | Experimental comparison + escape report |
| 10 | `stage_lab_loop` | Lab-in-the-loop iterative refinement |

---

### Mock Run Results (Post Gap-Closure)

```
Stage 0  (Escape panel)          10 escape mutants generated
Stage 1  (BCR repertoire)        20 mock sequences; atlas built (64-dim)
Stage 2  (LM scoring)            873 unique sequences
Stage 2a (Antigen-ALM)           873×2 affinity matrix; mean=0.499
Stage 2b (MD binding)            873×2 binding matrix; mean=0.293
Stage 2.5 (Structural pathways)  10 complexes (5 Ab × 2 Ag);
                                  Pathway MSM timescales ≈ [341, 174, 134]
Stage 2c (ALM fine-tune)         352 sequences; 3 epochs; loss=0.100
Stage 2d (Blind spot)            mean_score=0.379; at_risk=False
Stage 3  (VAE/GAN)               latent shape (100,64); checkpoint skipped (mock)
Stage 4  (MD+MSM)                timescales [313, 66.7, 41.7, 36.6, 30.1]
Stage 5  (Evolution)             38 candidates; diversity=0.227; best=0.980
Stage 6  (Repertoire screen)     38/38 candidates retained
Stage 7  (Cross-reactivity)      100% broadly neutralising; coverage=1.000
Stage 8  (Vaccine design)        20 candidates; best coverage=100%
Stage 9  (Validation)            pearson_r=0.297; roc_auc=0.648
Stage 10 (Lab loop)              r_before=0.113; r_after=0.120; new_escape=11
```

Total wall time: ~5 s (mock mode, CPU only).

---

### All Files Changed This Session

| File | Change type | Summary |
|---|---|---|
| `config.py` | Updated | Expanded `BCRConfig` + `MDConfig` docstrings |
| `main.py` | Updated | Import `MODELS_DIR` + structural pathways; OAS log; RCSB log; VAE checkpoint logic; `stage_structural_pathways()`; stage 2.5 wired into `run_pipeline()` + return dict |
| `models/vae.py` | Updated | `save_checkpoint()` + `load_checkpoint()` |
| `md_simulations/binding_md.py` | Updated | `MMPBSACalculator`; rewritten `_md_interface_energy()`; `pdb_source` kwarg on `BindingMDPredictor` |
| `md_simulations/structural_pathways.py` | **New** | `AgAbComplexBuilder`, `BindingPathwaySimulator` |

---

*End of session — 2026-02-24 (Gap Closure)*