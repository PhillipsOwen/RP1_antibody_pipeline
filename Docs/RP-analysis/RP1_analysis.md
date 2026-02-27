# RP1: Antibody Discovery Pipeline — Analysis

---

## Specific Aims Framework

The RP1 pipeline is organized around three Specific Aims that operate at distinct biological scales, together bridging atomic-resolution binding mechanisms to population-level immune repertoire coverage. The pipeline stages below implement these aims computationally.

### SA1 — Atomic Scale: Association Intermediate Structures for Affinity Prediction

**Hypothesis:** Key association intermediate structures play an essential role in antibody-antigen affinity, and AI models need these structural insights in their training data for high-accuracy affinity predictions.

**Approach:** Machine learning-accelerated molecular dynamics (ML-MD) is used to examine association intermediate structures for a large, diverse dataset of antibody-antigen complex structures. A lab-in-the-loop framework validates these simulations using binding affinity assays against panels of existing antibodies. The resulting dataset of association mechanisms identifies recurrent structural motifs and provides deeper structural context for affinity prediction models.

**Pipeline mapping:**
- **Stage 2b (MD Binding)** — Core ML-accelerated MD simulations of Ag-Ab complexes; MM/PBSA energy calculations capture the energetic landscape of association intermediates
- **Stage 2.5 (Structural Pathways)** — Explicit analysis of the binding pathway; MSM-based extraction of metastable intermediate states along the association trajectory
- **Stage 4 (MD + MSM)** — Markov State Models over MD trajectories quantify the kinetics and populations of association intermediates; implied timescales and stationary distributions characterize each state
- **Stage 2c (ALM Fine-tuning)** — MD-derived binding scores (capturing intermediate-state information) are used as the training signal to fine-tune ESM2; this is the direct computational implementation of the hypothesis that structural intermediate data improves affinity prediction accuracy
- **Stage 9 (Experimental Validation)** — Binding affinity assays against existing antibody panels close the lab-in-the-loop; Pearson/Spearman correlations and RMSE quantify whether intermediate-informed MD improves prediction
- **Stage 10 (Lab-in-the-Loop)** — Iterative refinement: experimental binding measurements update the model, driving the next round of MD and ALM fine-tuning

**Key scientific output:** A rich dataset of antibody-antigen association mechanisms annotated with recurrent structural motifs at intermediate states, used to train higher-accuracy affinity prediction models.

---

### SA2 — Repertoire Scale: Structure-Aware BCR Repertoire Analysis and Evolvability Modeling

**Hypothesis:** The massive sequence diversity of B-cell receptors (BCRs) can be distilled by homing in on structural convergence across sequences.

**Approach:** Protein language models (PLMs) construct and scan structure-aware representations of paratopes (antibody regions that contact antigen) across the BCR repertoire. Epitope-binning assays and computational prediction identify epitopes on the antigen and the corresponding paratopes from existing antibodies, enabling quantification of each epitope's coverage. Models of repertoire evolvability are built from known somatic hypermutation (SHM) hotspots; PLMs then predict pathogen evolution that evades the existing BCR repertoire.

**Pipeline mapping:**
- **Stage 1 (BCR Repertoire + Immune Atlas)** — Loads BCR sequence data from OAS; constructs the immune atlas (centroid + covariance matrix) as a compressed structure-aware representation of paratope diversity across the repertoire
- **Stage 2 (LM Scoring)** — ESM2 PLM scores each BCR for fitness; the PLM captures structural convergence implicitly through learned evolutionary constraints, implementing the SA2 hypothesis computationally
- **Stage 2a (Antigen-ALM Binding Profile)** — Binding site prediction across antibody × antigen pairs; computationally identifies which paratopes correspond to which epitopes, equivalent to computational epitope-binning
- **Stage 2d (Blind Spot Analysis)** — Quantifies per-epitope coverage in the existing BCR repertoire; identifies epitopes with insufficient paratope coverage (immune blind spots), directly implementing SA2's coverage quantification goal
- **Stage 5 (Synthetic Evolution)** — Genetic algorithm modeling of in-silico affinity maturation; SHM hotspot-aware mutation rates model repertoire evolvability; fitness landscape exploration captures potential antibody trajectories under selective pressure
- **Stage 6 (Repertoire Screening)** — Repertoire-scale evaluation of candidates; directly measures which fraction of the BCR repertoire provides functional coverage of each epitope bin
- **Stage 0 (Viral Escape Panel)** — Computational mutagenesis of the antigen implements the pathogen-evolution side of SA2: predicting mutations that evade existing BCR repertoire coverage

**Key scientific output:** Structure-aware paratope embeddings that quantify per-epitope BCR repertoire coverage; evolvability models predicting pathogen variants that outpace the existing immune response, with implications for medical countermeasures.

---

### SA3 — Pandemic Preparedness Case Studies: SARS-CoV-2 and HIV-1

**Approach:** Two parallel case studies apply the SA1 and SA2 methods to clinically relevant pathogens:

**SA3a — SARS-CoV-2 Variant-Response Prediction:**
- Models and experimentally validates binding sites of antibodies against spike protein variants
- Analyzes longitudinal BCR repertoire data to correlate the statistical distribution of variant-specific paratopes with observed disease progression
- Goal: A precision, individualized variant-susceptibility model

**SA3b — HIV-1 Immune Repertoire Characterization:**
- Antibody neutralization of HIV Env is categorized by binding mechanism (e.g., V1/V2 vs. V3 binders)
- Evaluates individual responses to HIV-1 acquisition by identifying BCRs structurally concordant with each broad neutralization category
- Creates a per-individual "fingerprint" of the B-cell immune response profile
- Relates the fingerprint to protection estimates against variants

**Pipeline mapping:**
- **Stage 0 (Viral Escape Panel)** — For SA3a: generates SARS-CoV-2 spike protein variants targeting ACE2-contact residues; for SA3b: generates Env protein escape variants across V1/V2 and V3 epitope regions
- **Stage 1 (BCR Repertoire)** — Longitudinal BCR data loading supports SA3a's correlation with disease progression; per-individual atlas construction supports SA3b's per-individual fingerprinting
- **Stage 7 (Cross-reactivity Analysis)** — Tests antibody candidates against the full escape panel for both SARS-CoV-2 variants and HIV Env variants; coverage matrix encodes which antibody categories (V1/V2, V3, etc.) neutralize which variants
- **Stage 8 (Vaccine Design)** — Selects broadly neutralizing candidate sets; for SA3b, the greedy set cover can be stratified by binding mechanism category to ensure representation across neutralization modalities
- **Stage 9 (Experimental Validation)** — Longitudinal binding assays validate the variant-susceptibility model (SA3a) and the per-individual fingerprint (SA3b); current metrics (Pearson r = 0.276, Spearman = 0.302) establish a baseline to improve

**Key scientific output (SA3a):** A precision, individualized SARS-CoV-2 variant-susceptibility model correlating paratope distribution with disease progression. **Key scientific output (SA3b):** A per-individual HIV-1 B-cell immune fingerprint relating antibody category composition to protection against variants.

---

## Aim-to-Stage Mapping — Consolidated Reference

| Specific Aim | Biological Scale | Primary Pipeline Stages | Secondary Stages |
|---|---|---|---|
| SA1: Association Intermediates | Atomic (residue/structure) | 2b, 2.5, 4, 2c | 9, 10 |
| SA2: Repertoire + Evolvability | Repertoire (sequence population) | 1, 2, 2a, 2d, 5, 6 | 0, 3 |
| SA3a: SARS-CoV-2 variants | Clinical case study | 0, 7, 8, 9 | 1, 2b, 10 |
| SA3b: HIV-1 fingerprint | Clinical case study | 0, 1, 7, 8, 9 | 6, 10 |

---

## Pipeline Overview

RP1 is a 16-stage end-to-end computational pipeline for discovering broadly neutralizing antibodies against viral escape mutants. It integrates molecular biology sequence data, molecular dynamics (MD) simulations, antibody language models (ALMs), generative machine learning (VAE/GAN), Markov state modeling (MSM), and a lab-in-the-loop experimental feedback cycle.

```
Viral Sequences (FASTA)
    |
    v
Stage 0: Viral Escape Panel Generation
    |
    v
Stage 1: BCR Repertoire + Immune Atlas Construction
    |
    v
Stage 2: Language Model (LM) Scoring
    |
    +-----> Stage 2a: Antigen-ALM Binding Profile
    |              |
    +-----> Stage 2b: MD Binding Prediction <-----+
    |              |
    +-----> Stage 2c: ALM Fine-tuning
    |
    +-----> Stage 2d: Blind Spot Analysis
    |
    v
Stage 2.5: Structural Pathway Analysis
    |
    v
Stage 3: Structural Modeling (VAE / GAN)
    |
    v
Stage 4: MD + Markov State Model (MSM)
    |
    v
Stage 5: Synthetic Evolution (genetic algorithms)
    |
    v
Stage 6: Repertoire-Scale Screening
    |
    v
Stage 7: Cross-reactivity Analysis
    |
    v
Stage 8: Vaccine Candidate Design
    |
    v
Stage 9: Experimental Validation
    |
    v
Stage 10: Lab-in-the-Loop Refinement
```

Checkpoints are saved automatically at each of the 16 stages under `experiments/checkpoints/<run_id>/`.

---

## Key Scientific Objectives

Organized by Specific Aim:

**SA1 (Atomic Scale)**
- **Characterize association intermediates** — use ML-accelerated MD to capture and annotate the ensemble of structures along the antibody-antigen association pathway, not just the final bound state.
- **Enrich AI training data** — incorporate association intermediate structural data into ALM fine-tuning (Stage 2c) to improve affinity prediction accuracy beyond what sequence-level data alone provides.
- **Identify structural motifs** — extract recurrent structural patterns from association intermediate ensembles that explain affinity differences across diverse Ag-Ab pairs.
- **Validate with binding assays** — close the lab-in-the-loop (Stages 9, 10) using experimental binding affinity measurements against panels of existing antibodies.

**SA2 (Repertoire Scale)**
- **Build structure-aware paratope representations** — use protein language models to embed BCR paratopes in a structure-informed space that captures convergence across diverse sequences.
- **Quantify epitope coverage** — computationally bin paratopes by epitope target and measure each epitope's fractional coverage in the BCR repertoire.
- **Model repertoire evolvability** — parameterize somatic hypermutation hotspot rates and use PLMs to predict future antibody trajectories under antigen pressure.
- **Predict immune-evading pathogen variants** — identify mutations that escape the existing BCR repertoire, flagging the need for medical countermeasures.

**SA3 (Pandemic Preparedness)**
- **SARS-CoV-2 individualized susceptibility** — correlate variant-specific paratope distributions from longitudinal BCR data with disease progression to build a precision variant-susceptibility model per individual.
- **HIV-1 per-individual B-cell fingerprint** — categorize individual BCR repertoires by structural concordance with known HIV Env neutralization mechanisms (V1/V2, V3) and relate each fingerprint to variant protection estimates.
- **Predict viral escape** — computationally generate mutant antigen variants (SARS-CoV-2 spike, HIV Env) that evade existing antibody coverage.
- **Design broadly neutralizing vaccine candidates** — select candidate sets that together cover the full escape panel, stratified by neutralization mechanism for HIV.

---

## Technical Concepts and Methods

| Concept | Role in Pipeline |
|---|---|
| ESM2 Antibody Language Model (ALM) | Pre-trained protein LM used to score sequence fitness and generate candidates (Stage 2) |
| BCR Repertoire / OAS Database | B-cell receptor sequence data source; basis for immune atlas construction (Stage 1) |
| Immune Atlas (centroid + covariance) | Statistical summary of the BCR embedding space representing observed immune coverage (Stage 1) |
| Computational Mutagenesis | Systematic point mutation of viral epitopes to generate escape panel (Stage 0) |
| MM/PBSA Energy Calculations | Physics-based binding free energy estimation from MD trajectories (Stage 2b) |
| OpenMM / GROMACS / CHARMM | MD simulation backends; configurable via `MDConfig.backend` (Stages 2b, 2.5, 4) |
| MDTraj | MD trajectory analysis library (Stages 2b, 2.5, 4) |
| PyEMMA / MSM | Markov State Model construction from MD trajectories to identify metastable binding states (Stage 4) |
| VAE (Variational Autoencoder) | Dimensionality reduction of antibody sequences into a continuous latent space (Stage 3) |
| GAN (Generative Adversarial Network) | Generative sampling of new antibody candidates from latent space (Stage 3) |
| Genetic Algorithm / Synthetic Evolution | In-silico affinity maturation — iterative mutation + selection of candidates (Stage 5) |
| Greedy Set Cover | Algorithm for selecting the minimal vaccine candidate set covering the escape panel (Stage 8) |
| Lab-in-the-Loop (Active Learning) | Experimental feedback cycle — new binding data updates model and prioritizes next experiments (Stage 10) |
| Checkpoint System | Incremental state persistence at each stage; enables resume from any point after failure |
| Association Intermediate Structures (SA1) | Metastable states along the antibody-antigen association pathway captured by ML-accelerated MD and MSM; these pre-bound structures are hypothesized to be essential for accurate affinity prediction |
| ML-Accelerated Molecular Dynamics (SA1) | Machine learning force fields or surrogate potentials that accelerate MD sampling beyond what classical MD can achieve in practical compute budgets; required to generate the large, diverse Ag-Ab trajectory dataset needed for SA1 |
| Structure-Aware Paratope Embedding (SA2) | PLM-derived representations of antibody paratopes that capture structural convergence across sequences with different primary sequences but similar 3D binding modes |
| Epitope Binning (SA2) | Experimental or computational classification of antibodies by which epitope region on the antigen they target; implemented computationally via the Stage 2a affinity matrix but requires discrete bin assignment logic |
| SHM Hotspot-Parameterized Evolvability (SA2) | Mutation rate models where substitution probabilities are position-dependent, reflecting known WRCY/RGYW somatic hypermutation hotspot motifs; enables realistic in-silico affinity maturation |
| Per-Individual BCR Fingerprint (SA3b) | A structured summary of an individual's B-cell immune response profile, categorizing BCR repertoire composition by neutralization mechanism category (e.g., V1/V2 vs. V3 binders); used to estimate protection against HIV-1 variants |
| Variant-Susceptibility Model (SA3a) | A precision, individualized model correlating the statistical distribution of variant-specific paratopes in longitudinal BCR data with observed SARS-CoV-2 disease progression |

---

## Stage 0: Viral Escape Panel Generation

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Wildtype antigen sequence | Full viral protein sequence (e.g., SARS-CoV-2 spike, ~1273 aa); loaded from FASTA at startup | Amino acid string | `.fasta` (`data/SARS-CoV-2_sequences.fasta`) |
| Epitope residue list | 0-indexed positions of ACE2-contact residues targeted for mutagenesis | List of integers | Hardcoded in `config.py` (`ViralEscapeConfig.epitope_residues`) |
| Panel config | `panel_size` (default 50), `max_mutations` (default 3), `binding_threshold` (default 0.5) | Numeric scalars | `config.py` (`ViralEscapeConfig`) |

**Data model:** Single wildtype sequence string; epitope residues as an integer index list. The escape generator produces combinatorial point mutations over those positions.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Escape mutant sequences | Panel of mutated antigen variants used as targets throughout the pipeline | List of amino acid strings | `.txt` (one sequence per line), `.npy` |
| Epitope residue positions | Confirmed contact residue indices passed downstream | List of integers | `.json` |
| Mutational landscape summary | Per-position mutation frequencies and scores | Tabular summary | `.json` (checkpoint `summary.json`) |

**Checkpoint:** `stage_0_escape_panel/`

---

## Stage 1: BCR Repertoire and Immune Atlas Construction

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| OAS bulk-download BCR data | B-cell receptor amino acid sequences from the Observed Antibody Space database; bulk CSV downloads by species/study/isotype | Tabular, column `sequence_alignment_aa` (VH AA sequence) + metadata | `.csv` (gzipped from OAS; decompressed into `data/oas/`) |
| Private BCR repertoire (optional) | Institution-specific BCR sequences | Tabular or sequence file | `.csv` or `.fasta` (path set in `BCRConfig.private_data_path`) |
| Disease label | Tag applied to the atlas for this repertoire (e.g., `"COVID-19"`) | String scalar | `config.py` (`BCRConfig.disease_label`) |

**Data model:** One row per BCR sequence; required column is `sequence_alignment_aa`; optional metadata columns (subject, isotype, disease, etc.). Multiple CSV files from OAS bulk downloads are loaded and merged.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| BCR sequences | Filtered set of amino acid sequences used as antibody candidate pool | List of strings | `.txt` (one per line) |
| Atlas centroid | Mean embedding vector representing the center of the immune response space | 1D numeric vector | `.npy` |
| Atlas covariance matrix | Covariance of embeddings capturing the spread of immune coverage | 2D numeric matrix | `.npy` |
| Disease labels | Per-sequence disease label assignments | List/array | `.json` |
| Serialized atlas | Full atlas object for downstream use | Python object | `.pkl` (path: `data/atlas.pkl`) |

**Checkpoint:** `stage_1_bcr_repertoire/`

---

## Stage 2: Language Model Scoring

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| BCR sequences from Stage 1 | Candidate antibody amino acid sequences to score | List of strings | Passed in-memory; also `.txt` from Stage 1 checkpoint |
| LM config | Model name (`facebook/esm2_t33_650M_UR50D`), `max_length`, `top_k`, `num_sequences` | Config scalars | `config.py` (`LMConfig`) |

**Data model:** Each sequence is tokenized and passed through ESM2 for per-residue log-likelihood scoring (perplexity-based).

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Scored sequences | Antibody candidates with associated LM fitness scores | List of strings + numeric array | `.txt` (sequences), `.npy` (scores) |
| LM scores | Log-likelihood scores per sequence | 1D numeric array | `.npy` |
| Generated candidates | New sequences sampled via top-k mutation from high-scoring positions | List of strings | `.txt` |

**Checkpoint:** `stage_2_lm_scoring/`

---

## Stage 2a: Antigen-ALM Binding Profile

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Scored antibody sequences | From Stage 2 | List of strings | In-memory / Stage 2 checkpoint |
| Antigen FASTA variants | Viral antigen sequence variants (wildtype + escape panel) | List of AA strings | `.fasta` |
| AntigenALM config | `n_antigen_sequences`, `similarity_metric` (cosine/dot/euclidean) | Config scalars | `config.py` (`AntigenALMConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Affinity matrix | Predicted binding scores for each antibody × antigen pair | 2D numeric matrix (Abs × Ags) | `.npy` |
| Binding site predictions | Per-residue binding site likelihood scores | 2D array | `.npy` |

**Checkpoint:** `stage_2a_antigen_profile/`

---

## Stage 2b: MD Binding Prediction

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Antibody sequences | Candidates from Stage 2 | List of strings | In-memory |
| Antigen escape panel | From Stage 0 | List of strings | In-memory |
| PDB structure file | 3D structure of the Ag-Ab complex for MD initialization | Atomic coordinate file | `.pdb` or `.cif` (from RCSB PDB, AlphaFold DB, or ESMFold) |
| Force field | AMBER14, CHARMM36, or GROMACS `.top`/`.itp` depending on backend | Field parameter files | `.xml` (OpenMM), `.top`/`.itp` (GROMACS) |
| MD config | Backend, simulation steps, step size, temperature, trajectory output path | Config scalars | `config.py` (`MDConfig`) |

**Data model:** Structure preparation required before MD — PDBFixer adds missing residues/hydrogens. Topology `.pdb` and trajectory `.xtc` paths set in `MDConfig`.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Binding matrix | MD-predicted binding affinity scores for each Ab × Ag pair (MM/PBSA) | 2D numeric matrix | `.npy` |
| MD trajectory | Atomic simulation trajectory for downstream MSM analysis | Binary trajectory | `.xtc` (GROMACS), `.dcd` (CHARMM), `.nc` (OpenMM); path: `md_simulations/output/trajectory.xtc` |
| Energy components | Per-pair electrostatic, van der Waals, solvation energy terms | Tabular | `.json`, `.npy` |
| Structural features | Per-frame RMSD, contact maps, buried surface area | Numeric arrays | `.npy` |

**Checkpoint:** `stage_2b_md_binding/`

---

## Stage 2c: ALM Fine-tuning

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| MD binding matrix | From Stage 2b — used as training signal | 2D numeric matrix | `.npy` |
| Antibody sequences | Candidates to fine-tune on | List of strings | In-memory |
| ALM finetune config | `learning_rate`, `n_epochs`, `margin`, `min_binding_score` | Config scalars | `config.py` (`ALMFinetuneConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Fine-tuned ALM weights | Updated ESM2 model weights after MD-guided pairwise margin ranking loss | Serialized model | `.pkl` or `.pt` (PyTorch state dict) |
| Training history | Loss per epoch | Tabular | `.json` |
| Updated scored sequences | Sequences re-scored with fine-tuned model | List of strings + array | `.txt`, `.npy` |

**Checkpoint:** `stage_2c_alm_finetune/`

---

## Stage 2d: Blind Spot Analysis

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Immune atlas (centroid + covariance) | From Stage 1 | Numeric arrays | `.npy` / `.pkl` |
| Escape panel sequences | From Stage 0 | List of strings | In-memory |
| Blind spot config | `blind_spot_threshold` (cosine similarity cutoff, default 0.5) | Scalar | `config.py` (`BlindSpotConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Blind spot report | JSON report identifying escape variants with insufficient immune atlas coverage | Structured document | `.json` (path: `experiments/output/blind_spot_report.json`) |
| Coverage statistics | Per-variant cosine similarity scores vs. atlas centroid | Tabular | `.json` (checkpoint `summary.json`) |

**Checkpoint:** `stage_2d_blind_spots/`

---

## Stage 2.5: Structural Pathway Analysis

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| MD trajectory | From Stage 2b | Binary trajectory | `.xtc` / `.dcd` / `.nc` |
| Topology file | Structural topology for MDTraj loading | Atomic coordinates | `.pdb` |
| Antibody + antigen sequences | For Ag-Ab complex construction | List of strings | In-memory |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Pathway MSM | Markov State Model of the binding pathway with transition probabilities | Serialized model | `.pkl` |
| Timescales | Implied timescales of metastable pathway states | 1D numeric array | `.npy` |
| Free energy landscape | Free energy surface over binding pathway coordinates | 2D numeric array | `.npy` |

**Checkpoint:** `stage_2_5_pathways/`

---

## Stage 3: Structural Modeling (VAE / GAN)

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Antibody sequences | From Stage 2 (scored candidates) | List of strings | In-memory |
| VAE config | `input_dim` (256), `hidden_dim` (512), `latent_dim` (64), training hyperparameters | Config scalars | `config.py` (`VAEConfig`) |
| GAN config | `noise_dim` (128), `hidden_dim` (256), `output_dim` (256), training hyperparameters | Config scalars | `config.py` (`GANConfig`) |

**Data model:** Sequences are featurized into fixed-length numeric vectors (dim=256 per `VAEConfig.input_dim`) before being passed to the VAE encoder.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Latent embeddings | VAE-encoded representations of antibody sequences in 64-dim latent space | 2D numeric matrix (sequences × latent_dim) | `.npy` |
| Reconstructed sequences | VAE decoder output sequences | List of strings | `.txt` |
| GAN-generated samples | New antibody candidate sequences sampled from latent space via GAN | List of strings | `.txt` |
| Model weights | Trained VAE and GAN weights | Serialized models | `.pkl` or `.pt` |

**Checkpoint:** `stage_3_structure/`

---

## Stage 4: MD + Markov State Model (MSM)

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| MD trajectory | From Stage 2b (and/or Stage 2.5) | Binary trajectory | `.xtc` / `.dcd` / `.nc` |
| Topology file | For MDTraj loading | `.pdb` | `.pdb` |
| MSM config | `lag_time` (default 10 frames), `n_states` (default 20), `n_jobs` (4) | Config scalars | `config.py` (`MSMConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| MSM model | PyEMMA Markov State Model object encoding metastable conformational states | Serialized model | `.pkl` |
| Implied timescales | Relaxation timescales of each metastable state | 1D numeric array | `.npy` |
| Stationary distribution | Equilibrium population of each MSM state | 1D numeric array | `.npy` |
| Transition matrix | Probability of transitioning between each pair of metastable states | 2D numeric matrix | `.npy` |

**Checkpoint:** `stage_4_msm/`

---

## Stage 5: Synthetic Evolution

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Candidate sequences | From Stage 2 or 3 | List of strings | In-memory |
| Scoring function | Combined LM score + MD binding score used as fitness | Callable (in-memory) | — |
| Evolution config | `n_generations` (10), `population_size` (500), `mutation_rate` (0.05), `top_fraction` (0.2) | Config scalars | `config.py` (`EvolutionConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Evolved sequences | Final population of optimized antibody candidates after N generations | List of strings | `.txt` |
| Fitness scores | Per-sequence composite fitness score at final generation | 1D numeric array | `.npy` |
| Generation history | Fitness statistics (mean, max, min) per generation | Tabular time series | `.json` |

**Checkpoint:** `stage_5_evolution/`

---

## Stage 6: Repertoire-Scale Screening

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Evolved candidates | From Stage 5 | List of strings | In-memory |
| Repertoire config | `batch_size` (256), `embedding_dim` (64), `top_candidates` (100), `n_workers` | Config scalars | `config.py` (`RepertoireConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Top sequences | Highest-scoring candidates after repertoire-scale evaluation | List of strings | `.txt` |
| Top scores | Composite scores for top candidates | 1D numeric array | `.npy` |
| Ranking metrics | Per-candidate ranking statistics | Tabular | `.json` |

**Checkpoint:** `stage_6_screening/`

---

## Stage 7: Cross-reactivity Analysis

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Top candidate sequences | From Stage 6 | List of strings | In-memory |
| Escape panel sequences | From Stage 0 | List of strings | In-memory |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Coverage matrix | Predicted binding scores for each top candidate × each escape mutant | 2D numeric matrix (candidates × mutants) | `.npy` |
| Adaptation summary | Per-candidate breadth metrics (fraction of panel covered above threshold) | Tabular | `.json` |
| Cross-reactivity heatmap | Visual representation of coverage matrix | Image | `.png` (`experiments/output/cross_reactivity_heatmap.png`) |

**Checkpoint:** `stage_7_cross_reactivity/`

---

## Stage 8: Vaccine Candidate Design

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Coverage matrix | From Stage 7 | 2D numeric matrix | `.npy` / in-memory |
| Vaccine design config | `min_coverage_fraction` (0.60), `top_candidates` (20) | Config scalars | `config.py` (`VaccineDesignConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Vaccine candidates | Selected antibody sequences providing broadest escape panel coverage | List of strings | `.txt` |
| Coverage statistics | Fraction of escape panel covered by selected set | Tabular | `.json` |
| Design rationale | Greedy set cover selection log | Structured document | `.json` |

**Checkpoint:** `stage_8_vaccine_design/`

---

## Stage 9: Experimental Validation

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Experimental binding measurements | Lab-measured binding affinities for a subset of candidates | Tabular: columns `sequence`, `measured_binding` | `.csv` (path: `experiments/binding_data.csv`) |
| Predicted scores | Pipeline-computed scores for the same sequences | 1D numeric array | In-memory |
| Experiment config | `correlation_method` (pearson/spearman), output plot path | Config scalars | `config.py` (`ExperimentConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Predicted vs. experimental scores | Side-by-side comparison table | Tabular | `.json` (`experiments/output/validation_report.json`) |
| Correlation statistics | Pearson/Spearman r, p-value, RMSE | Tabular | `.json` |
| Validation report | Full structured validation summary | Document | `.json` |
| Correlation plot | Scatter plot of predicted vs. measured binding | Image | `.png` (`experiments/output/correlation_plot.png`) |
| Score distribution plot | Distribution of predicted scores | Image | `.png` (`experiments/output/score_distribution.png`) |

**Checkpoint:** `stage_9_validation/`

---

## Stage 10: Lab-in-the-Loop Refinement

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Experimental binding CSV | New lab measurements from the latest wet-lab round | Tabular: columns `sequence`, `measured_binding` | `.csv` (path: `LabLoopConfig.experimental_csv`) |
| Current model state | Fine-tuned ALM and scoring functions from earlier stages | In-memory model objects | `.pkl` / `.pt` |
| Lab loop config | `n_suggestions` (20), `escape_threshold` (0.3), `output_dir` | Config scalars | `config.py` (`LabLoopConfig`) |

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Lab loop results | Updated model performance metrics after incorporating new data | Structured document | `.json` (per-iteration, in `experiments/output/lab_loop/`) |
| Suggested next experiments | Ranked list of sequences recommended for the next wet-lab round | List of strings | `.json` |
| Updated escape panel | Confirmed escape variants (binding < `escape_threshold`) added to escape panel | List of strings | `.json` |
| Updated model performance | Metrics showing improvement after incorporating feedback | Tabular | `.json` |

**Checkpoint:** `stage_10_lab_loop/`

---

## Data Models and File Types — Consolidated Reference

### Input Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 0 | Wildtype antigen sequence | Single AA string | `.fasta` |
| Stage 0 | Epitope residue list | List of integers | `config.py` |
| Stage 1 | OAS BCR bulk data | Tabular: rows = sequences, col `sequence_alignment_aa` + metadata | `.csv` (from OAS bulk download) |
| Stage 1 | Private BCR repertoire (optional) | Tabular or sequence file | `.csv`, `.fasta` |
| Stage 2 | BCR sequences | List of AA strings | `.txt` / in-memory |
| Stage 2a | Antigen FASTA variants | List of AA strings | `.fasta` |
| Stage 2b | PDB structure file | 3D atomic coordinates | `.pdb`, `.cif` |
| Stage 2b | Force field parameters | MD engine parameter files | `.xml` (OpenMM), `.top`/`.itp` (GROMACS) |
| Stage 2b | MD trajectory (input) | Binary trajectory frames | `.xtc`, `.dcd`, `.nc` |
| Stage 2c | MD binding matrix | 2D float matrix (Abs × Ags) | `.npy` |
| Stage 2d | Immune atlas | Centroid vector + covariance matrix | `.npy`, `.pkl` |
| Stage 3 | Candidate sequences | List of AA strings (featurized to dim-256 vectors) | `.txt` / in-memory |
| Stage 4 | MD trajectory | Binary trajectory frames | `.xtc`, `.dcd`, `.nc` |
| Stage 4 | Topology file | Atomic coordinates for MDTraj | `.pdb` |
| Stage 5 | Candidate sequences | List of AA strings | In-memory |
| Stage 9 | Experimental binding data | Tabular: `sequence`, `measured_binding` | `.csv` |
| Stage 10 | New lab binding measurements | Tabular: `sequence`, `measured_binding` | `.csv` |

### Output Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 0 | Escape mutant sequences | List of AA strings | `.txt`, `.npy` |
| Stage 0 | Epitope residue positions | List of integers | `.json` |
| Stage 1 | BCR sequences | List of AA strings | `.txt` |
| Stage 1 | Atlas centroid | 1D float vector | `.npy` |
| Stage 1 | Atlas covariance | 2D float matrix | `.npy` |
| Stage 1 | Serialized atlas | Python atlas object | `.pkl` |
| Stage 2 | LM scores | 1D float array | `.npy` |
| Stage 2 | Scored + generated sequences | List of AA strings | `.txt` |
| Stage 2a | Affinity matrix | 2D float matrix (Abs × Ags) | `.npy` |
| Stage 2b | Binding matrix (MD) | 2D float matrix (Abs × Ags) | `.npy` |
| Stage 2b | MD trajectory (output) | Binary trajectory frames | `.xtc` / `.dcd` / `.nc` |
| Stage 2c | Fine-tuned ALM weights | Serialized model | `.pkl`, `.pt` |
| Stage 2d | Blind spot report | Structured coverage document | `.json` |
| Stage 2.5 | Pathway MSM | Serialized MSM object | `.pkl` |
| Stage 2.5 | Free energy landscape | 2D float array | `.npy` |
| Stage 3 | Latent embeddings | 2D float matrix (sequences × 64) | `.npy` |
| Stage 3 | Generated sequences (GAN) | List of AA strings | `.txt` |
| Stage 4 | MSM model | Serialized PyEMMA MSM | `.pkl` |
| Stage 4 | Transition matrix | 2D float matrix (states × states) | `.npy` |
| Stage 5 | Evolved sequences | List of AA strings | `.txt` |
| Stage 5 | Generation history | Per-generation fitness stats | `.json` |
| Stage 6 | Top candidate sequences | List of AA strings | `.txt` |
| Stage 7 | Coverage matrix | 2D float matrix (candidates × mutants) | `.npy` |
| Stage 7 | Cross-reactivity heatmap | Raster image | `.png` |
| Stage 8 | Vaccine candidates | List of AA strings | `.txt` |
| Stage 9 | Validation report | Predicted vs. experimental comparison | `.json` |
| Stage 9 | Correlation plot | Scatter image | `.png` |
| Stage 10 | Suggested next experiments | Ranked sequence list | `.json` |
| Stage 10 | Updated escape panel | Confirmed escape variant sequences | `.json` |

### Checkpoint File Formats (All Stages)

| File | Contents | Format |
|---|---|---|
| `metadata.json` | Stage name, timestamp, pipeline config snapshot | `.json` |
| `summary.json` | Quick statistics (array shapes, sequence counts, score ranges) | `.json` |
| `*.npy` | NumPy arrays — matrices, vectors, embeddings | Binary NumPy |
| `*.txt` | Sequence files — one amino acid sequence per line | Plain text |
| `*.json` | Dictionaries — reports, histories, structured outputs | JSON |
| `*.pkl` | Python pickled objects — models, atlases, MSMs | Python pickle |
| `*.pt` | PyTorch model state dicts | PyTorch binary |
| `*.xtc` / `*.dcd` / `*.nc` | MD simulation trajectories | Binary trajectory |
| `*.png` | Visualization outputs | Raster image |

### Cross-Stage Interface Formats

| Interface | Data Passed | Format | Gap / Risk |
|---|---|---|---|
| Stage 0 → Stage 7 | Escape panel sequences (used throughout) | List of strings (in-memory); `.txt` at checkpoint | Escape panel is reused at Stages 2a, 2d, 7, 8, 10 — schema must remain stable across runs |
| Stage 1 → Stage 2 | BCR sequences for LM scoring | List of strings / `.txt` | OAS CSV column naming may vary across bulk downloads; `bcr_loader.py` must normalize |
| Stage 2b → Stage 2c | MD binding matrix as ALM training signal | `.npy` 2D float matrix | MD proxy scores are approximations — fine-tuning quality depends on MD accuracy |
| Stage 2b → Stage 4 | MD trajectory for MSM construction | `.xtc` / `.dcd` / `.nc` | Trajectory file format depends on MD backend; MDTraj handles all three but paths must match `MDConfig` |
| Stage 3 → Stage 5 | Latent embeddings used as genetic algorithm seed population | `.npy` / in-memory | Latent space dimensionality (64) must match `RepertoireConfig.embedding_dim` downstream |
| Stage 9 → Stage 10 | Experimental binding CSV | `.csv` columns: `sequence`, `measured_binding` | CSV schema must match exactly — no validation guard present in `lab_loop.py` |

---

## Data Sharing and Constraints

| Data | Shareable? | Constraint |
|---|---|---|
| SARS-CoV-2 FASTA sequences | Yes | Publicly available (NCBI GenBank) |
| OAS BCR bulk data | Yes | Publicly available via OAS database |
| Private BCR repertoire | Institution-dependent | Optional input; may be restricted |
| PDB structure files | Yes (public structures) | RCSB PDB and AlphaFold DB are public; proprietary structures are not |
| MD trajectory files | Large | 1–10 GB per run; not typically shared in full |
| Checkpoint data | Yes | 100–500 MB per run; sharable for reproducibility |
| Experimental binding CSV | Institution-dependent | Lab measurement data; sharing subject to institutional policy |
| Trained model weights | Shareable | `.pkl`/`.pt` files; ~500 MB–2 GB for full ALM |

---

## Open Questions and Considerations

| Area | Issue |
|---|---|
| OAS schema normalization | BCR bulk CSVs from different OAS studies may use inconsistent column names; `bcr_loader.py` normalization logic must be validated against new downloads |
| MD proxy accuracy | Stage 2b uses MM/PBSA proxy models for speed; proxy binding scores may diverge from experimental values, propagating error into ALM fine-tuning (Stage 2c) |
| Escape panel stability | The escape panel is generated once (Stage 0) and reused across stages 2a, 2d, 7, 8, 10 — any mutation to panel composition mid-run invalidates downstream results |
| Lab loop CSV contract | `experiments/binding_data.csv` must have columns `sequence` and `measured_binding`; no runtime schema validation is present — a malformed CSV will cause a silent failure |
| Trajectory format coupling | MD trajectory format (`.xtc`/`.dcd`/`.nc`) is determined by `MDConfig.backend`; if backend is changed between runs, Stage 4 MSM loading will fail unless trajectory path is updated |
| Checkpoint versioning | No schema versioning on checkpoint directories — if pipeline logic changes between runs, older checkpoints may be incompatible with resume |
| GPU dependency | Full ALM (ESM2 650M) requires GPU for practical runtimes; mock mode substitutes random scores, which are not scientifically valid for production use |
| Wet lab integration latency | Lab-in-the-loop (Stage 10) requires physical experimental turnaround time; the pipeline has no built-in mechanism to pause and wait for new `binding_data.csv` — this handoff is manual |

### SA1-Specific Open Questions

| Area | Issue |
|---|---|
| Association intermediate definition | The pipeline currently analyzes MD trajectories post-hoc via MSM (Stages 2.5, 4); there is no explicit intermediate-state annotation step that labels frames as "association intermediates" vs. bound/unbound states — this labeling logic must be defined before motif identification can occur |
| ML-accelerated MD coupling | Stage 2b uses standard MD backends (OpenMM/GROMACS); ML-accelerated MD (e.g., TorchMD, OpenMM-ML, MLFF) is not yet integrated — the SA1 requirement for "ML-accelerated MD" implies a backend upgrade or a separate simulation layer |
| Dataset diversity requirement | SA1 specifies a "large dataset of diverse antibody-antigen complex structures"; current Stage 2b is scoped to a single complex (one PDB structure); a multi-complex, multi-target dataset pipeline requires structural data orchestration not currently present |
| Recurrent motif identification | No downstream analysis stage currently identifies or annotates recurrent structural motifs across the association intermediate ensemble — this is a gap between the MSM output (States 2.5/4) and the SA1 scientific goal |

### SA2-Specific Open Questions

| Area | Issue |
|---|---|
| Structure-aware paratope representation | Current Stage 2 (LM Scoring) uses ESM2 sequence-level embeddings; SA2 requires explicitly structure-aware paratope representations — this may require integrating structure prediction (ESMFold, AlphaFold) before embedding, not yet implemented |
| Epitope-binning integration | SA2 specifies epitope-binning assays to identify epitopes and corresponding paratopes; Stage 2a computes an affinity matrix but does not perform discrete epitope-bin assignment — a binning classification step is needed |
| SHM hotspot modeling | Stage 5 (Synthetic Evolution) uses a uniform mutation rate (`mutation_rate` scalar in `EvolutionConfig`); SA2 requires mutation rates parameterized by known SHM hotspot positions (WRCY/RGYW motifs), which are sequence-position-dependent and not currently modeled |
| Pathogen evolution prediction | SA2 explicitly aims to predict pathogen variants that evade the existing BCR repertoire — this is partially covered by Stage 0 (escape panel generation via computational mutagenesis), but SA2 implies a PLM-driven forward prediction of likely future variants, not exhaustive combinatorial mutagenesis |

### SA3-Specific Open Questions

| Area | Issue |
|---|---|
| Longitudinal BCR data handling | SA3a requires correlating the statistical distribution of variant-specific paratopes with disease progression over time — current Stage 1 loads BCR data as a static snapshot; longitudinal time-indexing of BCR samples per individual is not yet implemented |
| Per-individual atlas | SA3b requires a per-individual B-cell immune fingerprint; current Stage 1 builds a single population-level atlas (centroid + covariance) — per-individual atlases require subject-stratified loading and separate atlas objects per donor |
| HIV Env epitope categorization | SA3b categorizes antibodies by binding mechanism (V1/V2 vs. V3); Stage 7 cross-reactivity analysis produces an undifferentiated coverage matrix — a mechanism-category label layer must be added to stratify coverage by neutralization modality |
| Variant-susceptibility model output | SA3a's precision susceptibility model is not a current pipeline output; Stage 9 produces Pearson/Spearman correlations between predicted and measured binding, but a per-individual susceptibility score (integrating paratope distribution, variant profile, and disease outcome) is not yet formalized |
| Baseline performance gap | Current experimental validation metrics (Pearson r = 0.276, Spearman = 0.302, ROC AUC = 0.648) are modest; SA1's hypothesis that adding association intermediate structural data to training will improve these is testable but unverified — these metrics constitute the pre-SA1-enrichment baseline |
