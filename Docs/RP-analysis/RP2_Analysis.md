# RP2: Spatiotemporal Dynamics of Cell-Virus Interactions — Analysis

---

## Pipeline Overview

The pipeline is a four-stage abstraction ladder moving from empirical biology to mathematical prediction:

```
Experimental Data (cell/molecular scale)
    |
    v
ABM — Agent-Based Model (cell scale, synthetic tissue)
    |
    v
PINN — Physics-Informed Neural Network (SI model integration)
    |
    v
ODE/PDE/SDE — Differential Equations (organ scale, forward simulation)
```

Each stage compresses and abstracts the previous, ending at interpretable, computationally cheap equations that encode spatial and temporal dynamics lost in traditional compartmental models.

---

## Stage 1: Experimental

**Input:** Tissue explants + single-cell RNA sequencing (scRNA-seq)
**Output:** TB-scale raw data → standardized format for ABM

The ETL conversion layer ("code to convert for the simulation") is the highest-risk integration point in the entire pipeline. It must define a stable schema contract between wet lab and computational stages. The "standardized output for Kevin (ABMs)" implies this is a named interface — likely a specific file format (HDF5/Parquet/AnnData `.h5ad`) with versioned cell type ontology, spatial coordinates, and temporal indexing.

**At terabyte scale**, full in-memory processing is not feasible. This implies chunked/streaming ETL before any modeling begins.

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| scRNA-seq reads | Raw sequencing reads per cell, capturing gene expression profiles across thousands of genes | High-dimensional sparse matrix (cells × genes) | `.fastq` (raw reads), `.bam` (aligned), `.h5`, `.h5ad` (AnnData), `.mtx` + barcodes/features (10x Genomics format) |
| Tissue explant metadata | Sample provenance, tissue type, donor info, experimental conditions | Tabular key-value | `.csv`, `.json` |
| Spatial coordinates | Physical location of each cell within the tissue section (if spatially resolved) | Numeric 2D/3D coordinate array | `.csv`, `.h5ad` (spatial slot), `.zarr` |
| Cell type annotations | Assigned cell type labels per cell (e.g., B cell, T cell, dendritic cell) | Categorical vector, one label per cell | `.csv`, `.h5ad` (obs slot) |

**Data model:** Cells × genes sparse count matrix, supplemented by per-cell metadata (coordinates, annotations, sample labels). Standard container is AnnData (`.h5ad`), which stores the matrix, obs (cell metadata), var (gene metadata), and spatial embeddings in a single file.

### Output Data (ETL Layer → ABM)

| Field | Description | Type | File Format |
|---|---|---|---|
| Standardized cell state vectors | Per-cell gene expression profiles reduced to biologically meaningful features (e.g., PCA/UMAP embeddings or marker gene signatures) | Numeric matrix, one row per cell | `.h5ad`, `.parquet`, `.csv` |
| Cell type and state labels | Ontology-controlled cell type assignments used to initialize agent types in the ABM | Categorical, one label per cell | `.csv`, `.h5ad` obs slot |
| Spatial layout | Cell positions used to initialize agent locations in the synthetic tissue geometry | 2D/3D coordinate array | `.csv`, `.h5ad` spatial slot |
| Versioned ETL schema | Documented field definitions, ontology versions, and transformation logic for reproducibility | Schema definition | `.json` schema, `.yaml`, or README |

**Data model:** The "standardized output for Kevin (ABMs)" is a versioned, schema-controlled flat or hierarchical table mapping each cell to its type, state, and spatial position — the minimum information needed to initialize an ABM agent.

---

## Stage 2: ABM (Agent-Based Model)

**Scale:** Individual cells as agents in a synthetic lymph node geometry

**Core differentiator:** Spatial neighborhood effects. Agents interact by proximity, not global averages. This captures:
- Localized viral spread patterns
- Germinal center follicular architecture
- T follicular helper / B cell proximity-dependent selection

**Cell fates as state machines:** Each agent transitions through biologically defined states (naive B cell → GC B cell → memory/plasma cell, or susceptible → infected → latent/lytic). The ABM generates **synthetic cell fate trajectories** used to train the PINN.

**Scaling risk:** Naive ABMs are O(N²). At tissue-realistic cell counts, spatial partitioning (octrees, grid hashing) is a hard engineering requirement, not mentioned in the description.

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Agent initialization table | Per-cell type, state, and spatial position from ETL output | Tabular, one row per agent | `.csv`, `.h5ad`, `.parquet` |
| Tissue geometry | Synthetic lymph node geometry defining spatial zones (cortex, paracortex, germinal center follicles) | Geometric mesh or grid | `.csv` (grid), `.vtk`, `.json` (zone definitions) |
| Interaction rules / parameters | Biologically defined rules governing agent behavior (infection probability, cell migration rate, contact range, division/death rates) | Key-value config | `.json`, `.yaml`, `.csv` |
| Viral dynamics parameters | Pathogen-specific parameters (burst size, latency period, lytic/lysogenic probabilities for EBV/HIV) | Key-value scalars | `.json`, `.csv` |

**Data model (agents):** One row per agent; columns include agent ID, cell type, current state, x/y/z coordinates, and any state-specific attributes (e.g., viral load if infected). Initialized from Stage 1 ETL output.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Synthetic cell fate trajectories | Time-series record of each agent's state transitions throughout the simulation — the primary input for PINN training | Event log: agent ID, state, timestep | `.csv`, `.parquet`, `.h5` |
| Spatial snapshots | Periodic captures of agent positions and states across the simulated tissue at each timestep | Array of agent tables keyed by timestep | `.csv`, `.h5`, `.zarr` |
| Aggregate infection dynamics | Summary counts per timestep (susceptible, infected, latent, dead) across the simulated tissue | Tabular time series | `.csv` |
| Simulation config / RNG seed | Full parameter set and random seed for reproducible replay | Config + scalar | `.json`, `.yaml` |

**Data model (trajectories):** One row per agent per timestep (or one row per state-change event in a sparse event-log format); columns include agent ID, cell type, state, timestep, and spatial coordinates. This is the training dataset for the PINN in Stage 3.

---

## Stage 3: PINN (Physics-Informed Neural Network)

**Role:** Extract physically meaningful parameters from noisy, high-dimensional ABM output

The PINN loss function = **data loss** (fit ABM trajectories) + **physics loss** (comply with SI-model equations):

```
dS/dt = -β·S·I
dI/dt =  β·S·I
```

Rather than a scalar β, the PINN learns **spatiotemporally varying parameter fields** — a β that changes across lymph node zones (cortex, paracortex, germinal centers) and over infection time. This is the bridge from stochastic, spatial simulation to smooth, structured mathematics.

**Risk:** PINNs are sensitive to data/physics loss weighting. The SI prior may be too rigid for EBV/HIV, which have latency phases. Extension to SEIR or latency-inclusive compartments may be necessary.

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Synthetic cell fate trajectories | ABM output — per-agent state and spatial time series used as the data component of the PINN loss | Tabular time series or event log | `.csv`, `.parquet`, `.h5` |
| Physics constraints | SI (or extended) ODE equations defining the physics loss term | Symbolic or numeric equation specification | Hardcoded in model definition (`.py`, `.jl`), optionally `.json` / `.yaml` config |
| Spatial zone map | Zone boundaries within the lymph node geometry (used to tile spatially varying β fields) | Geometric definition | `.json`, `.csv`, `.vtk` |
| Training hyperparameters | Learning rate, loss weighting (data vs. physics), network architecture, batch size | Key-value config | `.json`, `.yaml` |

**Data model:** Training samples are (time, x, y, z, S, I) tuples drawn from ABM trajectory snapshots, augmented with physics residual evaluations at collocation points. High volume at tissue-realistic agent counts.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Spatiotemporally varying β field | Learned transmission rate parameter as a function of lymph node zone and time — the core scientific output | Numeric tensor (spatial grid × time) | `.npy`, `.h5`, `.csv` (flattened) |
| Trained PINN weights | Neural network weights encoding the learned parameter fields | Serialized model weights | `.pt` (PyTorch), `.h5` (Keras/TF), framework-specific |
| Fitted S/I trajectories | PINN-smoothed susceptible/infected population curves, free of ABM stochastic noise | Tabular time series | `.csv` |
| Loss curves | Training and validation loss over epochs (data loss + physics loss components) | Tabular time series | `.csv` |

**Data model:** The primary scientific output is the β field tensor indexed by spatial zone and timestep. Downstream ODE/PDE solvers consume this as a spatially varying coefficient.

---

## Stage 4: Differential Equations

**Scale:** Organ-level forward simulation
**Model types:**
- **ODEs** — well-mixed population compartments, temporal dynamics
- **PDEs** — spatially resolved (cytokine gradients, cell migration, reaction-diffusion across lymph node zones)
- **Stochastic processes** — SDEs or Gillespie algorithms for biological randomness

**Fault tolerance via MongoDB:**
- State checkpointed to local MongoDB incrementally
- Each document = full simulation state at time T (population vectors, parameters, RNG seed, metadata)
- "Pick up where you left off" = resumable from last checkpoint
- "Semi-fault tolerant" is an honest qualifier — mid-step failures and database write corruption are not handled

MongoDB's schemaless documents suit variable-structure simulation states as model architecture evolves during research.

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| PINN-derived parameter fields | Spatiotemporally varying β (and other rate parameters) learned by the PINN in Stage 3 | Numeric tensor (spatial grid × time) | `.npy`, `.h5`, `.csv` |
| Initial conditions | Starting population compartment sizes (S₀, I₀, etc.) and spatial concentration fields | Numeric vector or spatial grid | `.csv`, `.json`, `.h5` |
| ODE/PDE model definition | Equation structure, compartments, and boundary conditions for the forward solver | Symbolic or numeric specification | Code (`.py`, `.jl`, `.m`), optionally `.xml` (SBML) |
| Checkpoint state (resume) | Previously saved simulation state from MongoDB for resumable execution | BSON document (MongoDB native) | MongoDB collection (`.bson` on disk) |

**Data model:** Initial conditions are vectors (ODE) or spatial grids (PDE). PINN-derived parameter fields are tensors indexed by zone and time, interpolated by the solver at each integration step.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Population trajectory | Time-series of compartment sizes (S, I, and additional compartments) at organ scale | Tabular time series | `.csv`, MongoDB document |
| Spatial concentration fields | PDE solution fields (cytokine gradients, cell density maps) across lymph node zones over time | Numeric tensor (space × time) | `.h5`, `.vtk`, `.zarr` |
| Simulation checkpoints | Full simulation state at each checkpoint interval for fault tolerance and resumability | Schemaless document | MongoDB collection (BSON); exportable to `.json` |
| Stochastic ensemble output | Multiple SDE/Gillespie realizations capturing biological randomness | Collection of trajectory tables | `.csv`, `.h5` |
| Synthetic cell fate + population predictions | Final scientific outputs combining cell-level fates and organ-level population dynamics | Tabular summaries + trajectory tables | `.csv`, `.h5` |

**Data model (checkpoints):** Each MongoDB document contains the full simulation state at time T — population vectors, current parameter values, RNG seed, and run metadata. Schema is intentionally flexible to accommodate model architecture changes during research.

---

## Scientific Applications

**EBV → Abnormal B Cell Development → Autoimmunity**
EBV hijacks the germinal center reaction via latency genes (LMP1, LMP2A, EBNA2) that mimic activation signals, generating autoreactive B cell clones that escape tolerance. These clones are implicated in SLE, MS, and rheumatoid arthritis. The ABM's spatial neighborhood modeling directly captures whether an infected B cell is selected vs. eliminated based on follicular dendritic cell and T_FH proximity.

**HIV + Lymph Node on a Chip**
Microfluidic organ-on-a-chip devices recreate 3D lymph node architecture ex vivo, enabling controlled drug perturbation. The ABM parameterized from this captures collateral damage to healthy bystander cells — something population-level studies miss.

**SARS-CoV-2 Generalizability**
Any pathogen with ex vivo-compatible infection dynamics can parameterize a new ABM instance without changing PINN or ODE/PDE architecture. The pipeline is designed as a platform.

---

## Data Models and File Types — Consolidated Reference

### Input Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 1 — Experimental | scRNA-seq reads | Sparse matrix: cells × genes | `.fastq`, `.bam`, `.h5`, `.h5ad`, `.mtx` |
| Stage 1 — Experimental | Tissue explant metadata | Key-value table | `.csv`, `.json` |
| Stage 1 — Experimental | Spatial coordinates | 2D/3D coordinate array, one row per cell | `.csv`, `.h5ad` (spatial slot) |
| Stage 1 — Experimental | Cell type annotations | Categorical vector, one label per cell | `.csv`, `.h5ad` (obs slot) |
| Stage 2 — ABM | Agent initialization (from ETL) | Flat table: one row per agent, type/state/position | `.csv`, `.h5ad`, `.parquet` |
| Stage 2 — ABM | Tissue geometry | Spatial zone definitions / mesh | `.csv`, `.vtk`, `.json` |
| Stage 2 — ABM | Interaction rules / parameters | Key-value config | `.json`, `.yaml`, `.csv` |
| Stage 2 — ABM | Viral dynamics parameters | Key-value scalars | `.json`, `.csv` |
| Stage 3 — PINN | Synthetic cell fate trajectories (from ABM) | Tabular time series or event log: agent ID, state, timestep | `.csv`, `.parquet`, `.h5` |
| Stage 3 — PINN | Physics constraints (SI equations) | Symbolic equation spec embedded in model code | `.py`, `.jl` (hardcoded); `.json`/`.yaml` (config) |
| Stage 3 — PINN | Spatial zone map | Zone boundary geometry | `.json`, `.csv`, `.vtk` |
| Stage 3 — PINN | Training hyperparameters | Key-value config | `.json`, `.yaml` |
| Stage 4 — ODE/PDE/SDE | PINN-derived parameter fields | Tensor: spatial grid × time | `.npy`, `.h5`, `.csv` |
| Stage 4 — ODE/PDE/SDE | Initial conditions | Numeric vector (ODE) or spatial grid (PDE) | `.csv`, `.json`, `.h5` |
| Stage 4 — ODE/PDE/SDE | Model definition | Equation structure and boundary conditions | `.py`, `.jl`, `.m`, `.xml` (SBML) |
| Stage 4 — ODE/PDE/SDE | Checkpoint state (resume) | Full simulation state document | MongoDB BSON collection |

### Output Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 1 — ETL | Standardized cell state vectors | Wide table: one row per cell, columns = reduced features | `.h5ad`, `.parquet`, `.csv` |
| Stage 1 — ETL | Cell type / state labels | Categorical vector | `.csv`, `.h5ad` obs slot |
| Stage 1 — ETL | Spatial layout | 2D/3D coordinate array | `.csv`, `.h5ad` spatial slot |
| Stage 1 — ETL | Versioned ETL schema | Schema definition document | `.json`, `.yaml` |
| Stage 2 — ABM | Synthetic cell fate trajectories | Event log: agent ID, state, timestep, coordinates | `.csv`, `.parquet`, `.h5` |
| Stage 2 — ABM | Spatial snapshots | Array of agent tables keyed by timestep | `.csv`, `.h5`, `.zarr` |
| Stage 2 — ABM | Aggregate infection dynamics | Time series: timestep, S count, I count | `.csv` |
| Stage 2 — ABM | Simulation config / RNG seed | Key-value config + scalar | `.json`, `.yaml` |
| Stage 3 — PINN | Spatiotemporally varying β field | Tensor: spatial zone × time | `.npy`, `.h5`, `.csv` |
| Stage 3 — PINN | Trained PINN weights | Serialized neural network | `.pt` (PyTorch), `.h5` (Keras/TF) |
| Stage 3 — PINN | Fitted S/I trajectories | Time series: timestep, S, I | `.csv` |
| Stage 3 — PINN | Loss curves | Time series: epoch, data loss, physics loss | `.csv` |
| Stage 4 — ODE/PDE/SDE | Population trajectory | Time series: timestep, compartment sizes | `.csv`, MongoDB document |
| Stage 4 — ODE/PDE/SDE | Spatial concentration fields | Tensor: space × time | `.h5`, `.vtk`, `.zarr` |
| Stage 4 — ODE/PDE/SDE | Simulation checkpoints | Schemaless state documents | MongoDB collection (BSON) |
| Stage 4 — ODE/PDE/SDE | Stochastic ensemble output | Collection of trajectory tables | `.csv`, `.h5` |

### Cross-Stage Interface Formats

| Interface | Data Passed | Current Format | Gap / Risk |
|---|---|---|---|
| Stage 1 → Stage 2 (ETL) | Standardized cell type, state, spatial layout | `.h5ad` or `.csv` (schema not versioned) | Highest-risk interface — format drift breaks all downstream stages |
| Stage 2 → Stage 3 (PINN) | Synthetic cell fate trajectories | `.csv` / `.parquet` / `.h5` event log | Volume at tissue-realistic cell counts may require chunked loading |
| Stage 3 → Stage 4 (ODE/PDE) | Spatiotemporally varying β field | `.npy` / `.h5` tensor | Interpolation strategy between PINN grid resolution and ODE/PDE solver grid must be defined |
| Stage 4 → MongoDB | Full simulation state checkpoints | BSON documents | Mid-step write failures not handled; checkpoint granularity not specified |

---

## Key Gaps and Risks

| Area | Issue |
|------|-------|
| ETL contract | No versioned schema defined — format drift breaks all downstream stages |
| ABM validation | No stated criteria for validating synthetic tissue against real tissue |
| PINN physics prior | SI model may be too simple for latent-infection pathogens (EBV/HIV) |
| Parameter identifiability | Multiple parameter sets can produce identical ODE outputs — needs sensitivity analysis |
| ABM scaling | O(N²) neighbor checking at tissue-realistic cell counts requires spatial data structures |
| Checkpoint granularity | Trade-off between checkpoint frequency and storage/compute overhead not defined |
| Stochasticity | Random seed management and ensemble size for reproducible stochastic simulations not specified |

---

## Integration Points Summary

```
[scRNA-seq / Tissue Explants]
        |
   [ETL Module] ← versioned schema critical
        |
   [ABM Engine] ← spatial partitioning needed at scale
        |
   "standardized output for Kevin"
        |
   [PINN Training] ← SI physics loss, spatially varying β
        |
   [ODE/PDE/SDE Solver]
        |
   MongoDB checkpoints (semi-fault tolerant)
        |
   [Synthetic Cell Fates + Population Predictions]
```

The pipeline is scientifically ambitious and architecturally sound in concept. The two most critical implementation priorities are: (1) defining and stabilizing the ETL output schema, and (2) validating the ABM against ground-truth tissue data before the PINN and ODE stages can produce trustworthy results.
