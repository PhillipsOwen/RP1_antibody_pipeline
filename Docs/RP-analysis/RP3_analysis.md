# RP3: Link Host Immune Response to Population Disease Transmission — Analysis

---

## Pipeline Overview

The pipeline is a three-stage abstraction ladder linking within-host immunological dynamics to population-level epidemic behavior:

```
Differential Equations (individual scale, within-host immunology)
    |
    v
ABM — Agent-Based Model (synthetic dynamic networks, individual interactions)
    |
    v
Stochastic Epidemic Model (population scale, CTMC-based inference)
```

Each stage abstracts and compresses the previous. ODE outputs parameterize agent behavior; ABM-simulated outbreak trajectories inform the stochastic epidemic model's inference. The full chain translates immune-system dynamics into population-level predictions.

---

## Key Scientific Objectives

- **Account for susceptibility, transmissibility, and contact patterns** — factors commonly neglected in standard epidemiological network models.
- **Represent within-host immune dynamics** — model how different arms of the immune system respond to infection or vaccination and how those defenses wane over time.
- **Evaluate public health interventions** — test how policies such as vaccination campaigns or social distancing alter epidemic trajectories.
- **Improve computational efficiency** — predict disease spread more efficiently than fully complex simulations while retaining mechanistic accuracy.
- **Explain super-spreader events** — identify when and why super-spreader events occur by linking viral load, symptom onset, and contact network structure.

---

## Technical Concepts and Methods

| Concept | Role in Pipeline |
|---|---|
| ODE (Ordinary Differential Equations) | Within-host immunological dynamics at individual scale |
| Bernoulli Process | Probabilistic modeling of infection events |
| Synthetic Viral Dynamics | Simulated within-host viral trajectories for use where real data cannot be shared |
| Synthetic Dynamic Networks | Simulated contact networks with realistic temporal structure |
| CTMC with SEM (Stochastic Epidemic Model) | Continuous-Time Markov Chain framework for epidemic state transitions |
| CTMC with DA (Data Augmentation) | Bayesian inference using CTMC with missing data imputation |
| Posterior Sampling | Bayesian parameter estimation from epidemic trajectory data |
| Synthetic Stochastic Epidemics | Simulated epidemic realizations used for model validation and testing |
| SBML (Systems Biology Markup Language) | Proposed standard for encoding and sharing differential equation models |

---

## Stage 1: Differential Equations (Within-Host Immunology)

### Input Data

Not explicitly specified in notes. Based on context, inputs are immunological assay measurements at the individual level.

| Field | Description | Type | File Format |
|---|---|---|---|
| Immunological assay readings | Per-individual measurements of immune markers (e.g., antibody titers, T-cell counts, cytokine levels) over time | Numeric time series, high-dimensional vector per individual | `.csv`, `.tsv`, or proprietary assay export formats |
| Model parameters | ODE rate constants (e.g., viral clearance rate, immune activation rate, waning rate) | Numeric scalars/vectors | `.csv`, `.json`, `.xml` (SBML) |
| Initial conditions | Starting immune state per individual at time zero | Numeric vector | `.csv`, `.json` |

**Data model:** One record per individual per timepoint; columns represent distinct immune compartments or assay channels. High-dimensional — potentially dozens to hundreds of columns per row.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Immune state trajectories | Time-series vectors of immune compartment values per individual (e.g., antibody concentration, viral load, effector cell counts) | Numeric time series, one vector per individual per timestep | `.csv`, `.tsv`, `.h5` (HDF5 for high-dimensional data) |
| ODE model definition | Encoded model structure for sharing and reproduction | XML-based markup | `.xml` (SBML) |
| Synthetic assay data | Simulated individual-level immune trajectories generated to substitute for restricted real data | Numeric time series | `.csv`, `.tsv` |

**Data model:** High-dimensional matrix — rows are timesteps, columns are immune state variables, replicated per individual. Downstream consumers (ABM) will need a reduced/summarized form (e.g., scalar infectiousness score per individual per day).

### Model Dependencies

- SBML support mentioned as a potential addition — tools are available to assist with generating SBML-compliant model representations
- Outputs are intended to eventually feed into the ABM as inputs (referred to as "Bruce's outputs")

### Compute Requirements

Not specified for this stage.

### Security and Data Access

- Data **cannot be shared at this time**
- Synthetic data generation is under consideration as a workaround to enable collaboration without exposing restricted data

### Notes

- The high dimensionality of immunological assay outputs poses a dimensionality challenge for downstream integration
- Synthetic data generation, if pursued, must accurately reproduce the statistical properties and covariance structure of real assay outputs to remain scientifically valid

---

## Stage 2: ABM — Agent-Based Model (Dynamic Contact Networks)

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Node attributes | Per-individual demographic or behavioral attributes (e.g., age, risk group, vaccination status) | Tabular, one row per agent | `.csv` (large; hundreds of thousands of rows, 5–6 columns) |
| Edge list | Pairwise contact records with timing — who contacted whom and when | Tabular, one row per contact event | `.csv` |
| Disease process parameters | Transmission probability, latency period, recovery rate, and other spreading parameters | Numeric scalars/vectors | `.csv`, `.json`, R list/environment (`.rds`, `.RData`) |
| ODE-derived individual infectiousness | (Future) Scalar or time-series infectiousness score per individual derived from Stage 1 outputs | Numeric vector or time series | `.csv`, `.tsv` |

**Data model (nodes):** One row per agent; columns include agent ID, type/class, and attributes. Compiled from multiple real-world sources into a single flat CSV.

**Data model (edges):** One row per contact event; columns include source node ID, target node ID, timestamp, and optionally contact duration or setting. This is a temporal edge list, not a static adjacency matrix.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Simulated outbreak realizations | Multiple independent runs of the epidemic simulation, each producing a full trajectory | Tabular or nested list, one record per run | `.csv`, `.rds`, `.RData` |
| Transition timestamps | Per-individual, per-transition event log — records the time each agent moved between disease states (S→E, E→I, I→R, etc.) | Tabular event log, one row per transition event | `.csv` |
| Network evolution records | Snapshots or edge-change logs describing how the contact network changed during the simulation | Tabular edge list with timestamps, or adjacency snapshots | `.csv`, `.graphml`, `.gexf` (no field-wide standard) |
| Epidemic summary statistics | Aggregate counts per timestep (incidence, prevalence, cumulative cases) derived from simulations | Tabular time series | `.csv` |

**Data model (transition log):** One row per event; columns include agent ID, from-state, to-state, and timestamp. This is the primary output format and the intended input for Stage 3.

### Model Dependencies

- Model implemented in **R** with a **Julia** backend that interfaces with R for performance-intensive computation
- The network-generating model can be shared publicly in a more compact form, separate from the full simulation

### Compute Requirements

- Capable of running on a **laptop or local VM** for networks on the order of thousands of agents
- Scaling to larger populations would require additional computational resources (HPC or cloud infrastructure)

### Security and Data Access

- Simulated networks are re-generated from statistical properties driving network tie formation — no real individual-level data is exposed
- Outputs **can be shared publicly**

### Notes

- Models are built from the ground up using agent-based processes developed on real-world networks
- The giant CSV compilation (hundreds of thousands of rows, 5–6 columns) represents aggregated multi-source contact data and is the empirical foundation for network simulation
- The R/Julia split suggests performance bottlenecks in pure R — the Julia backend likely handles the core stochastic simulation loop

---

## Stage 3: Stochastic Epidemic Model (Population-Scale Inference)

### Input Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Aggregate surveillance counts | Daily or weekly case counts, deaths, hospitalizations, or similar population-level observables | Tabular time series, one row per reporting period | `.csv` |
| Individual transition timestamps | (Future / expanded input) Per-individual infection and recovery times from Stage 2 ABM | Tabular event log, one row per transition | `.csv` |
| Prior distributions | Bayesian priors on epidemic parameters (e.g., R0, latency period, recovery rate) | Numeric or distributional specification | `.csv`, `.json`, R script (`.R`) |

**Data model (current):** One row per time period; columns are date/week, case count, death count, and optional stratifiers (age group, region). Coarse-grained — no individual-level resolution.

**Data model (expanded, future):** Individual-level event log from ABM Stage 2; enables data augmentation (DA) to impute missing transition times.

### Output Data

| Field | Description | Type | File Format |
|---|---|---|---|
| Imputed missing events | Estimated times of unobserved infection/recovery events for individuals not directly observed | Tabular event log with uncertainty | `.csv` |
| Parameter estimates | Posterior distributions or point estimates for epidemic parameters (e.g., transmission rate, recovery rate, R0) with credible intervals | Tabular, one row per parameter | `.csv` |
| Predicted outbreak trajectories | Forward-projected epidemic curves under current or counterfactual conditions | Tabular time series with uncertainty bands | `.csv` |
| Posterior samples | Full MCMC sample chains for Bayesian parameter inference | Matrix of samples, one row per iteration | `.csv`, `.rds`, `.RData` |

**Data model (estimates):** One row per parameter; columns include parameter name, point estimate, lower credible interval, upper credible interval. Standard format compatible with public health reporting tools.

### Model Dependencies

- Framework: **Continuous-Time Markov Chain (CTMC)** supporting both:
  - Direct stochastic epidemic model (SEM) formulation
  - Data Augmentation (DA) for Bayesian inference under partial observation
- Posterior sampling methods for parameter inference

### Compute Requirements

- Currently runs on **laptop or local machine**
- Designed to remain **computationally feasible** — scalability and user accessibility are explicit design goals
- Statistics are run in a user-friendly manner, suggesting wrapped tooling or accessible interfaces rather than raw research code

### Security and Data Access

- Not explicitly specified
- Input data is aggregate surveillance data, which is typically already public or de-identified

### Notes

- The deliberate constraint to remain computationally feasible on a laptop distinguishes this from simulation-heavy approaches and is a core design philosophy
- The openness to expanding inputs from upstream stages is the primary integration opportunity for the full pipeline

---

## Integration Points Summary

```
[Within-Host Immunological Assays] <- data access restricted; synthetic data under consideration
        |
   [ODE/Diff Eq Model] <- high-dimensional individual immune state vectors
        |             (SBML encoding possible)
   "Bruce's outputs"
        |
   [ABM Engine — R + Julia]
        | node attributes, edge timing, disease parameters
        | re-simulated networks (publicly shareable)
        |
   [Simulated Outbreak Realizations] <- timestamped transition records
        |
   [Stochastic Epidemic Model — CTMC]
        | currently uses aggregate counts; aims to absorb richer ABM outputs
        |
   [Posterior Estimates, Predicted Outbreaks, Uncertainty Intervals]
```

---

## Data Models and File Types — Consolidated Reference

### Input Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 1 — ODE | Immunological assay readings | Wide table: rows = individuals × timepoints, columns = immune markers | `.csv`, `.tsv`, proprietary assay exports |
| Stage 1 — ODE | ODE parameters | Key-value or flat table of rate constants | `.csv`, `.json`, `.xml` (SBML) |
| Stage 1 — ODE | Initial conditions | One vector per individual | `.csv`, `.json` |
| Stage 2 — ABM | Node attributes | Flat table: one row per agent, 5–6 columns | `.csv` (large; ~100k+ rows) |
| Stage 2 — ABM | Edge / contact list | Temporal edge list: source ID, target ID, timestamp | `.csv` |
| Stage 2 — ABM | Disease parameters | Key-value scalar/vector config | `.csv`, `.json`, `.rds`, `.RData` |
| Stage 2 — ABM | ODE infectiousness (future) | Per-individual scalar or time series | `.csv`, `.tsv` |
| Stage 3 — SEM | Aggregate surveillance counts | Time series table: date, case count, deaths | `.csv` |
| Stage 3 — SEM | Individual transitions (future) | Event log: agent ID, from-state, to-state, timestamp | `.csv` |
| Stage 3 — SEM | Prior distributions | Parameter specification table or distributional config | `.csv`, `.json`, `.R` |

### Output Data Models

| Stage | Data | Schema / Structure | File Format(s) |
|---|---|---|---|
| Stage 1 — ODE | Immune state trajectories | Wide time series: rows = timesteps, columns = immune compartments, replicated per individual | `.csv`, `.tsv`, `.h5` |
| Stage 1 — ODE | SBML model definition | XML-encoded ODE system | `.xml` (SBML) |
| Stage 1 — ODE | Synthetic assay data | Same schema as real assay input | `.csv`, `.tsv` |
| Stage 2 — ABM | Simulated outbreak realizations | Collection of per-run trajectory tables | `.csv`, `.rds`, `.RData` |
| Stage 2 — ABM | Transition event log | Event log: agent ID, from-state, to-state, timestamp | `.csv` |
| Stage 2 — ABM | Network evolution records | Temporal edge list or adjacency snapshots | `.csv`, `.graphml`, `.gexf` |
| Stage 2 — ABM | Epidemic summary statistics | Time series: date, incidence, prevalence, cumulative | `.csv` |
| Stage 3 — SEM | Imputed missing events | Event log with uncertainty estimates | `.csv` |
| Stage 3 — SEM | Parameter estimates | One row per parameter: name, estimate, lower CI, upper CI | `.csv` |
| Stage 3 — SEM | Predicted outbreak trajectories | Time series with uncertainty bands | `.csv` |
| Stage 3 — SEM | Posterior samples | MCMC sample matrix: rows = iterations, columns = parameters | `.csv`, `.rds`, `.RData` |

### Cross-Stage Interface Formats

| Interface | Data Passed | Current Format | Gap / Risk |
|---|---|---|---|
| Stage 1 → Stage 2 | Individual infectiousness or immune state summary ("Bruce's outputs") | **Not yet defined** | Highest-risk interface — schema must be agreed upon |
| Stage 2 → Stage 3 | Individual transition timestamps from simulated outbreaks | `.csv` event log (intended) | Currently not consumed — Stage 3 uses aggregate counts only |
| External → Stage 2 | Multi-source compiled contact data | Large `.csv` (~100k+ rows, 5–6 cols) | No standard schema across sources; internal normalization needed |

---

## Data Sharing and Constraints

| Stage | Shareable? | Constraint |
|---|---|---|
| Diff Eq (ODE) — real assay data | No | Data access restricted; cannot be shared at this time |
| Diff Eq (ODE) — synthetic data | Potentially | Synthetic data generation is under consideration |
| ABM — input network CSV | Partially | Raw multi-source CSV may contain sensitive attributes; re-simulated version is shareable |
| ABM — simulated outputs | Yes | Re-simulated networks based on statistical properties; publicly shareable |
| Stochastic Epidemic Model — inputs | Yes (typically) | Aggregate surveillance counts are usually public |
| Stochastic Epidemic Model — outputs | Yes | Estimates and predictions are research outputs |

---

## Open Questions and Considerations

| Area | Issue |
|---|---|
| ODE-to-ABM interface | The schema for passing "Bruce's outputs" into the ABM is not yet defined — this is the highest-risk integration point |
| Synthetic data fidelity | If real assay data cannot be shared, synthetic data must reproduce covariance structure and marginal distributions accurately to remain scientifically valid |
| ABM input expansion | Stochastic epidemic model currently uses aggregate counts; integrating individual-level ABM transition timestamps would substantially increase model richness but requires format agreement |
| Field standardization | No standard output format exists for dynamic network simulations — an internal schema should be established to ensure downstream compatibility |
| Scaling boundary | The transition from laptop-scale (thousands of agents) to HPC-scale is not defined — the threshold at which local compute becomes insufficient needs characterization |
| R/Julia interface stability | The R-to-Julia bridge introduces a cross-language dependency; version pinning and interface contracts should be documented |
| SBML adoption | SBML is mentioned as a potential addition for the ODE stage — adopting it early would improve model portability and reproducibility |
| Parameter identifiability | Passing high-dimensional immune state vectors through ABM into epidemic inference risks non-identifiability — sensitivity analysis across the full pipeline is needed |
| Super-spreader mechanism | The pipeline claims to explain super-spreader events via viral load and contact networks, but the mechanistic link from ODE viral dynamics through ABM to population-level outbreak statistics requires explicit validation |
