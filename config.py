"""
config.py — Global configuration for the antibody discovery pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from RP1_antibody_pipeline.utils.helpers import load_spike_from_fasta

# ─── Base Paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MD_DIR = BASE_DIR / "md_simulations"
MSM_DIR = BASE_DIR / "msm_analysis"
EVOLUTION_DIR = BASE_DIR / "synthetic_evolution"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
UTILS_DIR = BASE_DIR / "utils"

# Ensure output directories exist
for _d in [DATA_DIR, MD_DIR / "output", MSM_DIR / "output",
           EVOLUTION_DIR / "output", EXPERIMENTS_DIR / "output"]:
    _d.mkdir(parents=True, exist_ok=True)


# ─── Language Model ───────────────────────────────────────────────────────────

@dataclass
class LMConfig:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    max_length: int = 512
    device: str = "cpu"          # "cuda" if GPU available
    top_k: int = 10              # top-k sampling for mutations
    num_sequences: int = 1000    # sequences to generate per round


# ─── VAE ─────────────────────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    input_dim: int = 256         # flattened residue feature dim
    hidden_dim: int = 512
    latent_dim: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32


# ─── GAN ─────────────────────────────────────────────────────────────────────

@dataclass
class GANConfig:
    noise_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 256        # matches VAE input_dim
    learning_rate: float = 2e-4
    epochs: int = 200
    batch_size: int = 32


# ─── BCR Repertoire ───────────────────────────────────────────────────────────

@dataclass
class BCRConfig:
    """
    Configuration for BCR repertoire loading and atlas construction.

    oas_data_dir      : directory containing OAS bulk-download CSV files.
                        Data acquisition — Observed Antibody Space (OAS):
                          1. Visit https://opig.stats.ox.ac.uk/webapps/oas/
                          2. Select 'Bulk download' → choose species/study/
                             isotype (e.g. Human, COVID-19, IgG).
                          3. Download gzipped CSV files; decompress with
                             ``gzip -d *.gz`` and place them in this directory.
                          CSV columns expected: sequence_alignment_aa (VH
                          amino acid sequence) plus optional metadata columns
                          (subject, disease, isotype, etc.).
    private_data_path : path to private BCR repertoire CSV or FASTA (optional).
    disease_label     : label used to tag the resulting atlas (e.g. 'COVID-19').
    max_sequences     : cap on sequences loaded per source (None = no limit).
    atlas_output_path : where to save the disease-specific embedding atlas.
    """
    oas_data_dir: str = str(DATA_DIR / "oas")
    private_data_path: Optional[str] = None
    disease_label: str = "unknown"
    max_sequences: Optional[int] = 10_000
    atlas_output_path: str = str(DATA_DIR / "atlas.pkl")


# ─── Molecular Dynamics ──────────────────────────────────────────────────────

@dataclass
class MDConfig:
    """
    Configuration for molecular dynamics simulations.

    backend           : MD engine to use — 'openmm' | 'gromacs' | 'charmm'.
                        OpenMM:  conda install -c conda-forge openmm
                        GROMACS: conda install -c bioconda gromacs
                        CHARMM:  https://www.charmm.org (academic licence)
    pdb_source        : path to input PDB structure file.
                        Structure acquisition:
                          • Experimental structures (RCSB PDB):
                            https://www.rcsb.org/
                            Search by PDB ID, UniProt accession, or keyword;
                            download the .pdb or .cif file.
                          • Pre-computed structures (AlphaFold DB):
                            https://alphafold.ebi.ac.uk/
                            Covers >200 million proteins; no GPU required.
                            Download as PDB from the entry page or via the
                            AlphaFold API (af2 CLI or direct HTTPS).
                          • De-novo prediction for novel sequences:
                            ESMFold (fast, single-sequence):
                              https://esmatlas.com/resources?action=fold
                              or locally: pip install fair-esm
                            AlphaFold2-Multimer (Ag-Ab complexes):
                              https://github.com/deepmind/alphafold
                        Structure preparation before MD:
                          • PDBFixer (adds missing residues, hydrogens,
                            removes non-standard residues):
                              pip install pdbfixer
                              python -m pdbfixer input.pdb --output clean.pdb
    forcefield        : force-field identifier (OpenMM / GROMACS / CHARMM name).
    """
    backend: str = "openmm"              # openmm | gromacs | charmm
    pdb_source: str = str(MD_DIR / "input.pdb")
    topology_file: str = str(MD_DIR / "topology.pdb")
    trajectory_file: str = str(MD_DIR / "output" / "trajectory.xtc")
    temperature_k: float = 300.0
    simulation_steps: int = 1_000_000
    step_size_fs: float = 2.0            # femtoseconds
    report_interval: int = 1000
    forcefield: str = "amber14-all.xml"  # OpenMM; GROMACS uses .top/.itp files


# ─── MSM ─────────────────────────────────────────────────────────────────────

@dataclass
class MSMConfig:
    lag_time: int = 10           # in trajectory frames
    n_states: int = 20           # number of macro-states
    n_jobs: int = 4


# ─── Synthetic Evolution ──────────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    n_generations: int = 10
    population_size: int = 500
    mutation_rate: float = 0.05
    top_fraction: float = 0.2    # fraction selected each generation
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"


# ─── Repertoire Scale ────────────────────────────────────────────────────────

@dataclass
class RepertoireConfig:
    n_workers: int = os.cpu_count() or 4
    batch_size: int = 256
    embedding_dim: int = 64
    top_candidates: int = 100


# ─── Experimental Validation ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    binding_data_csv: str = str(EXPERIMENTS_DIR / "binding_data.csv")
    output_plot: str = str(EXPERIMENTS_DIR / "output" / "validation_plot.png")
    correlation_method: str = "pearson"  # or "spearman"


# ─── Viral Escape (RP1) ───────────────────────────────────────────────────────

@dataclass
class ViralEscapeConfig:
    """
    Configuration for the viral escape mutant panel — the antigen side of RP1.

    antigen_sequence   : wildtype viral antigen (e.g., RBD, HA head domain).
                         Replace with a real FASTA-loaded sequence in production.
    epitope_residues   : 0-indexed positions the target antibody contacts.
                         Empty list → generator treats full sequence as epitope.
    panel_size         : number of escape mutants to generate.
    max_mutations      : max simultaneous point mutations per variant.
    binding_threshold  : binding score cutoff — below this = escaped.
    """
    # Full spike protein translated from data/SARS-CoV-2_sequences.fasta
    # (Wuhan reference coordinates nt 21562–25384; ~1273 aa).
    antigen_sequence: str = field(default_factory=load_spike_from_fasta)
    # Key ACE2-contact residues in the spike RBD (0-indexed).
    # Covers major escape hotspots: K417, L452, T478, E484, N501 and neighbours.
    # Source: structural data from Lan et al. 2020 + VOC mutation surveys.
    epitope_residues: List[int] = field(
        default_factory=lambda: [
            416, 443, 445, 448, 451, 454, 455,   # K417, Y444, P446, G449, L452, Y455, F456
            477, 483, 489, 492, 493, 495,          # T478, E484, F490, Q493, S494, G496
            497, 499, 500, 501, 502, 504,          # Q498, T500, N501, G502, V503, Y505
        ]
    )
    panel_size: int = 50
    max_mutations: int = 3
    binding_threshold: float = 0.50     # fraction threshold to count as 'covered'


# ─── Immune Blind Spot (RP1) ─────────────────────────────────────────────────

@dataclass
class BlindSpotConfig:
    """
    Settings for immune blind spot analysis.

    blind_spot_threshold : coverage score below which an epitope position is
                           flagged as a blind spot (cosine similarity, [0,1]).
    output_path          : where to save the blind spot JSON report.
    """
    blind_spot_threshold: float = 0.5
    output_path: str = str(EXPERIMENTS_DIR / "output" / "blind_spot_report.json")


# ─── Lab-in-the-Loop (RP1) ────────────────────────────────────────────────────

@dataclass
class LabLoopConfig:
    """
    Settings for laboratory-in-the-loop iterative refinement.

    experimental_csv     : path to CSV with experimental binding results.
                           Columns: sequence, measured_binding (required).
    n_suggestions        : number of sequences to recommend for next round.
    escape_threshold     : normalised binding score below which a sequence
                           is added to the escape panel as a confirmed variant.
    output_dir           : directory for per-iteration JSON reports.
    """
    experimental_csv: Optional[str] = None
    n_suggestions: int = 20
    escape_threshold: float = 0.3
    output_dir: str = str(EXPERIMENTS_DIR / "output" / "lab_loop")


# ─── Antigen-ALM Profiling (RP1) ─────────────────────────────────────────────

@dataclass
class AntigenALMConfig:
    """
    Settings for comparing pathogen antigen sequences against ALM binding
    site representations.

    n_antigen_sequences : number of FASTA variants to profile (0 = all).
    similarity_metric   : 'cosine' | 'dot' | 'euclidean'.
    """
    n_antigen_sequences: int = 3
    similarity_metric: str = "cosine"


# ─── ALM Fine-tuning (RP1) ────────────────────────────────────────────────────

@dataclass
class ALMFinetuneConfig:
    """
    Settings for MD-guided ALM fine-tuning.

    Only antibodies with md_binding_score >= min_binding_score are used as
    positive training examples.  Pairwise margin ranking loss is applied to
    the LM's FFN layers (attention weights are frozen).
    """
    learning_rate: float = 1e-5
    n_epochs: int = 3
    margin: float = 0.1
    min_binding_score: float = 0.3       # threshold for positive training pairs


# ─── Vaccine Design (RP1) ─────────────────────────────────────────────────────

@dataclass
class VaccineDesignConfig:
    """
    Settings for selecting broadly neutralising vaccine candidates.

    min_coverage_fraction : antibody must cover ≥ this fraction of escape panel.
    top_candidates        : maximum number of vaccine candidates to report.
    """
    min_coverage_fraction: float = 0.60
    top_candidates: int = 20


# ─── Master Config ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    lm: LMConfig = field(default_factory=LMConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    md: MDConfig = field(default_factory=MDConfig)
    msm: MSMConfig = field(default_factory=MSMConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    repertoire: RepertoireConfig = field(default_factory=RepertoireConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    viral_escape: ViralEscapeConfig = field(default_factory=ViralEscapeConfig)
    vaccine_design: VaccineDesignConfig = field(default_factory=VaccineDesignConfig)
    bcr: BCRConfig = field(default_factory=BCRConfig)
    antigen_alm: AntigenALMConfig = field(default_factory=AntigenALMConfig)
    alm_finetune: ALMFinetuneConfig = field(default_factory=ALMFinetuneConfig)
    blind_spot: BlindSpotConfig = field(default_factory=BlindSpotConfig)
    lab_loop: LabLoopConfig = field(default_factory=LabLoopConfig)


# Default global config instance
config = PipelineConfig()
