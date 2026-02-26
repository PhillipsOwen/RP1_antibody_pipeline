"""
main.py — Antibody Discovery Pipeline Orchestrator  (RP1 edition)

RP1: Predicting Antibody Responses to Viral Escape Mutants
    BCR repertoire → ALM atlas → Antigen profiling → MD binding → ALM fine-tune
    → Structural modelling → MD/MSM → Evolution → Vaccine design → Validation

Pipeline stages
---------------
  0.  Viral escape panel     — generate escape mutants from the wildtype antigen
  1.  BCR repertoire         — load public (OAS) + private BCR data; build atlas
  2.  Antibody LM            — generate / score antibody sequences
  2a. Antigen-ALM profile    — compare spike variants vs ALM binding sites
  2b. MD binding prediction  — embedding-proxy (or OpenMM/GROMACS/CHARMM) ΔG
  2c. ALM fine-tuning        — align LM PLLs with MD binding scores
  2d. Immune blind spots     — identify epitope regions not covered by repertoire
  3.  Structural modelling   — VAE + GAN conformational space
  4.  MD + MSM               — molecular dynamics & kinetics
  5.  Synthetic evolution    — affinity maturation simulation (escape-aware scorer)
  6.  Repertoire screening   — parallel LM scoring, top candidates
  7.  Cross-reactivity       — score every candidate against the escape panel
  8.  Vaccine design         — select broadly neutralising candidates
  9.  Experimental validation— compare predictions vs. measured data
  10. Lab-in-the-loop        — ingest wet-lab results, refine ALM, suggest next round

Run:
    python -m RP1_antibody_pipeline.main
    python -m RP1_antibody_pipeline.main --mock     # fast dry-run, no large models
    python -m RP1_antibody_pipeline.main --help
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List

import numpy as np

from RP1_antibody_pipeline.config import config, PipelineConfig, MODELS_DIR
from RP1_antibody_pipeline.models.antibody_lm import get_lm
from RP1_antibody_pipeline.models.vae import AntibodyVAE
from RP1_antibody_pipeline.models.gan import AntibodyGAN
from RP1_antibody_pipeline.models.alm_finetuner import ALMFineTuner
from RP1_antibody_pipeline.md_simulations.md_runner import TrajectoryAnalyzer, generate_mock_trajectory
from RP1_antibody_pipeline.md_simulations.binding_md import BindingMDPredictor
from RP1_antibody_pipeline.md_simulations.structural_pathways import AgAbComplexBuilder, BindingPathwaySimulator
from RP1_antibody_pipeline.msm_analysis.msm_builder import MSMBuilder
from RP1_antibody_pipeline.synthetic_evolution.evolution import RepertoireEvolver, diversity_score
from RP1_antibody_pipeline.viral_escape.escape_mutant import EscapeMutantGenerator, EscapeMutant
from RP1_antibody_pipeline.viral_escape.binding_predictor import CrossReactivityScorer
from RP1_antibody_pipeline.viral_escape.antigen_profile import AntigenBindingSiteProfiler
from RP1_antibody_pipeline.viral_escape.blind_spot import BlindSpotAnalyzer
from RP1_antibody_pipeline.data.bcr_loader import load_repertoire, BCRRepertoire
from RP1_antibody_pipeline.experiments.lab_loop import LabInTheLoop
from RP1_antibody_pipeline.experiments.validation import (
    ValidationDataset, generate_report, plot_score_distribution,
    generate_escape_report,
)
from RP1_antibody_pipeline.utils.helpers import (
    setup_logging, save_sequences, load_sequences,
    parallel_map, deduplicate, load_all_spike_sequences_from_fasta,
)
from RP1_antibody_pipeline.utils.checkpoint_manager import (
    CheckpointManager,
    save_stage_0_escape_panel,
    save_stage_1_bcr_repertoire,
    save_stage_2_lm_scoring,
    save_stage_2a_antigen_profile,
    save_stage_2b_md_binding,
    save_stage_2c_alm_finetune,
    save_stage_2d_blind_spots,
    save_stage_2_5_pathways,
    save_stage_3_structure,
    save_stage_4_msm,
    save_stage_5_evolution,
    save_stage_6_screening,
    save_stage_7_cross_reactivity,
    save_stage_8_vaccine_design,
    save_stage_9_validation,
    save_stage_10_lab_loop,
)

logger = logging.getLogger(__name__)

# ─── Seed sequences (VH-like, length 120) ────────────────────────────────────
# In a real run these would be loaded from a FASTA or CSV file.

EXAMPLE_SEEDS = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQGLEWMGRIDPNSGGTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR",
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVSNIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR",
]


# ─── Pipeline stages ─────────────────────────────────────────────────────────

def stage_bcr_repertoire(
    cfg: PipelineConfig,
    mock: bool = False,
) -> tuple[BCRRepertoire, dict]:
    """
    Stage 1 — Load BCR repertoire data and build a disease-specific atlas.

    Loads public OAS data from cfg.bcr.oas_data_dir (if the directory exists)
    and any private data from cfg.bcr.private_data_path, then builds an
    embedding atlas representing the disease-specific antibody landscape.

    In mock mode a small synthetic repertoire is used instead.

    Returns
    -------
    repertoire : BCRRepertoire with loaded sequences.
    atlas      : disease-specific embedding atlas dict (from lm.build_atlas).
    """
    logger.info("=== Stage 1: BCR Repertoire Loading & Atlas Construction ===")
    bcr_cfg = cfg.bcr

    if mock:
        # Generate a tiny synthetic repertoire for pipeline testing
        from RP1_antibody_pipeline.viral_escape.escape_mutant import AA_ALPHABET
        synthetic = [
            "".join(random.choices(AA_ALPHABET, k=random.randint(90, 110)))
            for _ in range(20)
        ]
        from RP1_antibody_pipeline.data.bcr_loader import BCRSequence
        sequences = [
            BCRSequence(sequence_id=f"mock_{i}", sequence_aa=s,
                        disease=bcr_cfg.disease_label, source="mock")
            for i, s in enumerate(synthetic)
        ]
        repertoire = BCRRepertoire(sequences=sequences,
                                   disease_label=bcr_cfg.disease_label)
    else:
        all_seqs = []
        oas_dir = bcr_cfg.oas_data_dir
        if oas_dir and __import__("pathlib").Path(oas_dir).exists():
            rep = load_repertoire("oas", oas_dir, bcr_cfg.disease_label,
                                  max_sequences=bcr_cfg.max_sequences)
            all_seqs.extend(rep.sequences)
            logger.info("OAS: %d sequences loaded.", len(rep.sequences))
        else:
            logger.info("OAS data directory not found (%s) — skipping.", oas_dir)
            logger.info(
                "To load real BCR data, download OAS bulk CSV files from "
                "https://opig.stats.ox.ac.uk/webapps/oas/ and place them in: %s",
                oas_dir,
            )

        if bcr_cfg.private_data_path:
            p = __import__("pathlib").Path(bcr_cfg.private_data_path)
            fmt = "fasta" if p.suffix.lower() in (".fa", ".fasta") else "csv"
            priv = load_repertoire(fmt, str(p), bcr_cfg.disease_label,
                                   max_sequences=bcr_cfg.max_sequences)
            all_seqs.extend(priv.sequences)
            logger.info("Private data: %d sequences loaded.", len(priv.sequences))

        if not all_seqs:
            logger.warning("No BCR data loaded — atlas will be empty.")
        from RP1_antibody_pipeline.data.bcr_loader import BCRSequence
        repertoire = BCRRepertoire(sequences=all_seqs,
                                   disease_label=bcr_cfg.disease_label)

    # Build atlas using LM embeddings
    lm = get_lm(use_mock=mock)
    seq_strs = repertoire.get_sequences()
    atlas = lm.build_atlas(seq_strs or ["PLACEHOLDER"],
                           disease_label=bcr_cfg.disease_label)

    logger.info(
        "BCR atlas: disease=%s  n_sequences=%d  embedding_dim=%d",
        bcr_cfg.disease_label, len(seq_strs), atlas["centroid"].shape[0],
    )
    return repertoire, atlas


def stage_antigen_alm_profile(
    cfg: PipelineConfig,
    antibody_sequences: List[str],
    lm,
    mock: bool = False,
) -> np.ndarray:
    """
    Stage 2a — Compare pathogen antigen variants against ALM binding sites.

    Loads spike protein sequences from the FASTA file, then builds an
    (n_antibodies × n_antigen_variants) affinity matrix using the ALM's
    embedding space to score binding compatibility at the defined epitope.

    Returns
    -------
    np.ndarray of shape (n_antibodies, n_antigen_variants), values in [0, 1].
    """
    logger.info("=== Stage 2a: Antigen-ALM Binding Site Profile ===")
    alm_cfg = cfg.antigen_alm

    n_load = None if alm_cfg.n_antigen_sequences == 0 else alm_cfg.n_antigen_sequences
    antigen_seqs = load_all_spike_sequences_from_fasta(max_records=n_load)

    if not antigen_seqs:
        logger.warning("No antigen sequences loaded from FASTA — using wildtype only.")
        antigen_seqs = [cfg.viral_escape.antigen_sequence]

    if mock:
        antigen_seqs = antigen_seqs[:2]

    profiler = AntigenBindingSiteProfiler(
        lm=lm,
        epitope_residues=cfg.viral_escape.epitope_residues,
        similarity_metric=alm_cfg.similarity_metric,
    )
    affinity_matrix = profiler.build_affinity_matrix(antibody_sequences, antigen_seqs)
    logger.info(
        "Antigen-ALM profile: %d antibodies × %d antigen variants  mean=%.3f",
        affinity_matrix.shape[0], affinity_matrix.shape[1], affinity_matrix.mean(),
    )
    return affinity_matrix


def stage_md_binding_prediction(
    cfg: PipelineConfig,
    antibody_sequences: List[str],
    antigen_sequences: List[str],
    lm,
    mock: bool = False,
) -> np.ndarray:
    """
    Stage 2b — Predict antibody-antigen binding scores via MD.

    Uses embedding-space distance as an MD proxy (mock mode) or delegates to
    the configured MD backend (real mode) to compute a binding score matrix.

    Returns
    -------
    np.ndarray of shape (n_antibodies, n_antigens), values in [0, 1].
    """
    logger.info("=== Stage 2b: MD Binding Prediction ===")

    predictor = BindingMDPredictor(
        lm=lm,
        mock=mock or cfg.md.backend == "openmm",
        temperature_k=cfg.md.temperature_k,
    )
    binding_matrix = predictor.predict_binding_scores(
        antibody_sequences, antigen_sequences
    )
    top = predictor.top_pairs(antibody_sequences, antigen_sequences,
                               binding_matrix, top_n=5)
    logger.info(
        "MD binding: top pair score=%.3f  matrix shape=%s  mean=%.3f",
        top[0][2] if top else 0.0, binding_matrix.shape, binding_matrix.mean(),
    )
    return binding_matrix


def stage_alm_finetune(
    cfg: PipelineConfig,
    antibody_sequences: List[str],
    binding_matrix: np.ndarray,
    lm,
    mock: bool = False,
) -> ALMFineTuner:
    """
    Stage 2c — Fine-tune the ALM using MD-predicted binding scores.

    Takes the mean binding score across all antigen variants as the per-
    antibody training signal, then applies pairwise margin ranking loss to
    the LM's FFN layers.

    Returns
    -------
    ALMFineTuner instance (call .score_with_finetuning() for adjusted scores).
    """
    logger.info("=== Stage 2c: ALM Fine-tuning (MD-guided) ===")
    ft_cfg = cfg.alm_finetune

    # Aggregate per-antibody binding score (mean over antigen variants)
    mean_binding = binding_matrix.mean(axis=1).tolist()

    # Filter to sequences above the min_binding_score threshold
    train_seqs = [s for s, sc in zip(antibody_sequences, mean_binding)
                  if sc >= ft_cfg.min_binding_score]
    train_scores = [sc for sc in mean_binding
                    if sc >= ft_cfg.min_binding_score]

    if len(train_seqs) < 2:
        logger.warning(
            "Fewer than 2 sequences above min_binding_score=%.2f — "
            "fine-tuning skipped.", ft_cfg.min_binding_score
        )
        train_seqs = antibody_sequences[:max(2, len(antibody_sequences))]
        train_scores = mean_binding[:len(train_seqs)]

    finetuner = ALMFineTuner(
        lm=lm,
        learning_rate=ft_cfg.learning_rate,
        n_epochs=ft_cfg.n_epochs,
        margin=ft_cfg.margin,
        mock=mock,
    )
    history = finetuner.finetune(train_seqs, train_scores)
    final_loss = history["loss_history"][-1] if history["loss_history"] else float("nan")
    logger.info(
        "ALM fine-tuning complete: trained on %d sequences  final_loss=%.4f",
        len(train_seqs), final_loss,
    )
    return finetuner


def stage_blind_spot_analysis(
    cfg: PipelineConfig,
    atlas: dict,
    antigen_sequences: List[str],
    lm,
    mock: bool = False,
) -> dict:
    """
    Stage 2d — Predict immune blind spots in the pathogen antigen.

    Compares the BCR atlas (typical human repertoire landscape) against the
    antigen epitope to identify regions of low repertoire coverage.

    Returns
    -------
    blind_spot_report dict from BlindSpotAnalyzer.analyze().
    """
    logger.info("=== Stage 2d: Immune Blind Spot Analysis ===")
    bs_cfg = cfg.blind_spot

    antigens = antigen_sequences[:2] if mock else antigen_sequences
    if not antigens:
        antigens = [cfg.viral_escape.antigen_sequence]

    analyzer = BlindSpotAnalyzer(
        lm=lm,
        atlas=atlas,
        epitope_residues=cfg.viral_escape.epitope_residues,
        blind_spot_threshold=bs_cfg.blind_spot_threshold,
    )
    report = analyzer.analyze(antigens, output_path=bs_cfg.output_path)

    logger.info(
        "Blind spot summary: %s",
        analyzer.summary_string(report),
    )
    return report


def stage_structural_pathways(
    cfg: PipelineConfig,
    top_seqs: List[str],
    antigen_seqs: List[str],
    lm,
    mock: bool = False,
) -> dict:
    """
    Stage 2.5 — Build synthetic Ag-Ab structural complexes and simulate
    binding pathways.

    Constructs mock (or real, when PDB structures are available) structural
    feature representations for representative antibody–antigen pairs, then
    simulates binding/unbinding pathways and builds a pathway-level MSM to
    characterise binding kinetics.  This MSM is separate from the
    conformational MSM constructed in stages 3/4.

    Parameters
    ----------
    top_seqs     : antibody sequences to pair with antigens.
    antigen_seqs : antigen sequences.
    lm           : AntibodyLM instance (reserved for future embedding use).
    mock         : if True, use fast mock structures and short trajectories.

    Returns
    -------
    dict with keys:
      pathway_msm        : MSMBuilder fitted on pathway features.
      complex_features   : list of feature dicts (one per Ag-Ab pair).
      pathway_timescales : top MSM implied timescales (np.ndarray).
      free_energy        : negative log of stationary distribution (np.ndarray).
    """
    logger.info("=== Stage 2.5: Ag-Ab Structural Pathways ===")

    if not top_seqs or not antigen_seqs:
        logger.warning("Empty antibody or antigen sequences — skipping stage 2.5.")
        return {
            "pathway_msm": None,
            "complex_features": [],
            "pathway_timescales": np.array([]),
            "free_energy": np.array([]),
        }

    # Limit pair count for speed
    n_ab = min(5 if mock else 20, len(top_seqs))
    n_ag = min(2 if mock else 5, len(antigen_seqs))
    ab_sample = top_seqs[:n_ab]
    ag_sample = antigen_seqs[:n_ag]

    # Cross-product: each antibody paired with each antigen
    pair_ab = [ab for ab in ab_sample for _ in ag_sample]
    pair_ag = [ag for _ in ab_sample for ag in ag_sample]

    # ── Build Ag-Ab complexes ─────────────────────────────────────────────
    builder = AgAbComplexBuilder(n_interface_residues=15 if mock else 20)
    complex_features = builder.batch_build(
        pair_ab, pair_ag, mock=True   # always mock: no PDBs in current pipeline
    )
    logger.info("Built %d Ag-Ab complexes (%d Ab × %d Ag).",
                len(complex_features), n_ab, n_ag)

    # ── Simulate binding pathways ─────────────────────────────────────────
    n_frames = 100 if mock else 200
    simulator = BindingPathwaySimulator(n_features=20)
    pathways = [
        simulator.simulate_mock_pathway(feat, n_frames=n_frames)
        for feat in complex_features
    ]

    # ── Featurize and stack for MSM ───────────────────────────────────────
    featurized = [simulator.featurize_pathway(traj) for traj in pathways]
    combined = np.vstack(featurized)   # (n_pairs * n_frames, n_features)

    # ── Build pathway-level MSM ───────────────────────────────────────────
    n_features_out = combined.shape[1]
    tica_dim = min(5, n_features_out - 1)
    n_micro = min(20, max(2, len(combined) // 5))
    pathway_msm = MSMBuilder(
        lag_time=max(1, cfg.msm.lag_time // 2),
        n_states=min(10, max(2, len(combined) // 10)),
        n_jobs=cfg.msm.n_jobs,
    )
    projected = pathway_msm.tica(combined, dim=tica_dim)
    dtrajs = pathway_msm.cluster(projected, n_micro=n_micro)
    pathway_msm.estimate(dtrajs)

    ts = pathway_msm.timescales
    ts_top = ts[:min(3, len(ts))]
    pi = pathway_msm.stationary_distribution
    free_energy = -np.log(np.maximum(pi, 1e-10))

    logger.info(
        "Pathway MSM: %d pairs × %d frames = %d total  "
        "top timescales=%s",
        len(complex_features), n_frames, len(combined), np.round(ts_top, 1),
    )
    return {
        "pathway_msm": pathway_msm,
        "complex_features": complex_features,
        "pathway_timescales": ts_top,
        "free_energy": free_energy,
    }


def stage_lab_loop(
    cfg: PipelineConfig,
    sequences: List[str],
    scores: List[float],
    finetuner,
    escape_panel: list,
    lm,
    mock: bool = False,
) -> dict:
    """
    Stage 10 — Laboratory-in-the-loop iterative refinement.

    Ingests experimental binding data (real CSV or mock), re-fine-tunes
    the ALM with real measurements, suggests the next sequences to test,
    and updates the escape panel with confirmed escape variants.

    Returns
    -------
    dict from LabInTheLoop.run_iteration().
    """
    logger.info("=== Stage 10: Lab-in-the-Loop Validation & Refinement ===")
    ll_cfg = cfg.lab_loop

    lab = LabInTheLoop(
        lm=lm,
        finetuner=finetuner,
        escape_panel=escape_panel,
        output_dir=ll_cfg.output_dir,
        mock=mock,
    )

    result = lab.run_iteration(
        candidate_sequences=sequences,
        predicted_scores=scores,
        experimental_csv=ll_cfg.experimental_csv,
        n_suggestions=ll_cfg.n_suggestions,
    )

    logger.info(
        "Lab loop: r_before=%.3f  r_after=%.3f  new_escape=%d  "
        "suggestions=%d  report=%s",
        result.get("pearson_r_before", float("nan")),
        result.get("pearson_r_after", float("nan")),
        result.get("updated_escape", 0),
        len(result.get("suggested_next", [])),
        result.get("report_path", ""),
    )
    return result


def stage_lm(cfg: PipelineConfig, seeds: List[str],
             mock: bool = False) -> tuple[List[str], List[float]]:
    """Stage 1 — Generate mutations and score with Antibody LM."""
    logger.info("=== Stage 1: Antibody Language Model ===")
    lm = get_lm(model_name=cfg.lm.model_name, device=cfg.lm.device,
                use_mock=mock)
    all_seqs, all_scores = list(seeds), []

    for seed in seeds:
        mutations = lm.generate_mutations(
            seed_sequence=seed,
            n_mutations=1,
            n_samples=cfg.lm.num_sequences // len(seeds),
            top_k=cfg.lm.top_k,
        )
        for seq, score in mutations:
            all_seqs.append(seq)
            all_scores.append(score)

    # Score the seeds themselves
    seed_scores = lm.score(seeds)
    all_scores = seed_scores + all_scores
    all_seqs, all_scores = deduplicate(all_seqs,
                                        all_scores[len(seeds):] + seed_scores)
    # re-pair correctly
    seed_scores_all = lm.score(all_seqs)
    logger.info("LM stage: %d unique sequences", len(all_seqs))
    return all_seqs, seed_scores_all


def stage_structure(cfg: PipelineConfig, sequences: List[str],
                    mock: bool = False) -> np.ndarray:
    """Stage 2 — Learn conformational latent space (VAE + GAN)."""
    logger.info("=== Stage 2: Structural Modelling (VAE / GAN) ===")

    # Feature generation: use one-hot + random projection as a stand-in for
    # real structural descriptors (AlphaFold / RoseTTAFold output)
    from RP1_antibody_pipeline.utils.helpers import one_hot_encode
    max_len = max(len(s) for s in sequences[:100])
    raw = one_hot_encode(sequences[:100], max_len=max_len)
    features = raw.reshape(raw.shape[0], -1)[:, :cfg.vae.input_dim]
    # Pad if too short
    if features.shape[1] < cfg.vae.input_dim:
        pad = np.zeros((features.shape[0],
                        cfg.vae.input_dim - features.shape[1]))
        features = np.hstack([features, pad])

    # VAE — load from checkpoint if available (non-mock mode only)
    checkpoint_path = str(MODELS_DIR / "vae_checkpoint.pt")
    if not mock and Path(checkpoint_path).exists():
        logger.info("Loading VAE checkpoint from %s", checkpoint_path)
        vae = AntibodyVAE.load_checkpoint(checkpoint_path)
    else:
        vae = AntibodyVAE(
            input_dim=cfg.vae.input_dim,
            hidden_dim=cfg.vae.hidden_dim,
            latent_dim=cfg.vae.latent_dim,
        )
        epochs = 5 if mock else cfg.vae.epochs
        vae.fit(features, epochs=epochs, batch_size=cfg.vae.batch_size,
                lr=cfg.vae.learning_rate)
        if not mock:
            vae.save_checkpoint(checkpoint_path)
    latent_embeddings = vae.get_latent_embeddings(features)
    synthetic_conformations = vae.generate_structures(n=50)
    logger.info("VAE: latent shape=%s  synthetic=%d",
                latent_embeddings.shape, len(synthetic_conformations))

    # GAN (refine realism of synthetic structures)
    gan = AntibodyGAN(
        noise_dim=cfg.gan.noise_dim,
        hidden_dim=cfg.gan.hidden_dim,
        output_dim=cfg.gan.output_dim,
    )
    epochs_gan = 10 if mock else cfg.gan.epochs
    gan.fit(features, epochs=epochs_gan, batch_size=cfg.gan.batch_size)
    gan_samples = gan.generate(n=50)
    logger.info("GAN: generated %d synthetic structures", len(gan_samples))

    return latent_embeddings


def stage_msm(cfg: PipelineConfig, mock: bool = False) -> MSMBuilder:
    """Stage 3 — MD trajectory loading + MSM construction."""
    logger.info("=== Stage 3: MD Simulation & MSM ===")

    topo = cfg.md.topology_file
    traj = cfg.md.trajectory_file

    if mock or not Path(topo).exists():
        logger.info("No MD files found — using mock trajectory.")
        if not mock:
            logger.info(
                "To use real MD data: download structures from "
                "https://www.rcsb.org/ (experimental) or "
                "https://alphafold.ebi.ac.uk/ (predicted). "
                "Prepare with PDBFixer (pip install pdbfixer). "
                "Place topology at '%s' and trajectory at '%s'.",
                topo, traj,
            )
        features = generate_mock_trajectory(n_frames=500, n_features=50)
    else:
        analyzer = TrajectoryAnalyzer(topo, traj)
        logger.info("Trajectory summary: %s", analyzer.summary())
        features = analyzer.featurize(method="distances")

    msm = MSMBuilder(
        lag_time=cfg.msm.lag_time,
        n_states=cfg.msm.n_states,
        n_jobs=cfg.msm.n_jobs,
    )
    projected = msm.tica(features, dim=10)
    dtrajs = msm.cluster(projected, n_micro=min(50, len(projected) // 5))
    msm.estimate(dtrajs)

    ts = msm.timescales[:5]
    logger.info("MSM timescales (frames): %s", np.round(ts, 1))
    return msm


def stage_evolution(cfg: PipelineConfig, seeds: List[str],
                    scorer, mock: bool = False) -> List[str]:
    """Stage 4 — Synthetic repertoire evolution."""
    logger.info("=== Stage 4: Synthetic Repertoire Evolution ===")

    n_gen = 2 if mock else cfg.evolution.n_generations
    pop_size = 20 if mock else cfg.evolution.population_size

    evolver = RepertoireEvolver(
        scorer=scorer,
        mutation_fn="cdr",
        mutation_rate=cfg.evolution.mutation_rate,
        n_generations=n_gen,
        population_size=pop_size,
        top_fraction=cfg.evolution.top_fraction,
    )
    history = evolver.run(seeds)
    top = evolver.top_candidates(history, n=cfg.repertoire.top_candidates)

    top_seqs = [ab.sequence for ab in top]
    top_scores = [ab.score for ab in top]
    div = diversity_score(top_seqs)
    logger.info("Evolution: %d candidates | diversity=%.3f | best=%.4f",
                len(top_seqs), div, max(top_scores))

    save_sequences(
        top_seqs,
        str(cfg.md.topology_file).replace("topology.pdb", "") + "evolved_candidates.csv",
        scores=top_scores,
    )
    return top_seqs


def stage_repertoire_screen(cfg: PipelineConfig,
                             sequences: List[str],
                             scorer) -> tuple[List[str], List[float]]:
    """Stage 5 — Repertoire-scale parallel scoring and filtering."""
    logger.info("=== Stage 5: Repertoire-Scale Screening ===")

    def _eval(seq: str) -> float:
        return scorer([seq])[0]

    scores = parallel_map(_eval, sequences, n_workers=cfg.repertoire.n_workers)
    pairs = sorted(zip(sequences, scores), key=lambda x: x[1], reverse=True)
    top_seqs = [p[0] for p in pairs[:cfg.repertoire.top_candidates]]
    top_scores = [p[1] for p in pairs[:cfg.repertoire.top_candidates]]
    logger.info("Repertoire screen: top %d / %d candidates",
                len(top_seqs), len(sequences))
    return top_seqs, top_scores


def stage_escape_panel(cfg: PipelineConfig,
                       mock: bool = False) -> List[EscapeMutant]:
    """
    Stage 0 — Generate the viral escape mutant panel.

    Produces a ranked set of antigen variants that are predicted to evade
    antibody detection based on mutations at the antibody-contacting epitope.
    In mock mode a small panel (10 mutants) is returned quickly.
    """
    logger.info("=== Stage 0: Viral Escape Mutant Panel ===")

    esc_cfg = cfg.viral_escape
    panel_size = 10 if mock else esc_cfg.panel_size

    generator = EscapeMutantGenerator(
        wildtype_sequence=esc_cfg.antigen_sequence,
        epitope_residues=esc_cfg.epitope_residues,
        panel_size=panel_size,
        max_mutations=esc_cfg.max_mutations,
    )
    panel = generator.generate_panel()

    hotspots = generator.mutation_hotspots(n_top=5)
    logger.info(
        "Escape panel: %d mutants | top hotspot positions: %s",
        len(panel),
        [pos for pos, _ in hotspots],
    )
    return panel


def stage_cross_reactivity(
    cfg: PipelineConfig,
    antibody_sequences: List[str],
    escape_panel: List[EscapeMutant],
    lm,
    mock: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Stage 6 — Score every candidate antibody against the full escape panel.

    Answers: *which antibodies can still bind viral escape mutants?*

    Returns
    -------
    coverage_matrix : np.ndarray of shape (n_antibodies, n_variants)
    adaptation_summary : dict from CrossReactivityScorer.predict_immune_adaptation()
    """
    logger.info("=== Stage 6: Cross-Reactivity Against Escape Panel ===")

    scorer = CrossReactivityScorer(
        lm=lm,
        epitope_residues=cfg.viral_escape.epitope_residues,
        binding_threshold=cfg.viral_escape.binding_threshold,
    )

    coverage_matrix = scorer.build_coverage_matrix(antibody_sequences, escape_panel)

    adaptation = scorer.predict_immune_adaptation(
        antibody_sequences, escape_panel, coverage_matrix=coverage_matrix
    )

    logger.info(
        "Immune adaptation: broadly_neutralising=%.1f%%  mean_panel_coverage=%.3f",
        100.0 * adaptation["fraction_broadly_neutralising"],
        adaptation["mean_panel_coverage"],
    )
    logger.info(
        "Most vulnerable variants (lowest mean binding): positions %s",
        adaptation["most_vulnerable_variants"],
    )
    return coverage_matrix, adaptation


def stage_vaccine_design(
    cfg: PipelineConfig,
    antibody_sequences: List[str],
    escape_panel: List[EscapeMutant],
    lm,
    coverage_matrix: np.ndarray,
) -> List[str]:
    """
    Stage 7 — Select broadly neutralising vaccine candidates.

    Criteria: candidate must cover ≥ min_coverage_fraction of the escape panel
    and is ranked by mean binding score across all variants.

    Returns
    -------
    List of selected vaccine candidate sequences (best-first).
    """
    logger.info("=== Stage 7: Vaccine Candidate Design ===")

    vax_cfg = cfg.vaccine_design
    scorer = CrossReactivityScorer(
        lm=lm,
        epitope_residues=cfg.viral_escape.epitope_residues,
        binding_threshold=cfg.viral_escape.binding_threshold,
    )

    candidates = scorer.vaccine_candidates(
        antibody_sequences=antibody_sequences,
        escape_panel=escape_panel,
        coverage_matrix=coverage_matrix,
        min_coverage=vax_cfg.min_coverage_fraction,
        top_n=vax_cfg.top_candidates,
    )

    selected_seqs = [seq for seq, _, _ in candidates]
    if selected_seqs:
        best_cov = candidates[0][1]
        best_score = candidates[0][2]
        logger.info(
            "Vaccine design: %d candidates selected | best coverage=%.1f%% "
            "mean_binding=%.3f",
            len(selected_seqs), best_cov * 100, best_score,
        )
    else:
        logger.warning(
            "No candidates met coverage threshold (%.0f%%). "
            "Consider lowering min_coverage_fraction in config.",
            vax_cfg.min_coverage_fraction * 100,
        )

    return selected_seqs


def stage_validation(cfg: PipelineConfig, sequences: List[str],
                     predicted_scores: List[float]) -> str:
    """Stage 6 — Experimental validation comparison (mock data if no CSV)."""
    logger.info("=== Stage 6: Experimental Validation ===")

    binding_csv = cfg.experiment.binding_data_csv
    if Path(binding_csv).exists():
        ds = ValidationDataset.from_csv(binding_csv)
    else:
        logger.info("No experimental data CSV found — generating mock data.")
        # Simulate experimental values correlated with predictions + noise
        np.random.seed(42)
        exp = np.array(predicted_scores) + np.random.randn(len(predicted_scores))
        ds = ValidationDataset(
            sequences=sequences,
            predicted=predicted_scores,
            experimental=exp.tolist(),
        )

    report_path = generate_report(ds, output_dir=str(
        Path(cfg.experiment.output_plot).parent))
    return report_path


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_pipeline(seeds: List[str] | None = None,
                 cfg: PipelineConfig | None = None,
                 mock: bool = False,
                 save_checkpoints: bool = True,
                 checkpoint_dir: str = "experiments/checkpoints") -> dict:
    """
    Run the complete RP1 antibody discovery pipeline.

    RP1: Predicting Antibody Responses to Viral Escape Mutants
    Pipeline: MD → Antibody LMs → Experimental validation

    Parameters
    ----------
    seeds : seed antibody sequences (uses EXAMPLE_SEEDS if None)
    cfg   : PipelineConfig (uses module default if None)
    mock  : if True, run with lightweight mock models (fast, no GPU required)
    save_checkpoints : if True, save intermediate data at each stage milestone
    checkpoint_dir : directory to store checkpoint data

    Returns
    -------
    dict with keys:
      sequences            : top antibody sequences
      scores               : their LM scores
      latent_embeddings    : VAE latent space embeddings
      escape_panel         : list of viral escape mutants tested
      coverage_matrix      : (n_ab × n_variants) binding probability matrix
      adaptation_summary   : immune adaptation stats
      vaccine_candidates   : broadly neutralising candidate sequences
      report_path          : path to validation JSON report
      escape_report_path   : path to escape coverage report
    """
    setup_logging()
    cfg = cfg or config
    seeds = seeds or EXAMPLE_SEEDS

    logger.info("RP1 Antibody–Escape Pipeline  |  mock=%s", mock)
    logger.info("Seeds: %d sequences", len(seeds))

    # Initialize checkpoint manager
    checkpoint_manager = None
    if save_checkpoints:
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        logger.info("Checkpoints will be saved to: %s", checkpoint_manager.run_dir)

    # ── Stage 0: Viral escape panel ──
    escape_panel = stage_escape_panel(cfg, mock=mock)
    if checkpoint_manager:
        save_stage_0_escape_panel(checkpoint_manager, escape_panel, cfg)

    # ── Stage 1: BCR repertoire + atlas ──
    repertoire, atlas = stage_bcr_repertoire(cfg, mock=mock)
    if checkpoint_manager:
        save_stage_1_bcr_repertoire(checkpoint_manager, repertoire, atlas, cfg)

    # ── Stage 2: LM scoring / generation ──
    sequences, lm_scores = stage_lm(cfg, seeds, mock=mock)
    if checkpoint_manager:
        save_stage_2_lm_scoring(checkpoint_manager, sequences, lm_scores, cfg)

    # Build a unified LM instance and score cache for downstream stages
    lm = get_lm(use_mock=mock)
    score_map = dict(zip(sequences, lm_scores))

    # Escape-aware scorer: blend LM fitness with cross-reactivity against panel
    # alpha=0.7 weights LM fitness more heavily; reduce to increase breadth pressure
    _xr_scorer = CrossReactivityScorer(
        lm=lm,
        epitope_residues=cfg.viral_escape.epitope_residues,
        binding_threshold=cfg.viral_escape.binding_threshold,
    )
    alpha = 0.7

    def scorer(seqs: List[str]) -> List[float]:
        results = []
        for s in seqs:
            lm_score = score_map[s] if s in score_map else lm.score([s])[0]
            xr_score = _xr_scorer.coverage_fraction(s, escape_panel)
            # Normalise LM score to [0,1] range for blending
            from RP1_antibody_pipeline.viral_escape.binding_predictor import _normalise_lm_score
            lm_norm = _normalise_lm_score(lm_score)
            blended = alpha * lm_norm + (1 - alpha) * xr_score
            results.append(blended)
        return results

    # Collect antigen sequences for stages 2a, 2b, 2.5
    _n_ag = cfg.antigen_alm.n_antigen_sequences or 3
    _antigen_seqs = load_all_spike_sequences_from_fasta(max_records=_n_ag) \
        or [cfg.viral_escape.antigen_sequence]
    if mock:
        _antigen_seqs = _antigen_seqs[:2]

    # ── Stage 2a: Antigen-ALM binding site profile ──
    affinity_matrix = stage_antigen_alm_profile(cfg, sequences, lm, mock=mock)
    if checkpoint_manager:
        save_stage_2a_antigen_profile(checkpoint_manager, affinity_matrix, sequences, _antigen_seqs)

    # ── Stage 2b: MD binding prediction ──
    binding_matrix = stage_md_binding_prediction(
        cfg, sequences, _antigen_seqs, lm, mock=mock
    )
    if checkpoint_manager:
        save_stage_2b_md_binding(checkpoint_manager, binding_matrix, sequences, _antigen_seqs)

    # ── Stage 2.5: Ag-Ab structural pathways ──
    pathway_result = stage_structural_pathways(
        cfg, sequences, _antigen_seqs, lm, mock=mock
    )
    if checkpoint_manager:
        save_stage_2_5_pathways(checkpoint_manager, pathway_result)

    # ── Stage 2c: ALM fine-tuning ──
    finetuner = stage_alm_finetune(cfg, sequences, binding_matrix, lm, mock=mock)
    if checkpoint_manager:
        save_stage_2c_alm_finetune(checkpoint_manager, finetuner, sequences, cfg)

    # Rebuild score_map using fine-tuned scores
    lm_scores_ft = finetuner.score_with_finetuning(sequences)
    score_map = dict(zip(sequences, lm_scores_ft))

    # ── Stage 2d: Immune blind spot analysis ──
    blind_spot_report = stage_blind_spot_analysis(
        cfg, atlas, _antigen_seqs, lm, mock=mock
    )
    if checkpoint_manager:
        save_stage_2d_blind_spots(checkpoint_manager, blind_spot_report)

    # ── Stage 3: Structure (VAE/GAN) ──
    latent = stage_structure(cfg, sequences, mock=mock)
    if checkpoint_manager:
        save_stage_3_structure(checkpoint_manager, latent, sequences, cfg)

    # ── Stage 4: MD + MSM ──
    msm = stage_msm(cfg, mock=mock)
    if checkpoint_manager:
        save_stage_4_msm(checkpoint_manager, msm, cfg)

    # ── Stage 5: Synthetic evolution (escape-aware scorer) ──
    evolved = stage_evolution(cfg, seeds, scorer, mock=mock)
    evolved_scores = scorer(evolved) if evolved else []
    if checkpoint_manager:
        save_stage_5_evolution(checkpoint_manager, evolved, evolved_scores, cfg)

    # ── Stage 6: Repertoire-scale screen ──
    top_seqs, top_scores = stage_repertoire_screen(cfg, evolved, scorer)
    if checkpoint_manager:
        save_stage_6_screening(checkpoint_manager, top_seqs, top_scores, cfg)

    # ── Stage 7: Cross-reactivity against escape panel ──
    coverage_matrix, adaptation = stage_cross_reactivity(
        cfg, top_seqs, escape_panel, lm, mock=mock
    )
    if checkpoint_manager:
        save_stage_7_cross_reactivity(checkpoint_manager, coverage_matrix, adaptation,
                                      top_seqs, escape_panel)

    # ── Stage 8: Vaccine candidate selection ──
    vaccine_seqs = stage_vaccine_design(
        cfg, top_seqs, escape_panel, lm, coverage_matrix
    )
    if checkpoint_manager:
        save_stage_8_vaccine_design(checkpoint_manager, vaccine_seqs, cfg)

    # ── Stage 9: Experimental validation ──
    report_path = stage_validation(cfg, top_seqs, top_scores)
    if checkpoint_manager:
        save_stage_9_validation(checkpoint_manager, top_seqs, top_scores, report_path)

    # Escape coverage report (cross-reactivity heatmap + coverage stats)
    escape_report_path = generate_escape_report(
        antibody_sequences=top_seqs,
        escape_panel=escape_panel,
        coverage_matrix=coverage_matrix,
        adaptation_summary=adaptation,
        output_dir=str(
            __import__("pathlib").Path(cfg.experiment.output_plot).parent
        ),
    )

    # ── Stage 10: Lab-in-the-loop ──
    lab_loop_result = stage_lab_loop(
        cfg, top_seqs, top_scores, finetuner, escape_panel, lm, mock=mock
    )
    if checkpoint_manager:
        save_stage_10_lab_loop(checkpoint_manager, lab_loop_result)

    logger.info("RP1 Pipeline complete.")
    if top_seqs:
        logger.info("Top sequence: %s  score=%.4f", top_seqs[0], top_scores[0])
    logger.info("Vaccine candidates: %d", len(vaccine_seqs))
    logger.info("Validation report: %s", report_path)
    logger.info("Escape report:      %s", escape_report_path)
    if checkpoint_manager:
        logger.info("Checkpoints saved to: %s", checkpoint_manager.run_dir)

    return {
        "sequences": top_seqs,
        "scores": top_scores,
        "latent_embeddings": latent,
        "escape_panel": escape_panel,
        "coverage_matrix": coverage_matrix,
        "adaptation_summary": adaptation,
        "vaccine_candidates": vaccine_seqs,
        "report_path": report_path,
        "escape_report_path": escape_report_path,
        # New RP1 outputs
        "bcr_repertoire": repertoire,
        "bcr_atlas": atlas,
        "affinity_matrix": affinity_matrix,
        "binding_matrix": binding_matrix,
        "alm_finetuner": finetuner,
        "blind_spot_report": blind_spot_report,
        "lab_loop_result": lab_loop_result,
        "pathway_result": pathway_result,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Computational Antibody Discovery Pipeline"
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Run with mock models (no GPU/large downloads required)"
    )
    p.add_argument(
        "--seeds", nargs="+", default=None,
        help="Seed antibody sequences (space-separated)"
    )
    p.add_argument(
        "--seeds-csv", default=None,
        help="Path to CSV file containing seed sequences"
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    p.add_argument(
        "--no-checkpoints", action="store_true",
        help="Disable saving intermediate checkpoints"
    )
    p.add_argument(
        "--checkpoint-dir", default="experiments/checkpoints",
        help="Directory to store checkpoints"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(level=args.log_level)

    seeds = args.seeds
    if args.seeds_csv:
        loaded_seqs, _ = load_sequences(args.seeds_csv)
        seeds = loaded_seqs or seeds

    results = run_pipeline(
        seeds=seeds,
        mock=args.mock,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_dir=args.checkpoint_dir
    )
    print(f"\nDone. Report written to: {results['report_path']}")
