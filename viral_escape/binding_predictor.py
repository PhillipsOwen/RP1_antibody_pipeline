"""
viral_escape/binding_predictor.py

Cross-reactivity and binding affinity prediction for antibody–escape-mutant pairs.

RP1 goal: for each candidate antibody, predict whether it can still bind viral
escape mutants — and identify which candidates are broadly neutralising enough
to be useful vaccine antigens.

Binding model (no 3-D structure required)
------------------------------------------
For antibody Ab and escape mutant M:

    binding_score(Ab, M) = sqrt(fitness(Ab) × epitope_similarity(M))

Where:
  fitness(Ab)              = LM pseudo-log-likelihood normalised to [0, 1]
                             (how 'natural' the antibody sequence is)
  epitope_similarity(M)    = fraction of antibody-contacting epitope positions
                             unchanged between WT and M
                             (1.0 → WT-like antigen; 0.0 → fully mutated)

The geometric mean rewards antibodies that are both high-fitness AND face
little epitope change.  ΔΔG is approximated as fitness × (similarity − 1),
which is negative when the epitope is disrupted (escape).

Outputs
-------
- Per-pair binding probabilities
- (n_antibodies × n_variants) coverage matrix
- Breadth score per antibody (fraction of panel covered above threshold)
- Vaccine candidate list (broad coverage + high mean binding)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .escape_mutant import EscapeMutant

logger = logging.getLogger(__name__)


# ─── Score normalisation ──────────────────────────────────────────────────────

# LM pseudo-log-likelihoods typically fall in roughly [-30, -5] for 100-residue
# sequences.  These bounds let us map scores to [0, 1] without a running min/max.
_LM_TYPICAL_MIN: float = -30.0
_LM_TYPICAL_MAX: float = -5.0


def _normalise_lm_score(score: float) -> float:
    """Map LM pseudo-log-likelihood → [0, 1]. Values outside range are clipped."""
    n = (score - _LM_TYPICAL_MIN) / (_LM_TYPICAL_MAX - _LM_TYPICAL_MIN)
    return float(np.clip(n, 0.0, 1.0))


# ─── Cross-reactivity scorer ──────────────────────────────────────────────────

class CrossReactivityScorer:
    """
    Predicts how well each antibody binds each viral escape mutant.

    Parameters
    ----------
    lm               : AntibodyLM or RandomAntibodyLM
        Language model used to compute antibody fitness scores.
    epitope_residues : List[int]
        0-indexed positions of antibody-contacting residues on the antigen.
        If empty, overall sequence identity is used as the similarity proxy.
    binding_threshold : float
        Minimum binding_score to consider an antibody as 'covering' a variant.
    """

    def __init__(
        self,
        lm,
        epitope_residues: Optional[List[int]] = None,
        binding_threshold: float = 0.5,
    ):
        self.lm = lm
        self.epitope = list(epitope_residues) if epitope_residues else []
        self.threshold = binding_threshold

    # ── Single-pair methods ───────────────────────────────────────────────────

    def epitope_similarity(self, mutant: EscapeMutant) -> float:
        """
        Fraction of epitope positions unchanged between wildtype and mutant.

        Returns 1.0 if no epitope residues changed (antibody will likely still
        bind), 0.0 if all changed (full escape).
        """
        wt, mut = mutant.wildtype_sequence, mutant.mutant_sequence
        if not self.epitope:
            # No epitope defined: fall back to overall sequence identity
            n = min(len(wt), len(mut))
            return sum(wt[i] == mut[i] for i in range(n)) / max(n, 1)

        unchanged = sum(
            1 for pos in self.epitope
            if pos < len(wt) and pos < len(mut) and wt[pos] == mut[pos]
        )
        return unchanged / len(self.epitope)

    def predict_binding(
        self,
        antibody_seq: str,
        mutant: EscapeMutant,
        antibody_fitness: Optional[float] = None,
    ) -> float:
        """
        Predict binding probability for one antibody–mutant pair in [0, 1].

        Parameters
        ----------
        antibody_seq     : single-letter AA sequence of the antibody
        mutant           : EscapeMutant to evaluate binding against
        antibody_fitness : pre-computed normalised LM score (speeds up batch calls)

        Returns
        -------
        float — binding probability (higher = stronger predicted binding)
        """
        if antibody_fitness is None:
            raw = self.lm.score([antibody_seq])[0]
            antibody_fitness = _normalise_lm_score(raw)

        sim = self.epitope_similarity(mutant)
        return float(np.sqrt(max(antibody_fitness, 0.0) * sim))

    def delta_affinity(self, antibody_seq: str, mutant: EscapeMutant) -> float:
        """
        Approximate ΔΔG (relative affinity change) when the antigen mutates.

        Convention: negative → binding lost (immune escape); 0 → no change.

        ΔΔG proxy = fitness × (epitope_similarity − 1)
        """
        raw = self.lm.score([antibody_seq])[0]
        fitness = _normalise_lm_score(raw)
        sim = self.epitope_similarity(mutant)
        return float(fitness * (sim - 1.0))

    # ── Panel / batch methods ──────────────────────────────────────────────────

    def score_panel(
        self,
        antibody_seq: str,
        escape_panel: List[EscapeMutant],
    ) -> np.ndarray:
        """
        Score one antibody against every escape mutant in *escape_panel*.

        Returns
        -------
        np.ndarray of shape (n_variants,) — binding probabilities in [0, 1]
        """
        raw = self.lm.score([antibody_seq])[0]
        fitness = _normalise_lm_score(raw)
        return np.array(
            [self.predict_binding(antibody_seq, m, fitness) for m in escape_panel],
            dtype=float,
        )

    def coverage_fraction(
        self,
        antibody_seq: str,
        escape_panel: List[EscapeMutant],
    ) -> float:
        """
        Fraction of escape panel variants the antibody can still bind
        (binding_score ≥ self.threshold).

        1.0 → broadly neutralising; 0.0 → no cross-reactivity.
        """
        scores = self.score_panel(antibody_seq, escape_panel)
        return float(np.mean(scores >= self.threshold))

    def build_coverage_matrix(
        self,
        antibody_sequences: List[str],
        escape_panel: List[EscapeMutant],
    ) -> np.ndarray:
        """
        Build the (n_antibodies × n_variants) binding probability matrix.

        Rows    = antibody sequences
        Columns = escape mutants (ordered as in escape_panel)
        Values  = binding probabilities in [0, 1]

        Implementation note: epitope_similarity depends only on the antigen
        side (pre-computed once), while fitness depends only on the antibody
        side — so the full matrix is the outer product sqrt(F ⊗ S), which
        avoids O(n×m) LM calls.
        """
        logger.info(
            "Building coverage matrix: %d antibodies × %d variants …",
            len(antibody_sequences), len(escape_panel),
        )
        # Antibody fitness vector (one LM call per antibody sequence)
        raw_scores = self.lm.score(antibody_sequences)
        fitnesses = np.array([_normalise_lm_score(s) for s in raw_scores])

        # Epitope similarity vector (one call per escape variant)
        sims = np.array([self.epitope_similarity(m) for m in escape_panel])

        # Outer product: matrix[i, j] = sqrt(fitness[i] * similarity[j])
        matrix = np.sqrt(np.clip(np.outer(fitnesses, sims), 0.0, None))
        matrix = np.clip(matrix, 0.0, 1.0)

        logger.info(
            "Coverage matrix: shape=%s  mean=%.3f  above_threshold=%.1f%%",
            matrix.shape,
            matrix.mean(),
            100.0 * (matrix >= self.threshold).mean(),
        )
        return matrix

    # ── Vaccine candidate selection ────────────────────────────────────────────

    def vaccine_candidates(
        self,
        antibody_sequences: List[str],
        escape_panel: List[EscapeMutant],
        coverage_matrix: Optional[np.ndarray] = None,
        min_coverage: float = 0.60,
        top_n: int = 20,
    ) -> List[Tuple[str, float, float]]:
        """
        Select broadly neutralising vaccine candidates.

        Ranking criterion: mean binding score across the full escape panel.
        Filter: must cover ≥ min_coverage fraction of variants.

        Parameters
        ----------
        antibody_sequences : candidate antibody sequences
        escape_panel       : list of escape mutants to evaluate against
        coverage_matrix    : pre-computed matrix (computed if None)
        min_coverage       : minimum fraction of panel that must be 'covered'
        top_n              : maximum candidates to return

        Returns
        -------
        List of (sequence, coverage_fraction, mean_binding_score) tuples,
        sorted by mean_binding_score descending.
        """
        if coverage_matrix is None:
            coverage_matrix = self.build_coverage_matrix(
                antibody_sequences, escape_panel
            )

        coverage_fractions = (coverage_matrix >= self.threshold).mean(axis=1)
        mean_scores = coverage_matrix.mean(axis=1)

        candidates = [
            (seq, float(cov), float(mean))
            for seq, cov, mean in zip(
                antibody_sequences, coverage_fractions, mean_scores
            )
            if cov >= min_coverage
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            "Vaccine design: %d / %d candidates meet coverage ≥ %.0f%%",
            len(candidates), len(antibody_sequences), min_coverage * 100,
        )
        return candidates[:top_n]

    # ── Mechanism-stratified analysis (SA3b) ────────────────────────────────────────

    def stratify_by_mechanism(
        self,
        antibody_sequences: "List[str]",
        escape_panel: "List[EscapeMutant]",
        mechanism_categories: "Dict[str, List[int]]",
        coverage_matrix: "Optional[np.ndarray]" = None,
    ) -> dict:
        """
        Stratify coverage analysis by antibody binding mechanism category (SA3b).

        SA3b categorises antibody neutralization of HIV Env by binding mechanism
        (e.g., V1/V2 vs V3 binders).  This method partitions the escape panel by
        epitope-region label and computes per-mechanism coverage fractions,
        enabling a per-individual B-cell fingerprint that relates antibody category
        composition to protection against variants.

        Parameters
        ----------
        antibody_sequences    : candidate antibody AA sequences.
        escape_panel          : list of EscapeMutant objects.
        mechanism_categories  : dict mapping category name → list of escape panel
                                indices belonging to that category.
                                Example for HIV Env::

                                    {
                                        'V1V2': [0, 1, 2, 5],
                                        'V3':   [3, 4, 6, 7],
                                        'gp41': [8, 9],
                                    }

                                For SARS-CoV-2, use RBD epitope sub-regions
                                (e.g., class I/II/III/IV by Barnes et al.).
        coverage_matrix       : pre-computed (n_ab x n_variants) matrix.
                                If None, computed internally.

        Returns
        -------
        dict with keys:
          'overall'          : overall coverage fraction (all variants, all abs)
          'per_mechanism'    : dict mapping category → {
                                  'mean_coverage': float,
                                  'fraction_broadly_neutralising': float,
                                  'per_antibody_coverage': List[float],
                              }
          'dominant_category': name of the mechanism category with highest mean
                               per-antibody coverage (the 'dominant' fingerprint).
        """
        if coverage_matrix is None:
            coverage_matrix = self.build_coverage_matrix(antibody_sequences, escape_panel)

        overall_coverage = float((coverage_matrix >= self.threshold).mean())

        per_mechanism: dict = {}
        for cat_name, variant_indices in mechanism_categories.items():
            if not variant_indices:
                continue
            valid_idx = [i for i in variant_indices if i < coverage_matrix.shape[1]]
            if not valid_idx:
                continue
            sub_matrix = coverage_matrix[:, valid_idx]  # (n_ab, n_cat_variants)
            per_ab_cov = (sub_matrix >= self.threshold).mean(axis=1)  # (n_ab,)
            per_mechanism[cat_name] = {
                "mean_coverage": float(per_ab_cov.mean()),
                "fraction_broadly_neutralising": float((per_ab_cov >= 0.5).mean()),
                "per_antibody_coverage": per_ab_cov.tolist(),
            }

        dominant = (
            max(per_mechanism, key=lambda k: per_mechanism[k]["mean_coverage"])
            if per_mechanism else None
        )

        logger.info(
            "Mechanism stratification: %d categories; dominant=%s overall_cov=%.3f",
            len(per_mechanism), dominant, overall_coverage,
        )
        return {
            "overall": overall_coverage,
            "per_mechanism": per_mechanism,
            "dominant_category": dominant,
        }

    # ── Immune adaptation prediction ──────────────────────────────────────────

    def predict_immune_adaptation(
        self,
        antibody_sequences: List[str],
        escape_panel: List[EscapeMutant],
        coverage_matrix: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Summarise immune system adaptation potential against the escape panel.

        Answers the RP1 question: *how well might the immune system adapt (or
        struggle) when viruses mutate?*

        Returns
        -------
        dict with keys:
          fraction_broadly_neutralising : fraction of antibodies covering ≥50% panel
          mean_panel_coverage           : average coverage across all antibodies
          most_vulnerable_variants      : escape mutant indices with lowest mean coverage
          most_resistant_antibodies     : indices of broadly neutralising antibodies
        """
        if coverage_matrix is None:
            coverage_matrix = self.build_coverage_matrix(
                antibody_sequences, escape_panel
            )

        ab_coverage = (coverage_matrix >= self.threshold).mean(axis=1)
        variant_coverage = coverage_matrix.mean(axis=0)

        # Variants where mean antibody binding is lowest → most likely to escape
        vulnerable_idx = np.argsort(variant_coverage)[:5].tolist()
        broadly_idx = np.where(ab_coverage >= 0.5)[0].tolist()

        return {
            "fraction_broadly_neutralising": float(np.mean(ab_coverage >= 0.5)),
            "mean_panel_coverage": float(ab_coverage.mean()),
            "most_vulnerable_variants": vulnerable_idx,
            "most_resistant_antibody_indices": broadly_idx[:10],
            "per_antibody_coverage": ab_coverage.tolist(),
            "per_variant_mean_binding": variant_coverage.tolist(),
        }