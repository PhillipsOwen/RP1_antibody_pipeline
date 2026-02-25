"""
viral_escape/blind_spot.py

Predicts whether typical human antibody repertoires will generate effective
antibodies against a new pathogen antigen, and identifies immune "blind spots"
— epitope regions that are poorly represented in the BCR repertoire.

Algorithm
---------
The BCR atlas (built from OAS + private repertoire data) encodes the
distribution of antibody sequence space that a population's immune system
can access.  For a given antigen, we ask: how far is this antigen's epitope
from anything the repertoire has 'seen'?

1.  Embed the antigen epitope using the ALM (via AntigenBindingSiteProfiler).
2.  Compute max cosine similarity between the epitope embedding and every
    antibody in the atlas (the *coverage score*).
        coverage_score ≈ 0  →  antigen looks familiar to the repertoire.
        coverage_score ≈ 1  →  antigen lies in a genuine immune blind spot.
3.  For per-position analysis, substitute each epitope residue with all 20
    amino acids and score coverage for each substitution.  Positions where
    *all* substitutions remain blind spots are flagged as "hard blind spots".
4.  Report a blind_spot_fraction: fraction of antigen variants (e.g. escape
    mutants) that fall outside repertoire coverage.

Outputs
-------
A blind_spot_report dict (and optional JSON) containing:
  blind_spot_scores          : per-antigen float in [0, 1]
  mean_blind_spot_score      : average across antigen panel
  per_position_coverage      : [(position, mean_coverage_score), ...] ascending
  blind_spot_positions       : epitope positions with mean_coverage < threshold
  hard_blind_spot_positions  : positions where ALL 20 substitutions are blind
  blind_spot_fraction        : fraction of antigens above blind_spot_threshold
  repertoire_at_risk         : bool — True if mean_blind_spot_score > 0.5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .antigen_profile import AntigenBindingSiteProfiler

logger = logging.getLogger(__name__)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

# Fraction of epitope positions that must be blind spots to flag whole antigen
_DEFAULT_BLIND_SPOT_THRESHOLD = 0.5


class BlindSpotAnalyzer:
    """
    Identifies immune blind spots in a pathogen antigen relative to a BCR
    repertoire atlas.

    Parameters
    ----------
    lm               : AntibodyLM or RandomAntibodyLM instance.
    atlas            : disease-specific atlas dict from lm.build_atlas().
    epitope_residues : 0-indexed antigen residue positions forming the epitope.
    blind_spot_threshold : coverage score below which a position is a blind spot.
                           Default 0.5 (on cosine similarity scale [0,1]).
    """

    def __init__(
        self,
        lm,
        atlas: dict,
        epitope_residues: List[int],
        blind_spot_threshold: float = _DEFAULT_BLIND_SPOT_THRESHOLD,
    ):
        self.lm = lm
        self.atlas = atlas
        self.epitope_residues = list(epitope_residues)
        self.blind_spot_threshold = blind_spot_threshold

        self._profiler = AntigenBindingSiteProfiler(
            lm=lm,
            epitope_residues=epitope_residues,
            similarity_metric="cosine",
        )

        # Pre-normalise atlas embeddings for fast cosine lookups
        embs = atlas.get("embeddings", np.zeros((1, 64)))
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self._atlas_normed: np.ndarray = embs / norms    # (N_atlas, D)

    # ── Core scoring ───────────────────────────────────────────────────────

    def coverage_score(self, antigen_seq: str) -> float:
        """
        Maximum cosine similarity between the antigen epitope embedding and
        any antibody in the BCR atlas, scaled to [0, 1].

        High score → epitope is well covered by the repertoire.
        Low score  → epitope lies in a blind spot.
        """
        ag_emb = self._profiler.profile_antigen_epitope(antigen_seq)
        ag_norm_vec = ag_emb / (np.linalg.norm(ag_emb) + 1e-9)
        sims = self._atlas_normed @ ag_norm_vec           # (N_atlas,)
        raw_max = float(sims.max())                       # in [-1, 1]
        return (raw_max + 1.0) / 2.0                      # → [0, 1]

    def blind_spot_score(self, antigen_seq: str) -> float:
        """
        Blind spot score = 1 − coverage_score.

        0 → antigen is well covered by the repertoire.
        1 → antigen is a genuine immune blind spot.
        """
        return 1.0 - self.coverage_score(antigen_seq)

    # ── Per-position analysis ──────────────────────────────────────────────

    def per_position_coverage(
        self,
        antigen_seq: str,
    ) -> List[Tuple[int, float]]:
        """
        For each epitope residue position, compute mean coverage across all
        single-amino-acid substitutions at that position.

        Positions with low mean coverage are "soft blind spots" — the
        repertoire is unlikely to cover mutations here.

        Returns
        -------
        List of (position, mean_coverage) sorted ascending by coverage
        (lowest coverage / biggest blind spot first).
        """
        results: List[Tuple[int, float]] = []
        for pos in self.epitope_residues:
            if pos >= len(antigen_seq):
                continue
            wt_aa = antigen_seq[pos]
            coverages = []
            for mut_aa in AA_ALPHABET:
                if mut_aa == wt_aa:
                    continue
                mutant = antigen_seq[:pos] + mut_aa + antigen_seq[pos + 1:]
                coverages.append(self.coverage_score(mutant))
            mean_cov = float(np.mean(coverages)) if coverages else 0.0
            results.append((pos, mean_cov))

        results.sort(key=lambda x: x[1])
        return results

    def hard_blind_spot_positions(self, antigen_seq: str) -> List[int]:
        """
        Return epitope positions where *all* 20 amino acid substitutions
        remain below the blind_spot_threshold.

        These positions are "hard blind spots" — no matter how the virus
        mutates here, typical repertoires are unlikely to respond.
        """
        hard: List[int] = []
        for pos in self.epitope_residues:
            if pos >= len(antigen_seq):
                continue
            wt_aa = antigen_seq[pos]
            all_blind = all(
                self.coverage_score(antigen_seq[:pos] + mut_aa + antigen_seq[pos + 1:])
                < self.blind_spot_threshold
                for mut_aa in AA_ALPHABET
                if mut_aa != wt_aa
            )
            if all_blind:
                hard.append(pos)
        return hard

    # ── Panel-level analysis ───────────────────────────────────────────────

    def analyze(
        self,
        antigen_seqs: List[str],
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Full blind spot analysis for a panel of antigen variants.

        Parameters
        ----------
        antigen_seqs : list of antigen amino acid sequences to evaluate.
        output_path  : if provided, save the report as JSON.

        Returns
        -------
        dict with keys:
          blind_spot_scores          : per-antigen blind spot score [0, 1]
          mean_blind_spot_score      : mean across panel
          per_position_coverage      : [(pos, mean_cov), ...] ascending
          blind_spot_positions       : positions with mean_cov < threshold
          hard_blind_spot_positions  : positions where ALL substitutions blind
          blind_spot_fraction        : fraction of antigens that are blind spots
          repertoire_at_risk         : True if mean blind spot score > 0.5
          n_antigens                 : number of antigens evaluated
        """
        logger.info(
            "Blind spot analysis: %d antigen variants, %d epitope positions",
            len(antigen_seqs), len(self.epitope_residues),
        )

        scores = [self.blind_spot_score(seq) for seq in antigen_seqs]

        # Per-position analysis on the first (wildtype) antigen
        per_pos = []
        hard_blind = []
        if antigen_seqs:
            per_pos = self.per_position_coverage(antigen_seqs[0])
            hard_blind = self.hard_blind_spot_positions(antigen_seqs[0])

        blind_pos = [pos for pos, cov in per_pos
                     if cov < self.blind_spot_threshold]
        blind_frac = float(
            sum(1 for s in scores if s > self.blind_spot_threshold)
            / max(len(scores), 1)
        )
        mean_score = float(np.mean(scores)) if scores else 0.0

        report = {
            "blind_spot_scores": scores,
            "mean_blind_spot_score": mean_score,
            "per_position_coverage": per_pos,
            "blind_spot_positions": blind_pos,
            "hard_blind_spot_positions": hard_blind,
            "blind_spot_fraction": blind_frac,
            "repertoire_at_risk": mean_score > 0.5,
            "n_antigens": len(antigen_seqs),
        }

        logger.info(
            "Blind spot report: mean_score=%.3f  blind_positions=%s  "
            "hard_blind=%s  at_risk=%s",
            mean_score,
            blind_pos[:5],
            hard_blind[:5],
            report["repertoire_at_risk"],
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Convert numpy types for JSON serialisation
            serialisable = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in report.items()
            }
            with open(output_path, "w") as fh:
                json.dump(serialisable, fh, indent=2)
            logger.info("Blind spot report saved to %s", output_path)

        return report

    def summary_string(self, report: Dict) -> str:
        """Return a human-readable one-paragraph summary of the blind spot report."""
        at_risk = report.get("repertoire_at_risk", False)
        mean_s = report.get("mean_blind_spot_score", 0.0)
        blind_pos = report.get("blind_spot_positions", [])
        hard_pos = report.get("hard_blind_spot_positions", [])
        frac = report.get("blind_spot_fraction", 0.0)

        risk_str = "AT RISK" if at_risk else "adequately covered"
        hard_str = (
            f"Hard blind spots (all substitutions uncovered): {hard_pos}."
            if hard_pos else
            "No hard blind spots detected."
        )
        return (
            f"Repertoire coverage assessment: {risk_str} "
            f"(mean blind-spot score = {mean_s:.3f}, "
            f"{100*frac:.0f}% of antigen variants are blind spots). "
            f"Lowest-coverage epitope positions: {blind_pos[:5]}. "
            f"{hard_str}"
        )
