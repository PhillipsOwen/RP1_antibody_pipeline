"""
experiments/lab_loop.py

Laboratory-in-the-loop validation and iterative model refinement.

Purpose
-------
After computational predictions are made, wet-lab experiments are run
(e.g., ELISA binding assays, pseudovirus neutralisation, SPR kinetics).
This module ingests those experimental results and uses them to:

  1. Refine the ALM — re-fine-tune with real binding measurements as labels
     (stronger signal than the MD embedding proxy).
  2. Identify active learning candidates — rank sequences by prediction
     uncertainty to suggest the most informative next round of experiments.
  3. Update the viral escape panel — add experimentally confirmed low/no-
     binding sequences as validated escape mutants.
  4. Produce a per-iteration report comparing predicted vs measured binding.

Experimental data format (CSV)
-------------------------------
Required columns:
    sequence          : antibody amino acid sequence
    measured_binding  : numeric binding score (e.g. log IC50, OD450, KD)

Optional columns:
    sequence_id, assay_type, replicate, date, notes

Iteration loop
--------------
Each call to LabInTheLoop.run_iteration() represents one wet-lab cycle:
    predict → experiment → ingest → refine → suggest next round
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class ExperimentalResult:
    """A single experimental binding measurement."""
    sequence: str
    measured_binding: float
    sequence_id: str = ""
    assay_type: str = ""
    notes: str = ""


# ─── Lab-in-the-loop ──────────────────────────────────────────────────────────

class LabInTheLoop:
    """
    Iterative laboratory-in-the-loop model refinement.

    Parameters
    ----------
    lm          : AntibodyLM or RandomAntibodyLM instance.
    finetuner   : ALMFineTuner instance (will be updated each iteration).
    escape_panel: current list of EscapeMutant objects.
    output_dir  : directory where per-iteration reports are written.
    mock        : if True, simulate experimental data generation.
    """

    def __init__(
        self,
        lm,
        finetuner,
        escape_panel: list,
        output_dir: str = "experiments/output/lab_loop",
        mock: bool = True,
    ):
        self.lm = lm
        self.finetuner = finetuner
        self.escape_panel = list(escape_panel)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self._iteration = 0
        self._history: List[Dict] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def run_iteration(
        self,
        candidate_sequences: List[str],
        predicted_scores: List[float],
        experimental_csv: Optional[str] = None,
        n_suggestions: int = 20,
    ) -> Dict:
        """
        Run one full lab-in-the-loop iteration.

        Parameters
        ----------
        candidate_sequences : pool of antibody sequences from the pipeline.
        predicted_scores    : pipeline-predicted binding scores per sequence.
        experimental_csv    : path to CSV with experimental binding results.
                              If None (or mock=True), synthetic data is used.
        n_suggestions       : number of sequences to suggest for next round.

        Returns
        -------
        dict with keys:
          iteration          : iteration number
          n_experimental     : number of experimental data points ingested
          refinement_loss    : final loss from ALM re-fine-tuning
          correlation        : Pearson r between predicted and measured binding
          suggested_next     : sequences recommended for next experiment
          updated_escape     : number of new escape mutants added to panel
          report_path        : path to saved JSON report
        """
        self._iteration += 1
        logger.info("=== Lab-in-the-loop iteration %d ===", self._iteration)

        # 1. Ingest experimental data
        if experimental_csv and Path(experimental_csv).exists():
            results = self.ingest_experimental_data(experimental_csv)
        elif self.mock:
            results = self._generate_mock_data(candidate_sequences,
                                               predicted_scores)
        else:
            logger.warning("No experimental CSV provided and mock=False. "
                           "Skipping refinement.")
            return {"iteration": self._iteration, "skipped": True}

        exp_seqs = [r.sequence for r in results]
        exp_scores = [r.measured_binding for r in results]

        # 2. Compute correlation before refinement
        corr_before = self._correlation(predicted_scores, exp_seqs, exp_scores)

        # 3. Refine ALM with experimental labels
        refinement = self.refine_model(exp_seqs, exp_scores)

        # 4. Re-score with fine-tuned LM
        refined_scores = self.finetuner.score_with_finetuning(exp_seqs)
        corr_after = self._pearson(exp_scores, refined_scores)

        # 5. Active learning: suggest next experiments
        suggestions = self.suggest_next_experiments(
            candidate_sequences, predicted_scores, n=n_suggestions
        )

        # 6. Update escape panel
        n_new_escape = self._update_escape_panel(exp_seqs, exp_scores)

        # 7. Save report
        report = {
            "iteration": self._iteration,
            "n_experimental": len(results),
            "refinement_loss": refinement.get("loss_history", [float("nan")])[-1]
                               if refinement.get("loss_history") else float("nan"),
            "pearson_r_before": corr_before,
            "pearson_r_after": corr_after,
            "suggested_next": suggestions,
            "updated_escape": n_new_escape,
        }
        report_path = self._save_report(report)
        report["report_path"] = report_path
        self._history.append(report)

        logger.info(
            "Iteration %d complete: n_exp=%d  r_before=%.3f  r_after=%.3f  "
            "new_escape=%d  suggested=%d",
            self._iteration, len(results),
            corr_before, corr_after,
            n_new_escape, len(suggestions),
        )
        return report

    def ingest_experimental_data(
        self,
        csv_path: str,
        sequence_col: str = "sequence",
        score_col: str = "measured_binding",
    ) -> List[ExperimentalResult]:
        """
        Load experimental binding data from a CSV file.

        The CSV must contain at least *sequence_col* and *score_col*.
        Additional columns (sequence_id, assay_type, notes) are optional.
        """
        results: List[ExperimentalResult] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                seq = row.get(sequence_col, "").strip()
                score_str = row.get(score_col, "").strip()
                if not seq or not score_str:
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                results.append(ExperimentalResult(
                    sequence=seq,
                    measured_binding=score,
                    sequence_id=row.get("sequence_id", "").strip(),
                    assay_type=row.get("assay_type", "").strip(),
                    notes=row.get("notes", "").strip(),
                ))
        logger.info("Ingested %d experimental results from %s",
                    len(results), csv_path)
        return results

    def refine_model(
        self,
        sequences: List[str],
        experimental_scores: List[float],
    ) -> Dict:
        """
        Re-fine-tune the ALM using experimental binding measurements.

        Experimental scores are normalised to [0, 1] and used as the
        ranking signal — replacing or augmenting the MD proxy scores from
        the initial fine-tuning round.

        Returns
        -------
        dict with 'loss_history'.
        """
        if not sequences or len(sequences) < 2:
            logger.warning("Fewer than 2 experimental sequences — skipping refinement.")
            return {"loss_history": []}

        # Normalise to [0, 1]
        arr = np.array(experimental_scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            normed = ((arr - lo) / (hi - lo)).tolist()
        else:
            normed = [0.5] * len(arr)

        logger.info(
            "Refining ALM on %d experimental data points (iteration %d) …",
            len(sequences), self._iteration,
        )
        return self.finetuner.finetune(sequences, normed)

    def suggest_next_experiments(
        self,
        candidate_sequences: List[str],
        predicted_scores: List[float],
        n: int = 20,
    ) -> List[str]:
        """
        Active learning: select the most informative sequences to test next.

        Strategy: uncertainty sampling.  Sequences whose predicted binding
        score is closest to the midpoint (most uncertain predictions) are
        most informative to measure.  A diversity term ensures the
        suggestions span different parts of sequence space.

        Scores are from the fine-tuned LM (if available), otherwise the
        raw predicted_scores.

        Returns
        -------
        List of up to *n* sequence strings recommended for the next round.
        """
        if not candidate_sequences:
            return []

        # Get fine-tuned scores where available
        ft_scores = self.finetuner.score_with_finetuning(candidate_sequences)

        # Normalise to [0, 1] for uniform uncertainty calculation
        arr = np.array(ft_scores)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            normed = (arr - lo) / (hi - lo)
        else:
            normed = np.full_like(arr, 0.5)

        # Uncertainty = distance from midpoint 0.5 (inverted)
        uncertainty = 1.0 - np.abs(normed - 0.5) * 2.0    # [0,1]: 1=most uncertain

        # Diversity: avoid near-identical sequences (simple length-based proxy)
        lengths = np.array([len(s) for s in candidate_sequences], dtype=float)
        len_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-9)

        # Combined score: 70% uncertainty + 30% diversity
        combined = 0.7 * uncertainty + 0.3 * len_norm

        # Rank and return top n
        ranked_idx = np.argsort(combined)[::-1]
        suggestions = [candidate_sequences[i] for i in ranked_idx[:n]]
        logger.info("Suggested %d sequences for next experimental round.", len(suggestions))
        return suggestions

    def get_history(self) -> List[Dict]:
        """Return the full iteration history."""
        return list(self._history)

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_mock_data(
        self,
        sequences: List[str],
        predicted_scores: List[float],
    ) -> List[ExperimentalResult]:
        """
        Generate synthetic experimental data correlated with predictions.
        Used when no real experimental CSV is provided and mock=True.
        """
        np.random.seed(self._iteration)
        arr = np.array(predicted_scores)
        # Normalise
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            normed = (arr - lo) / (hi - lo)
        else:
            normed = np.full_like(arr, 0.5)
        # Add noise to simulate real experimental variance
        noisy = np.clip(normed + np.random.randn(len(normed)) * 0.15, 0.0, 1.0)
        logger.info("Generated %d mock experimental data points.", len(sequences))
        return [
            ExperimentalResult(
                sequence=seq,
                measured_binding=float(score),
                assay_type="mock_ELISA",
            )
            for seq, score in zip(sequences, noisy.tolist())
        ]

    def _update_escape_panel(
        self,
        sequences: List[str],
        experimental_scores: List[float],
        escape_threshold: float = 0.3,
    ) -> int:
        """
        Add experimentally confirmed escape sequences to the panel.

        A sequence is considered an escape variant if its measured binding
        score falls below *escape_threshold* (normalised to [0, 1]).
        """
        from .validation import ValidationDataset

        arr = np.array(experimental_scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        normed = (arr - lo) / (hi - lo + 1e-9)

        n_added = 0
        existing_seqs = {m.mutant_sequence for m in self.escape_panel}

        for seq, norm_score in zip(sequences, normed.tolist()):
            if norm_score < escape_threshold and seq not in existing_seqs:
                # Import here to avoid circular dependency at module level
                from ..viral_escape.escape_mutant import EscapeMutant
                escape_mutant = EscapeMutant(
                    wildtype_sequence=seq,
                    mutant_sequence=seq,
                    mutations=[],
                    epitope_mutations=0,
                    escape_score=1.0 - norm_score,  # low binding = high escape
                )
                self.escape_panel.append(escape_mutant)
                existing_seqs.add(seq)
                n_added += 1

        if n_added:
            logger.info("Added %d experimentally confirmed escape variants to panel.",
                        n_added)
        return n_added

    def _correlation(
        self,
        predicted_scores: List[float],
        exp_seqs: List[str],
        exp_scores: List[float],
    ) -> float:
        """Pearson r between predicted (for exp_seqs) and measured scores."""
        pred_for_exp = self.finetuner.score_with_finetuning(exp_seqs)
        return self._pearson(exp_scores, pred_for_exp)

    @staticmethod
    def _pearson(a: List[float], b: List[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        if a_arr.std() == 0 or b_arr.std() == 0:
            return 0.0
        return float(np.corrcoef(a_arr, b_arr)[0, 1])

    def _save_report(self, report: Dict) -> str:
        path = self.output_dir / f"lab_loop_iter_{self._iteration:03d}.json"
        serialisable = {}
        for k, v in report.items():
            if isinstance(v, (list, str, int, float, bool)) or v is None:
                serialisable[k] = v
            elif hasattr(v, "tolist"):
                serialisable[k] = v.tolist()
            else:
                serialisable[k] = str(v)
        with open(path, "w") as fh:
            json.dump(serialisable, fh, indent=2)
        return str(path)
