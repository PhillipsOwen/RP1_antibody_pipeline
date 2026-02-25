"""
models/alm_finetuner.py

Fine-tunes the Antibody Language Model (ALM) using MD-predicted binding
scores as training signal.

Rationale
---------
ESM-2 was pre-trained on evolutionary sequence data; its pseudo-log-
likelihoods (PLLs) reflect general protein naturalness.  After MD binding
prediction we have per-sequence binding scores that reflect affinity to a
specific pathogen antigen.  Fine-tuning the LM to align PLLs with those
scores tailors the model to the target antigen without full retraining.

Training objective
------------------
Pairwise margin ranking loss:
    For pairs (i, j) where binding_score[i] > binding_score[j], push
        PLL(seq_i)  >  PLL(seq_j) + margin

This requires only relative ordering from MD, not absolute ΔG values.

Optimisation
------------
Only the feed-forward (FFN) layers are updated; attention weights are
frozen.  This resembles BitFit / adapter tuning and limits overfitting
on small MD datasets.

In mock mode the ALM's score-offset dict is adjusted directly (no gradients),
allowing full pipeline testing without GPU.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ALMFineTuner:
    """
    Fine-tunes an AntibodyLM using MD-derived binding scores.

    Parameters
    ----------
    lm            : AntibodyLM or RandomAntibodyLM instance.
    learning_rate : gradient step size (real mode) or offset step (mock).
    n_epochs      : number of passes over the training set.
    margin        : ranking loss margin.
    mock          : if True, simulate fine-tuning without gradients.
    """

    def __init__(
        self,
        lm,
        learning_rate: float = 1e-5,
        n_epochs: int = 3,
        margin: float = 0.1,
        mock: bool = True,
    ):
        self.lm = lm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.margin = margin
        self.mock = mock
        # Mock mode: per-sequence score adjustments applied after base LM score
        self._score_offsets: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def finetune(
        self,
        antibody_seqs: List[str],
        md_binding_scores: List[float],
    ) -> Dict[str, list]:
        """
        Fine-tune the ALM to align its PLLs with MD-predicted binding scores.

        Parameters
        ----------
        antibody_seqs     : antibody amino acid sequences.
        md_binding_scores : per-sequence binding scores from BindingMDPredictor,
                            values in [0, 1].

        Returns
        -------
        dict with 'loss_history': list of per-epoch mean losses.
        """
        if len(antibody_seqs) != len(md_binding_scores):
            raise ValueError(
                "antibody_seqs and md_binding_scores must be the same length."
            )
        if len(antibody_seqs) < 2:
            logger.warning("Need ≥2 sequences to compute ranking loss — skipping.")
            return {"loss_history": []}

        logger.info(
            "ALM fine-tuning: %d sequences, %d epochs, lr=%.2e, mock=%s",
            len(antibody_seqs), self.n_epochs, self.learning_rate, self.mock,
        )
        if self.mock:
            return self._mock_finetune(antibody_seqs, md_binding_scores)
        return self._real_finetune(antibody_seqs, md_binding_scores)

    def score_with_finetuning(self, sequences: List[str]) -> List[float]:
        """
        Score sequences using the fine-tuned LM.

        Applies mock offsets if present; otherwise delegates to lm.score().
        """
        base = self.lm.score(sequences)
        return [b + self._score_offsets.get(s, 0.0) for b, s in zip(base, sequences)]

    # ── Fine-tuning implementations ────────────────────────────────────────

    def _mock_finetune(
        self,
        antibody_seqs: List[str],
        md_binding_scores: List[float],
    ) -> Dict[str, list]:
        """
        Mock fine-tuning: adjust score offsets to simulate gradient updates.

        Sequences with high MD binding scores receive positive offsets;
        low-scoring sequences receive negative offsets, preserving the
        relative ordering imposed by MD.
        """
        loss_history = []
        for epoch in range(self.n_epochs):
            indices = list(range(len(antibody_seqs)))
            random.shuffle(indices)
            epoch_loss = 0.0
            pairs_seen = 0

            for k in range(0, len(indices) - 1, 2):
                i, j = indices[k], indices[k + 1]
                bi, bj = md_binding_scores[i], md_binding_scores[j]
                oi = self._score_offsets.get(antibody_seqs[i], 0.0)
                oj = self._score_offsets.get(antibody_seqs[j], 0.0)

                if bi > bj:
                    hinge = max(0.0, (oj + self.margin) - oi)
                    if hinge > 0:
                        self._score_offsets[antibody_seqs[i]] = oi + self.learning_rate * hinge
                        self._score_offsets[antibody_seqs[j]] = oj - self.learning_rate * hinge
                    epoch_loss += hinge
                elif bj > bi:
                    hinge = max(0.0, (oi + self.margin) - oj)
                    if hinge > 0:
                        self._score_offsets[antibody_seqs[j]] = oj + self.learning_rate * hinge
                        self._score_offsets[antibody_seqs[i]] = oi - self.learning_rate * hinge
                    epoch_loss += hinge
                pairs_seen += 1

            mean_loss = epoch_loss / max(pairs_seen, 1)
            loss_history.append(mean_loss)
            logger.info("Fine-tune epoch %d/%d  loss=%.4f",
                        epoch + 1, self.n_epochs, mean_loss)

        logger.info("Fine-tuning complete. Offsets updated for %d sequences.",
                    len(self._score_offsets))
        return {"loss_history": loss_history}

    def _real_finetune(
        self,
        antibody_seqs: List[str],
        md_binding_scores: List[float],
    ) -> Dict[str, list]:
        """
        Gradient-based fine-tuning of the LM's FFN layers.

        Freezes all attention weights; updates only dense/FFN parameters
        via pairwise margin ranking loss on pseudo-log-likelihoods.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for real fine-tuning: pip install torch")

        lm_model = self.lm.model
        tokenizer = self.lm.tokenizer

        # Freeze attention; collect FFN parameters
        trainable = []
        for name, param in lm_model.named_parameters():
            if any(k in name for k in ("dense", "fc", "ffn",
                                        "intermediate", "output")):
                param.requires_grad = True
                trainable.append(param)
            else:
                param.requires_grad = False

        if not trainable:
            logger.warning("No FFN parameters found — falling back to mock mode.")
            return self._mock_finetune(antibody_seqs, md_binding_scores)

        optimiser = torch.optim.AdamW(trainable, lr=self.learning_rate)
        loss_fn = nn.MarginRankingLoss(margin=self.margin)
        loss_history = []
        indices = list(range(len(antibody_seqs)))

        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            epoch_loss, steps = 0.0, 0

            for k in range(0, len(indices) - 1, 2):
                i, j = indices[k], indices[k + 1]
                pll_i = self._pll_tensor(antibody_seqs[i], tokenizer, lm_model)
                pll_j = self._pll_tensor(antibody_seqs[j], tokenizer, lm_model)

                target = torch.tensor(
                    [1.0 if md_binding_scores[i] >= md_binding_scores[j] else -1.0]
                )
                loss = loss_fn(pll_i.unsqueeze(0), pll_j.unsqueeze(0), target)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                steps += 1

            mean_loss = epoch_loss / max(steps, 1)
            loss_history.append(mean_loss)
            logger.info("Fine-tune (real) epoch %d/%d  loss=%.4f",
                        epoch + 1, self.n_epochs, mean_loss)

        return {"loss_history": loss_history}

    def _pll_tensor(self, sequence: str, tokenizer, model) -> "torch.Tensor":
        """
        Pseudo-log-likelihood as a differentiable scalar via simultaneous masking.

        All non-special tokens are masked in a single forward pass (approximation).
        More efficient than per-position masking while remaining differentiable.
        """
        import torch
        import torch.nn.functional as F

        enc = tokenizer(sequence, return_tensors="pt",
                        truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(self.lm.device)

        special = {tokenizer.cls_token_id,
                   tokenizer.eos_token_id,
                   tokenizer.pad_token_id}
        masked = input_ids.clone()
        for pos in range(masked.shape[1]):
            if masked[0, pos].item() not in special:
                masked[0, pos] = tokenizer.mask_token_id

        logits = model(input_ids=masked).logits[0]
        true_ids = input_ids[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs[range(len(true_ids)), true_ids]

        non_special = torch.tensor(
            [t.item() not in special for t in true_ids], dtype=torch.bool
        )
        return token_lp[non_special].mean()
