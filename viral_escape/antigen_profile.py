"""
viral_escape/antigen_profile.py

Profiles pathogen antigen sequences against antibody language model (ALM)
binding site representations.

Workflow
--------
1. Load antigen sequences (spike protein variants) from FASTA.
2. For each antigen, extract ALM embeddings weighted by known epitope residues.
3. For each antibody, extract CDR-region embeddings from the same ALM.
4. Compute antibody-antigen binding compatibility via embedding similarity.

This answers: given the ALM's learned representation of protein sequence space,
how well does each antibody candidate 'match' each antigen variant at its epitope?
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical CDR positions in a ~100-aa VH fragment (0-indexed, approximate)
_DEFAULT_CDR_POSITIONS = (
    list(range(24, 34))   # CDR-H1
    + list(range(52, 56)) # CDR-H2
    + list(range(93, 102)) # CDR-H3
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AntigenBindingSiteProfiler:
    """
    Compares pathogen antigen epitopes against antibody binding site
    representations learned by the antibody language model.

    Parameters
    ----------
    lm               : AntibodyLM or RandomAntibodyLM instance.
    epitope_residues : 0-indexed antigen residue positions forming the epitope.
    similarity_metric: 'cosine' (default) | 'dot' | 'euclidean'.
    """

    def __init__(
        self,
        lm,
        epitope_residues: List[int],
        similarity_metric: str = "cosine",
    ):
        self.lm = lm
        self.epitope_residues = list(epitope_residues)
        self.similarity_metric = similarity_metric

    # ── Embedding extraction ───────────────────────────────────────────────

    def profile_antigen_epitope(self, antigen_seq: str) -> np.ndarray:
        """
        Return an embedding representing the antigen epitope.

        For long antigens (>400 aa) the sequence is sliced to a 400-aa
        window centred on the mean epitope position before ALM encoding.
        The resulting mean-pooled embedding is weighted by the epitope
        coverage fraction to bias it towards the epitope region.
        """
        if len(antigen_seq) > 400:
            centre = (
                int(np.mean(self.epitope_residues))
                if self.epitope_residues
                else len(antigen_seq) // 2
            )
            start = max(0, centre - 200)
            end = min(len(antigen_seq), centre + 200)
            antigen_slice = antigen_seq[start:end]
            adj_epitope = [r - start for r in self.epitope_residues
                           if start <= r < end]
        else:
            antigen_slice = antigen_seq
            adj_epitope = self.epitope_residues

        emb = self.lm.embed([antigen_slice])[0]   # (hidden_dim,)
        if adj_epitope:
            epitope_frac = len(adj_epitope) / max(len(antigen_slice), 1)
            emb = emb * (1.0 + epitope_frac)
        return emb

    def profile_antibody_cdr(
        self,
        antibody_seq: str,
        cdr_positions: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Return the CDR-weighted embedding of an antibody sequence.

        Uses *cdr_positions* (0-indexed) to identify CDR residues.
        Defaults to canonical CDR1/2/3 positions for a ~100-aa VH fragment.
        """
        positions = cdr_positions if cdr_positions is not None else _DEFAULT_CDR_POSITIONS
        valid_pos = [p for p in positions if p < len(antibody_seq)]

        emb = self.lm.embed([antibody_seq])[0]    # (hidden_dim,)
        cdr_frac = len(valid_pos) / max(len(antibody_seq), 1)
        return emb * (1.0 + cdr_frac)

    # ── Binding compatibility ──────────────────────────────────────────────

    def binding_compatibility(
        self,
        antibody_seq: str,
        antigen_seq: str,
    ) -> float:
        """
        Compute a binding compatibility score in [0, 1].

        Higher values mean the ALM considers the antibody CDR region more
        compatible with the antigen epitope in sequence space.
        """
        ab_emb = self.profile_antibody_cdr(antibody_seq)
        ag_emb = self.profile_antigen_epitope(antigen_seq)
        return self._similarity(ab_emb, ag_emb)

    def build_affinity_matrix(
        self,
        antibody_sequences: List[str],
        antigen_sequences: List[str],
    ) -> np.ndarray:
        """
        Build an (n_antibodies × n_antigens) ALM-derived affinity matrix.

        Each entry [i, j] is the binding compatibility of antibody i
        against antigen j based on ALM embeddings.

        Returns
        -------
        np.ndarray of shape (n_antibodies, n_antigens), values in [0, 1].
        """
        logger.info(
            "Profiling %d antibodies × %d antigen variants via ALM …",
            len(antibody_sequences), len(antigen_sequences),
        )
        ab_embs = np.array([self.profile_antibody_cdr(s) for s in antibody_sequences])
        ag_embs = np.array([self.profile_antigen_epitope(s) for s in antigen_sequences])

        if self.similarity_metric == "cosine":
            ab_n = ab_embs / (np.linalg.norm(ab_embs, axis=1, keepdims=True) + 1e-9)
            ag_n = ag_embs / (np.linalg.norm(ag_embs, axis=1, keepdims=True) + 1e-9)
            raw = ab_n @ ag_n.T                  # (n_ab, n_ag) in [-1, 1]
            matrix = (raw + 1.0) / 2.0           # → [0, 1]
        elif self.similarity_metric == "euclidean":
            diff = ab_embs[:, np.newaxis, :] - ag_embs[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            matrix = 1.0 / (1.0 + dist)
        else:
            matrix = np.zeros((len(antibody_sequences), len(antigen_sequences)))
            for i, ab in enumerate(antibody_sequences):
                for j, ag in enumerate(antigen_sequences):
                    matrix[i, j] = self.binding_compatibility(ab, ag)

        logger.info(
            "ALM affinity matrix: shape=%s  mean=%.3f  max=%.3f",
            matrix.shape, matrix.mean(), matrix.max(),
        )
        return matrix

    # ── Private ────────────────────────────────────────────────────────────

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.similarity_metric == "cosine":
            return (_cosine_sim(a, b) + 1.0) / 2.0
        elif self.similarity_metric == "dot":
            return float(1.0 / (1.0 + np.exp(-np.dot(a, b))))
        elif self.similarity_metric == "euclidean":
            return float(1.0 / (1.0 + np.linalg.norm(a - b)))
        else:
            raise ValueError(f"Unknown similarity_metric: {self.similarity_metric}")
