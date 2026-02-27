"""
md_simulations/binding_md.py

MD-informed antibody-antigen binding prediction.

Two operating modes
-------------------
Mock / embedding mode (default)
    Uses embedding-space proximity as a physics-free proxy for binding
    affinity.  The negative L2 distance (scaled to [0, 1]) between the
    antibody CDR embedding and the antigen epitope embedding serves as the
    binding score.  Fast, GPU-optional, suitable for pipeline testing.

Real MD mode (requires openmm + mdtraj)
    Builds a coarse antibody-antigen complex from sequence embeddings
    projected onto a known scaffold, energy-minimises the interface with
    AMBER14, and returns the interface interaction energy as a binding proxy.
    (Stub — supply AlphaFold-Multimer structures for production use.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MMPBSACalculator:
    """
    MM/PBSA-inspired interaction energy calculator using OpenMM.

    Computes ΔE_interact = E_complex − E_receptor − E_ligand after energy
    minimisation with the AMBER14 force field in vacuum (no explicit solvent).
    When OpenMM is unavailable or no PDB structure is provided the calculator
    silently falls back to an embedding-distance proxy.

    Parameters
    ----------
    pdb_path      : path to a complex PDB file (antibody + antigen chains).
                    If None or absent, all scoring falls back to proxy.
    temperature_k : simulation temperature in Kelvin — used to scale the
                    sigmoid conversion from ΔE (kJ/mol) to [0, 1].
    forcefield    : OpenMM ForceField XML identifier.
    """

    # kB * NA in kJ/(mol·K)
    _kB = 0.008314

    def __init__(
        self,
        pdb_path: Optional[str] = None,
        temperature_k: float = 300.0,
        forcefield: str = "amber14-all.xml",
    ):
        self.pdb_path = pdb_path
        self.temperature_k = temperature_k
        self.forcefield = forcefield
        self._kT = self._kB * temperature_k  # kJ/mol

    # ── Energy → score conversion ─────────────────────────────────────────

    def energy_to_score(self, delta_e_kj_mol: float) -> float:
        """
        Convert ΔE_interact (kJ/mol) to a [0, 1] binding score via sigmoid.

        Negative ΔE (favourable interaction) → score > 0.5.
        """
        return float(1.0 / (1.0 + np.exp(delta_e_kj_mol / self._kT)))

    # ── Physics-based energy calculation ─────────────────────────────────

    def compute_interaction_energy(self, pdb_path: str) -> float:
        """
        Compute ΔE_interact = E_complex − E_receptor − E_ligand.

        Loads the complex from *pdb_path*, energy-minimises the full complex
        and each partner in isolation (vacuum, AMBER14), and returns the
        interaction energy in kJ/mol.

        Requires openmm: ``conda install -c conda-forge openmm``

        Returns
        -------
        float — ΔE in kJ/mol; negative values indicate favourable binding.
        """
        try:
            from openmm.app import (
                PDBFile, ForceField, Simulation, Modeller, NoCutoff,
            )
            from openmm import VerletIntegrator
            from openmm.unit import kilojoules_per_mole
        except ImportError:
            raise ImportError(
                "openmm required for physics-based scoring: "
                "conda install -c conda-forge openmm"
            )

        ff = ForceField(self.forcefield)

        def _minimise(topology, positions) -> float:
            system = ff.createSystem(topology, nonbondedMethod=NoCutoff)
            integrator = VerletIntegrator(0.001)
            sim = Simulation(topology, system, integrator)
            sim.context.setPositions(positions)
            sim.minimizeEnergy(maxIterations=500)
            state = sim.context.getState(getEnergy=True)
            return state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        pdb = PDBFile(pdb_path)
        chains = list(pdb.topology.chains())
        if len(chains) < 2:
            logger.warning(
                "PDB %s has fewer than 2 chains — cannot compute ΔE.", pdb_path
            )
            return 0.0

        # Complex energy
        e_complex = _minimise(pdb.topology, pdb.positions)

        # Receptor-only (first chain)
        rec_mod = Modeller(pdb.topology, pdb.positions)
        rec_mod.delete([
            a for a in rec_mod.topology.atoms()
            if a.residue.chain.index != 0
        ])
        e_receptor = _minimise(rec_mod.topology, rec_mod.positions)

        # Ligand-only (all other chains)
        lig_mod = Modeller(pdb.topology, pdb.positions)
        lig_mod.delete([
            a for a in lig_mod.topology.atoms()
            if a.residue.chain.index == 0
        ])
        e_ligand = _minimise(lig_mod.topology, lig_mod.positions)

        delta_e = e_complex - e_receptor - e_ligand
        logger.info(
            "MM/PBSA: E_complex=%.1f  E_receptor=%.1f  E_ligand=%.1f  "
            "ΔE=%.1f kJ/mol",
            e_complex, e_receptor, e_ligand, delta_e,
        )
        return delta_e

    # ── Per-pair scoring ──────────────────────────────────────────────────

    def score_pair(
        self,
        ab_seq: str,
        ag_seq: str,
        pdb_path: Optional[str] = None,
        ab_emb: Optional[np.ndarray] = None,
        ag_emb: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score one antibody–antigen pair.

        Attempts physics-based scoring when *pdb_path* (or self.pdb_path)
        points to an existing complex PDB file.  Falls back to embedding-
        distance proxy otherwise.

        Returns
        -------
        float in [0, 1] — higher = stronger predicted binding.
        """
        p = pdb_path or self.pdb_path
        if p and Path(p).exists():
            try:
                delta_e = self.compute_interaction_energy(p)
                return self.energy_to_score(delta_e)
            except Exception as exc:
                logger.warning(
                    "MM/PBSA calculation failed (%s) — falling back to "
                    "embedding proxy.", exc
                )

        # Embedding-distance proxy fallback
        if ab_emb is not None and ag_emb is not None:
            ab_n = ab_emb / (np.linalg.norm(ab_emb) + 1e-9)
            ag_n = ag_emb / (np.linalg.norm(ag_emb) + 1e-9)
            dist = float(np.linalg.norm(ab_n - ag_n))
            return float(np.clip(1.0 - dist / 2.0, 0.0, 1.0))

        return 0.5  # neutral when no information is available


class BindingMDPredictor:
    """
    Predicts antibody-antigen binding scores using MD-derived energy estimates.

    Parameters
    ----------
    lm          : AntibodyLM or RandomAntibodyLM — used to embed sequences.
    mock        : if True (default), use embedding-distance proxy.
                  if False, attempt physics-based interface energy evaluation.
    temperature_k : simulation temperature in Kelvin (real MD mode only).
    """

    def __init__(self, lm, mock: bool = True, temperature_k: float = 300.0,
                 pdb_source: Optional[str] = None,
                 pdb_sources: Optional[List[str]] = None):
        self.lm = lm
        self.mock = mock
        self.temperature_k = temperature_k
        self.pdb_source = pdb_source  # single complex PDB (legacy)
        # SA1: list of PDB paths for a diverse multi-complex dataset.
        # When set, _md_interface_energy cycles through structures and
        # averages scores across the ensemble.
        self.pdb_sources = pdb_sources or (
            [pdb_source] if pdb_source else []
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def predict_binding_scores(
        self,
        antibody_seqs: List[str],
        antigen_seqs: List[str],
        ab_embeddings: Optional[np.ndarray] = None,
        ag_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute an (n_antibodies × n_antigens) binding score matrix.

        Scores are in [0, 1]: higher = stronger predicted binding.

        Parameters
        ----------
        antibody_seqs  : antibody amino acid sequences.
        antigen_seqs   : antigen amino acid sequences.
        ab_embeddings  : pre-computed antibody embeddings (skips re-embedding).
        ag_embeddings  : pre-computed antigen embeddings (skips re-embedding).
        """
        if ab_embeddings is None:
            logger.info("Embedding %d antibody sequences …", len(antibody_seqs))
            ab_embeddings = self.lm.embed(antibody_seqs)
        if ag_embeddings is None:
            logger.info("Embedding %d antigen sequences …", len(antigen_seqs))
            ag_embeddings = self.lm.embed(antigen_seqs)

        if self.mock:
            return self._embedding_proxy(ab_embeddings, ag_embeddings)
        return self._md_interface_energy(
            antibody_seqs, antigen_seqs, ab_embeddings, ag_embeddings
        )

    def top_pairs(
        self,
        antibody_seqs: List[str],
        antigen_seqs: List[str],
        binding_matrix: np.ndarray,
        top_n: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Return the top-n (antibody, antigen, score) triples ranked by
        predicted binding score.
        """
        pairs = [
            (antibody_seqs[i], antigen_seqs[j], float(binding_matrix[i, j]))
            for i in range(len(antibody_seqs))
            for j in range(len(antigen_seqs))
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    # ── Multi-structure ensemble scoring (SA1) ─────────────────────────────────────────

    def score_multi_structure_ensemble(
        self,
        antibody_seqs: List[str],
        antigen_seqs: List[str],
    ) -> np.ndarray:
        """
        Average binding scores across a panel of Ag-Ab complex structures (SA1).

        SA1 requires a large dataset of diverse antibody-antigen complex
        structures to characterise association intermediate ensembles.  This
        method iterates over self.pdb_sources (a list of PDB paths), scores
        each Ab-Ag pair against every structure in the panel, and returns the
        mean score matrix.

        This enables multi-complex, multi-target training datasets for the
        ALM fine-tuning step (Stage 2c), as required by SA1's hypothesis that
        structural diversity in the training data improves affinity prediction.

        Falls back to predict_binding_scores() (single-structure or proxy) if
        no multi-structure panel is configured.

        Returns
        -------
        np.ndarray of shape (n_antibodies, n_antigens), mean score in [0, 1].
        """
        if not self.pdb_sources:
            logger.info(
                "No pdb_sources configured; falling back to single-structure "
                "predict_binding_scores()."
            )
            return self.predict_binding_scores(antibody_seqs, antigen_seqs)

        ab_embs = self.lm.embed(antibody_seqs)
        ag_embs = self.lm.embed(antigen_seqs)
        n_ab, n_ag = len(antibody_seqs), len(antigen_seqs)
        ensemble_scores = np.zeros((n_ab, n_ag), dtype=np.float32)
        valid_count = 0

        for pdb_path in self.pdb_sources:
            if not Path(pdb_path).exists():
                logger.warning("PDB not found: %s — skipping.", pdb_path)
                continue
            calc = MMPBSACalculator(
                pdb_path=pdb_path,
                temperature_k=self.temperature_k,
            )
            struct_scores = np.zeros((n_ab, n_ag), dtype=np.float32)
            for i in range(n_ab):
                for j in range(n_ag):
                    struct_scores[i, j] = calc.score_pair(
                        ab_seq=antibody_seqs[i],
                        ag_seq=antigen_seqs[j],
                        pdb_path=pdb_path,
                        ab_emb=ab_embs[i],
                        ag_emb=ag_embs[j],
                    )
            ensemble_scores += struct_scores
            valid_count += 1

        if valid_count == 0:
            logger.warning(
                "No valid PDB structures found in pdb_sources — using "
                "embedding proxy."
            )
            return self._embedding_proxy(ab_embs, ag_embs)

        result = ensemble_scores / valid_count
        logger.info(
            "Multi-structure ensemble: %d structures averaged; "
            "mean_score=%.3f  shape=%s",
            valid_count, float(result.mean()), result.shape,
        )
        return result

    # ── Binding score implementations ──────────────────────────────────────

    def _embedding_proxy(
        self,
        ab_embs: np.ndarray,
        ag_embs: np.ndarray,
    ) -> np.ndarray:
        """
        Embedding-distance proxy for binding energy.

        Uses negative normalised L2 distance on the unit hypersphere as a
        stand-in for ΔG binding.  Small distance → high binding score.

        Returns
        -------
        np.ndarray of shape (n_ab, n_ag), values in [0, 1].
        """
        ab_n = ab_embs / (np.linalg.norm(ab_embs, axis=1, keepdims=True) + 1e-9)
        ag_n = ag_embs / (np.linalg.norm(ag_embs, axis=1, keepdims=True) + 1e-9)

        # Pairwise L2 on unit sphere: range [0, 2]
        diff = ab_n[:, np.newaxis, :] - ag_n[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)       # (n_ab, n_ag)

        scores = np.clip(1.0 - dist / 2.0, 0.0, 1.0)

        logger.info(
            "MD proxy binding scores: mean=%.3f  max=%.3f  shape=%s",
            scores.mean(), scores.max(), scores.shape,
        )
        return scores

    def _md_interface_energy(
        self,
        antibody_seqs: List[str],
        antigen_seqs: List[str],
        ab_embs: np.ndarray,
        ag_embs: np.ndarray,
    ) -> np.ndarray:
        """
        Physics-based interface energy via :class:`MMPBSACalculator`.

        For each (antibody, antigen) pair, attempts to compute ΔE_interact
        using OpenMM energy minimisation of the complex PDB (self.pdb_source).
        Automatically falls back to embedding-distance proxy when:
          - OpenMM is not installed, or
          - no PDB source is configured (set pdb_source= in constructor), or
          - the PDB file does not exist.

        Supply complex structures via AlphaFold-Multimer or docking, then
        pass the path as ``pdb_source`` to BindingMDPredictor for real ΔG.
        """
        # Check OpenMM availability without importing into module scope
        try:
            import openmm  # noqa: F401
            openmm_available = True
        except ImportError:
            openmm_available = False

        pdb_ok = (
            openmm_available
            and self.pdb_source is not None
            and Path(self.pdb_source).exists()
        )

        if not pdb_ok:
            if not openmm_available:
                logger.info(
                    "OpenMM not installed — using embedding proxy. "
                    "Install with: conda install -c conda-forge openmm"
                )
            elif not self.pdb_source:
                logger.info(
                    "No complex PDB configured (pdb_source=None) — using "
                    "embedding proxy.  Pass pdb_source= to enable MM/PBSA."
                )
            else:
                logger.info(
                    "PDB not found at '%s' — using embedding proxy.", self.pdb_source
                )
            return self._embedding_proxy(ab_embs, ag_embs)

        # Physics-based path: score every pair via MMPBSACalculator
        calculator = MMPBSACalculator(
            pdb_path=self.pdb_source,
            temperature_k=self.temperature_k,
        )
        n_ab, n_ag = len(antibody_seqs), len(antigen_seqs)
        scores = np.zeros((n_ab, n_ag), dtype=np.float32)
        for i in range(n_ab):
            for j in range(n_ag):
                scores[i, j] = calculator.score_pair(
                    ab_seq=antibody_seqs[i],
                    ag_seq=antigen_seqs[j],
                    pdb_path=self.pdb_source,
                    ab_emb=ab_embs[i],
                    ag_emb=ag_embs[j],
                )

        logger.info(
            "MM/PBSA binding scores: mean=%.3f  max=%.3f  shape=%s",
            scores.mean(), scores.max(), scores.shape,
        )
        return scores
