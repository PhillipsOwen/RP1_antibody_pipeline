"""
msm_analysis/msm_builder.py
Markov State Model construction and analysis.

Takes MD trajectory features, builds an MSM, and exposes:
  - Macrostate assignments
  - Relaxation timescales
  - Stationary distribution
  - Transition pathways (via Transition Path Theory)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MSMBuilder:
    """
    High-level MSM builder wrapping PyEMMA (or a lightweight NumPy fallback).

    Parameters
    ----------
    lag_time   : lag time in trajectory frames
    n_states   : number of macro-states (after PCCA+ lumping)
    n_jobs     : parallel jobs for clustering / TICA
    output_dir : directory for saved models / plots
    """

    def __init__(self, lag_time: int = 10, n_states: int = 20,
                 n_jobs: int = 4,
                 output_dir: str = "msm_analysis/output"):
        self.lag_time = lag_time
        self.n_states = n_states
        self.n_jobs = n_jobs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._msm = None            # PyEMMA MSM object
        self._cluster = None        # cluster object
        self._tica = None           # TICA object

    # ── Data preparation ─────────────────────────────────────────────────────

    def tica(self, features: np.ndarray, dim: int = 10,
             lag: Optional[int] = None) -> np.ndarray:
        """
        Apply TICA (Time-lagged Independent Component Analysis) to *features*.

        Parameters
        ----------
        features : (n_frames, n_features) array
        dim      : number of TICA components to keep
        lag      : TICA lag (defaults to self.lag_time)

        Returns
        -------
        np.ndarray of shape (n_frames, dim)
        """
        lag = lag or self.lag_time
        try:
            import pyemma
            self._tica = pyemma.coordinates.tica(features, lag=lag, dim=dim)
            return self._tica.get_output()[0]
        except ImportError:
            logger.warning("PyEMMA not found — using PCA fallback.")
            return self._pca_fallback(features, dim)

    def cluster(self, projected: np.ndarray, n_micro: int = 200,
                method: str = "kmeans") -> np.ndarray:
        """
        Cluster projected features into micro-states.

        Parameters
        ----------
        projected : (n_frames, dim) TICA-projected data
        n_micro   : number of micro-states (k-means centroids)
        method    : 'kmeans' (only supported method currently)

        Returns
        -------
        dtrajs : (n_frames,) integer array of micro-state assignments
        """
        try:
            import pyemma
            cl = pyemma.coordinates.cluster_kmeans(
                projected, k=n_micro, max_iter=100, n_jobs=self.n_jobs
            )
            self._cluster = cl
            return cl.dtrajs[0]
        except ImportError:
            logger.warning("PyEMMA not found — using sklearn KMeans fallback.")
            return self._kmeans_fallback(projected, n_micro)

    # ── MSM estimation ────────────────────────────────────────────────────────

    def estimate(self, dtrajs: np.ndarray) -> "MSMBuilder":
        """
        Estimate MSM from discrete trajectories *dtrajs*.

        Sets self._msm and returns self for chaining.
        """
        try:
            import pyemma
            self._msm = pyemma.msm.estimate_markov_model(
                dtrajs, lag=self.lag_time, reversible=True
            )
            logger.info(
                "MSM estimated: %d active states, fraction active=%.2f",
                self._msm.nstates, self._msm.active_state_fraction
            )
        except ImportError:
            logger.warning("PyEMMA not found — using NumPy MSM fallback.")
            self._msm = _NumpyMSM(dtrajs, self.lag_time)

        return self

    # ── Analysis ──────────────────────────────────────────────────────────────

    @property
    def timescales(self) -> np.ndarray:
        """Relaxation timescales in frames (sorted descending)."""
        self._check_estimated()
        ts = self._msm.timescales()
        return np.sort(ts)[::-1]

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Equilibrium probability of each micro-state."""
        self._check_estimated()
        return self._msm.stationary_distribution

    def lump_to_macrostates(self) -> np.ndarray:
        """
        Apply PCCA+ to lump micro-states into *n_states* macro-states.

        Returns
        -------
        macro_assignments : array mapping micro-state → macro-state index
        """
        self._check_estimated()
        try:
            self._msm.pcca(self.n_states)
            return self._msm.metastable_assignments
        except AttributeError:
            # Fallback for NumPy MSM
            return np.zeros(self._msm.n_states, dtype=int)

    def transition_paths(self, source_states: List[int],
                         sink_states: List[int]) -> dict:
        """
        Compute Transition Path Theory (TPT) committor and net fluxes.

        Parameters
        ----------
        source_states : micro-state indices for 'unbound' (A set)
        sink_states   : micro-state indices for 'bound'  (B set)

        Returns
        -------
        dict with keys: 'forward_committor', 'net_flux', 'rate'
        """
        self._check_estimated()
        try:
            import pyemma
            tpt = pyemma.msm.tpt(self._msm, source_states, sink_states)
            return {
                "forward_committor": tpt.committor,
                "net_flux": tpt.net_flux,
                "rate": tpt.rate,
            }
        except (ImportError, AttributeError):
            logger.warning("TPT requires PyEMMA — returning empty result.")
            return {}

    def free_energy_landscape(self, projected: np.ndarray,
                               temperature_k: float = 300.0) -> np.ndarray:
        """
        Estimate free energy from histogram of projected coordinates.

        Returns
        -------
        np.ndarray of shape (n_bins, n_bins) — 2D FEL for first two dims.
        """
        kT = 0.008314 * temperature_k  # kJ/mol
        x, y = projected[:, 0], projected[:, 1]
        hist, _, _ = np.histogram2d(x, y, bins=50, density=True)
        hist = np.where(hist > 0, hist, 1e-10)
        return -kT * np.log(hist)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, tag: str = "msm") -> str:
        """Save MSM to disk; returns file path."""
        path = str(self.output_dir / f"{tag}.pkl")
        try:
            import pyemma
            self._msm.save(path, overwrite=True)
        except AttributeError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self._msm, f)
        logger.info("MSM saved to %s", path)
        return path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_estimated(self):
        if self._msm is None:
            raise RuntimeError("Call .estimate() before accessing MSM properties.")

    @staticmethod
    def _pca_fallback(features: np.ndarray, dim: int) -> np.ndarray:
        from sklearn.decomposition import PCA  # type: ignore[import]
        pca = PCA(n_components=dim)
        return pca.fit_transform(features)

    @staticmethod
    def _kmeans_fallback(data: np.ndarray, k: int) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]
        km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
        return km.fit_predict(data)


# ─── Lightweight NumPy MSM (no PyEMMA) ───────────────────────────────────────

class _NumpyMSM:
    """
    Minimal MSM estimated from a discrete trajectory using NumPy.
    Supports stationary distribution and implied timescales.
    """

    def __init__(self, dtrajs: np.ndarray, lag: int):
        self.lag = lag
        self._estimate(dtrajs)

    def _estimate(self, dtrajs: np.ndarray):
        states = np.unique(dtrajs)
        self.n_states = len(states)
        state_map = {s: i for i, s in enumerate(states)}
        mapped = np.array([state_map[s] for s in dtrajs])
        C = np.zeros((self.n_states, self.n_states))
        for i in range(len(mapped) - self.lag):
            C[mapped[i], mapped[i + self.lag]] += 1
        # Symmetrise for reversible MSM
        C = (C + C.T) / 2
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.T = C / row_sums  # transition matrix
        # Stationary distribution via power iteration
        pi = np.ones(self.n_states) / self.n_states
        for _ in range(1000):
            pi_new = pi @ self.T
            if np.allclose(pi, pi_new, atol=1e-10):
                break
            pi = pi_new
        self.stationary_distribution = pi / pi.sum()

    def timescales(self) -> np.ndarray:
        eigenvalues = np.linalg.eigvals(self.T)
        eigenvalues = np.sort(np.abs(eigenvalues.real))[::-1]
        eigenvalues = eigenvalues[1:]  # skip stationary eigenvalue (1)
        eigenvalues = np.clip(eigenvalues, 1e-10, 1 - 1e-10)
        return -self.lag / np.log(eigenvalues)
