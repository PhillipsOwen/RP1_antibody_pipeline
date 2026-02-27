"""
md_simulations/structural_pathways.py

Synthetic antibody-antigen (Ag-Ab) structural complexes and binding-pathway
simulation for the RP1 pipeline.

Core concept: "Synthetic Ag-Ab structural pathways"
---------------------------------------------------
Antibodies bind antigens along a defined approach/contact/dissociation
trajectory.  This module provides:

  1. Structural feature representations of Ag-Ab complexes (mock or MDTraj-
     based from real PDB files).
  2. Mock or OpenMM steered-MD binding-pathway trajectories.
  3. Featurisation of pathways into arrays suitable for TICA + MSM input,
     providing pathway-level kinetic information (binding on-rate proxies,
     free energy landscapes) separate from the conformational MSM in stage 4.

Classes
-------
AgAbComplexBuilder
    Build structural feature dicts for Ag-Ab complexes.
BindingPathwaySimulator
    Simulate or featurize Ag-Ab binding/unbinding pathways.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Ag-Ab complex builder ────────────────────────────────────────────────────

class AgAbComplexBuilder:
    """
    Build structural feature representations for antibody-antigen complexes.

    Two modes are supported:

    Mock
        Generates synthetic but physically plausible features from sequence
        length statistics.  No 3D coordinates or external tools required.

    Real (MDTraj)
        Loads an actual complex PDB file via MDTraj and computes inter-chain
        Cα distances, a binary contact map, and backbone dihedral angles.
        Requires ``pip install mdtraj``.

    Parameters
    ----------
    n_interface_residues : number of interface residue pairs to model.
    contact_cutoff_nm    : Cα–Cα distance threshold for contact definition (nm).
    """

    def __init__(
        self,
        n_interface_residues: int = 20,
        contact_cutoff_nm: float = 0.8,
    ):
        self.n_interface_residues = n_interface_residues
        self.contact_cutoff_nm = contact_cutoff_nm

    # ── Mock complex ──────────────────────────────────────────────────────────

    def build_mock_complex(
        self, ab_seq: str, ag_seq: str
    ) -> Dict[str, np.ndarray]:
        """
        Build mock structural features for an antibody-antigen complex.

        Parameters
        ----------
        ab_seq : antibody amino acid sequence.
        ag_seq : antigen amino acid sequence.

        Returns
        -------
        dict with keys:

        inter_chain_distances : (n_interface_residues,) Cα distances in nm.
        contact_map           : (n_ab_res, n_ag_res) binary float32 contact matrix.
        backbone_angles       : (n_interface_residues, 2) mock phi/psi in radians.
        n_contacts            : scalar total number of inter-chain contacts.
        """
        n = self.n_interface_residues
        seed = hash(ab_seq[:10] + ag_seq[:10]) % (2 ** 31)
        rng = np.random.default_rng(seed)

        # Inter-chain Cα distances: realistic interface range 0.3–1.0 nm
        distances = rng.uniform(0.3, 1.0, size=n).astype(np.float32)

        # Contact map: sample pairwise distances, threshold at cutoff
        n_ab = min(len(ab_seq), 50) or 10
        n_ag = min(len(ag_seq), 50) or 10
        pairwise = rng.uniform(0.3, 1.5, size=(n_ab, n_ag))
        contact_map = (pairwise < self.contact_cutoff_nm).astype(np.float32)

        # Backbone phi/psi angles: uniform in [-π, π]
        backbone_angles = rng.uniform(-np.pi, np.pi, size=(n, 2)).astype(np.float32)

        return {
            "inter_chain_distances": distances,
            "contact_map": contact_map,
            "backbone_angles": backbone_angles,
            "n_contacts": float(contact_map.sum()),
        }

    # ── Real complex (MDTraj) ─────────────────────────────────────────────────

    def build_from_pdb(
        self,
        pdb_path: str,
        ab_chain: int = 0,
        ag_chain: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Extract structural features from a real Ag-Ab complex PDB file.

        Requires ``pip install mdtraj``.

        Parameters
        ----------
        pdb_path : path to the complex PDB file.
        ab_chain : 0-indexed chain number for the antibody (default 0).
        ag_chain : 0-indexed chain number for the antigen  (default 1).

        Returns
        -------
        dict with same keys as :meth:`build_mock_complex`, computed from
        real Cα coordinates.
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError(
                "mdtraj required for real complex building: pip install mdtraj"
            )

        traj = md.load(pdb_path)
        top = traj.topology

        ab_idx = top.select(f"chainid {ab_chain} and name CA")
        ag_idx = top.select(f"chainid {ag_chain} and name CA")

        if len(ab_idx) == 0 or len(ag_idx) == 0:
            logger.warning(
                "Chain %d or %d not found in %s — using mock features.",
                ab_chain, ag_chain, pdb_path,
            )
            return self.build_mock_complex("", "")

        # Inter-chain distances for sampled interface pairs
        n = min(self.n_interface_residues, len(ab_idx), len(ag_idx))
        pairs = np.stack([ab_idx[:n], ag_idx[:n]], axis=1)
        distances = md.compute_distances(traj, pairs)[0].astype(np.float32)

        # Full contact map between the two chains
        all_pairs = np.array(
            [(a, b) for a in ab_idx for b in ag_idx], dtype=int
        )
        all_dists = md.compute_distances(traj, all_pairs)[0]
        contact_map = (
            all_dists.reshape(len(ab_idx), len(ag_idx)) < self.contact_cutoff_nm
        ).astype(np.float32)

        # Backbone dihedrals (first frame)
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
        n_dih = min(n, phi.shape[1], psi.shape[1])
        backbone_angles = np.stack(
            [phi[0, :n_dih], psi[0, :n_dih]], axis=1
        ).astype(np.float32)

        logger.info(
            "Complex built from PDB: ab_residues=%d  ag_residues=%d  "
            "contacts=%d",
            len(ab_idx), len(ag_idx), int(contact_map.sum()),
        )
        return {
            "inter_chain_distances": distances,
            "contact_map": contact_map,
            "backbone_angles": backbone_angles,
            "n_contacts": float(contact_map.sum()),
        }

    # ── Batch builder ─────────────────────────────────────────────────────────

    def batch_build(
        self,
        ab_seqs: List[str],
        ag_seqs: List[str],
        mock: bool = True,
        pdb_paths: Optional[List[str]] = None,
        ab_chain: int = 0,
        ag_chain: int = 1,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Build structural features for multiple Ag-Ab pairs.

        Parameters
        ----------
        ab_seqs   : antibody amino acid sequences.
        ag_seqs   : antigen amino acid sequences (must match length of ab_seqs).
        mock      : if True, use :meth:`build_mock_complex` for all pairs.
                    If False, attempts :meth:`build_from_pdb` when a valid
                    path is available; falls back to mock otherwise.
        pdb_paths : per-pair PDB paths (same length as ab_seqs, or None).
        ab_chain, ag_chain : chain indices for :meth:`build_from_pdb`.

        Returns
        -------
        List of feature dicts, one per (ab, ag) pair.
        """
        if len(ab_seqs) != len(ag_seqs):
            raise ValueError(
                f"ab_seqs length ({len(ab_seqs)}) != "
                f"ag_seqs length ({len(ag_seqs)})"
            )

        results = []
        for i, (ab, ag) in enumerate(zip(ab_seqs, ag_seqs)):
            pdb = pdb_paths[i] if (pdb_paths and i < len(pdb_paths)) else None
            if not mock and pdb and Path(pdb).exists():
                results.append(
                    self.build_from_pdb(pdb, ab_chain=ab_chain, ag_chain=ag_chain)
                )
            else:
                results.append(self.build_mock_complex(ab, ag))
        return results


# ─── Binding pathway simulator ────────────────────────────────────────────────

class BindingPathwaySimulator:
    """
    Simulate antibody-antigen binding and unbinding pathways.

    Two modes:

    Mock
        Generates a sinusoidal approach → bound → (separation) trajectory
        without any external dependencies.  Extends the
        ``generate_mock_trajectory`` pattern from ``md_runner.py``.

    Steered MD (OpenMM)
        Runs OpenMM with a ``CustomExternalForce`` that linearly increases a
        pulling force on specified antibody atoms, simulating forced
        dissociation along the separation vector.
        Requires openmm: ``conda install -c conda-forge openmm``

    Parameters
    ----------
    n_features : number of structural features per frame in mock mode.
    """

    def __init__(self, n_features: int = 30):
        self.n_features = n_features

    # ── Mock pathway ──────────────────────────────────────────────────────────

    def simulate_mock_pathway(
        self,
        complex_features: Dict[str, np.ndarray],
        n_frames: int = 200,
    ) -> np.ndarray:
        """
        Generate a mock binding-pathway trajectory (three-phase).

        Phase 1 — approach   (frames 0   … n/3)   : antibody closes in.
        Phase 2 — bound      (frames n/3 … 2n/3)  : stable complex.
        Phase 3 — separation (frames 2n/3 … n)    : antibody dissociates.

        Parameters
        ----------
        complex_features : feature dict from :class:`AgAbComplexBuilder`.
        n_frames         : total number of frames in the trajectory.

        Returns
        -------
        np.ndarray of shape (n_frames, n_features), dtype float32.
        """
        t = np.linspace(0, 1, n_frames)
        n_f = self.n_features

        # Gaussian binding-depth envelope: peaks at t = 0.5
        phase = np.exp(-((t - 0.5) ** 2) / (2 * 0.12 ** 2))

        # Base inter-chain distances from complex features
        d0 = complex_features.get(
            "inter_chain_distances", np.ones(n_f, dtype=np.float32) * 1.0
        )
        # Tile / truncate to match n_f
        if len(d0) >= n_f:
            bound_d = d0[:n_f]
        else:
            reps = (n_f // len(d0)) + 1
            bound_d = np.tile(d0, reps)[:n_f]

        # Distance trajectory: unbound ↔ bound modulated by Gaussian phase
        dist_traj = (
            np.outer(1.0 - phase, bound_d * 1.5)
            + np.outer(phase, bound_d * 0.7)
        )

        # Random thermal fluctuations
        noise = np.random.randn(n_frames, n_f) * 0.02

        # Slow sinusoidal mode gives MSM a detectable slow process
        slow = np.outer(np.sin(t * 2.0 * np.pi), np.ones(n_f)) * 0.05

        features = (dist_traj + noise + slow).astype(np.float32)
        return features

    # ── Steered MD (OpenMM) ───────────────────────────────────────────────────

    def simulate_steered_md(
        self,
        pdb_path: str,
        pull_residues: Optional[List[int]] = None,
        n_steps: int = 50_000,
        output_dir: str = "md_simulations/output/pathways",
    ) -> str:
        """
        Run steered MD to simulate antibody dissociation from antigen.

        A ``CustomExternalForce`` applies a linearly increasing pulling force
        on *pull_residues* along the antibody–antigen separation vector,
        driving the complex toward the unbound state.

        Requires openmm: ``conda install -c conda-forge openmm``

        Parameters
        ----------
        pdb_path      : path to the starting Ag-Ab complex PDB.
        pull_residues : 0-indexed atom indices (Cα) of antibody residues to
                        pull.  If None, the first 10 Cα atoms in chain 0 are
                        used automatically.
        n_steps       : integration steps for the steered simulation.
        output_dir    : directory for output DCD trajectory.

        Returns
        -------
        Path to the output DCD trajectory file (str).
        """
        try:
            from openmm.app import (
                PDBFile, ForceField, Simulation, DCDReporter, NoCutoff,
            )
            from openmm import (
                LangevinMiddleIntegrator, CustomExternalForce,
            )
            from openmm.unit import kelvin, picosecond, femtosecond
        except ImportError:
            raise ImportError(
                "openmm required for steered MD: "
                "conda install -c conda-forge openmm"
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        traj_path = str(out / "steered_pathway.dcd")

        logger.info("Setting up steered MD from %s …", pdb_path)
        pdb = PDBFile(pdb_path)
        topology = pdb.topology
        positions = pdb.positions

        # Identify atoms to pull (antibody = chain 0 Cα atoms)
        if pull_residues is None:
            ca_atoms = [
                a.index for a in topology.atoms()
                if a.name == "CA" and a.residue.chain.index == 0
            ]
            pull_atoms = ca_atoms[:10]
        else:
            pull_atoms = pull_residues

        if not pull_atoms:
            raise ValueError("No pull atoms identified; check PDB chain layout.")

        # Compute pull direction: antibody centroid → antigen centroid
        pos_array = np.array([[v.x, v.y, v.z] for v in positions])
        ag_atoms = [
            a.index for a in topology.atoms() if a.residue.chain.index != 0
        ]
        ab_centroid = pos_array[pull_atoms].mean(axis=0)
        ag_centroid = (
            pos_array[ag_atoms].mean(axis=0) if ag_atoms else np.zeros(3)
        )
        pull_dir = ab_centroid - ag_centroid
        norm = np.linalg.norm(pull_dir)
        if norm > 0:
            pull_dir /= norm

        # Build system
        ff = ForceField("amber14-all.xml")
        system = ff.createSystem(topology, nonbondedMethod=NoCutoff)

        # Pulling force: F = k * step_frac * (pull_dir · r)
        # step_frac increases from 0 → 1 over the simulation
        dx, dy, dz = pull_dir
        pull_force = CustomExternalForce(
            f"k * frac * ({dx:.6f} * x + {dy:.6f} * y + {dz:.6f} * z)"
        )
        pull_force.addGlobalParameter("k", 1000.0)   # kJ/(mol·nm)
        pull_force.addGlobalParameter("frac", 0.0)
        for idx in pull_atoms:
            pull_force.addParticle(idx)
        system.addForce(pull_force)

        integrator = LangevinMiddleIntegrator(
            300.0 * kelvin, 1.0 / picosecond, 2.0 * femtosecond
        )
        sim = Simulation(topology, system, integrator)
        sim.context.setPositions(positions)
        sim.minimizeEnergy(maxIterations=200)
        sim.reporters.append(DCDReporter(traj_path, max(1, n_steps // 100)))

        # Run in segments with linearly increasing pull fraction
        n_segments = 20
        segment_steps = max(1, n_steps // n_segments)
        for seg in range(n_segments):
            frac = seg / n_segments
            sim.context.setParameter("frac", frac)
            sim.step(segment_steps)

        logger.info("Steered MD trajectory saved to %s", traj_path)
        return traj_path

    # ── Association intermediate state labeling (SA1) ─────────────────────

    def label_intermediate_states(
        self,
        trajectory: np.ndarray,
        bound_threshold: float = 0.55,
        unbound_threshold: float = 0.85,
    ) -> np.ndarray:
        """
        Assign discrete state labels to each trajectory frame (SA1).

        SA1 hypothesis: association intermediate structures play an essential
        role in antibody-antigen affinity.  This method provides the explicit
        frame-level annotation needed to:
          1. Identify which frames belong to the association intermediate regime.
          2. Feed labelled intermediate frames into MSM motif extraction.
          3. Provide training signal distinguishing intermediate-state features
             from both the unbound and final bound states.

        Labels
        ------
        0 -- unbound      : mean inter-chain distance above *unbound_threshold*.
        1 -- intermediate : mean distance between thresholds (association pathway).
        2 -- bound        : mean distance below *bound_threshold*.

        Parameters
        ----------
        trajectory        : (n_frames, n_features) float array from
                            featurize_pathway() or simulate_mock_pathway().
        bound_threshold   : mean-distance cutoff below which the complex is
                            considered bound (nm units in real MD; arbitrary
                            units in mock mode).
        unbound_threshold : mean-distance cutoff above which the complex is
                            considered fully separated.

        Returns
        -------
        np.ndarray of shape (n_frames,), dtype int8.
        Values: 0 = unbound, 1 = intermediate, 2 = bound.
        """
        mean_dist = trajectory.mean(axis=1)  # (n_frames,)

        labels = np.ones(len(mean_dist), dtype=np.int8)  # default: intermediate
        labels[mean_dist < bound_threshold] = 2           # bound
        labels[mean_dist > unbound_threshold] = 0         # unbound

        counts = {
            "unbound": int((labels == 0).sum()),
            "intermediate": int((labels == 1).sum()),
            "bound": int((labels == 2).sum()),
        }
        logger.info(
            "State labels assigned: unbound=%d  intermediate=%d  bound=%d",
            counts["unbound"], counts["intermediate"], counts["bound"],
        )
        return labels

    # ── Pathway featurisation ─────────────────────────────────────────────────

    def featurize_pathway(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Featurize a binding-pathway trajectory for TICA / MSM input.

        Appends summary statistics (mean distance, std, mean velocity) to
        the raw feature matrix to help TICA detect the slow binding mode.

        Parameters
        ----------
        trajectory : (n_frames, n_raw_features) float array.

        Returns
        -------
        np.ndarray of shape (n_frames, n_raw_features + 3), dtype float32.
        """
        # Per-frame mean distance (proxy for Ab–Ag separation)
        mean_dist = trajectory.mean(axis=1, keepdims=True)
        # Per-frame std (proxy for interface flexibility)
        std_dist = trajectory.std(axis=1, keepdims=True)
        # Per-frame mean velocity magnitude (proxy for binding kinetics)
        velocity = np.zeros_like(trajectory)
        velocity[1:] = trajectory[1:] - trajectory[:-1]
        mean_vel = np.abs(velocity).mean(axis=1, keepdims=True)

        return np.hstack(
            [trajectory, mean_dist, std_dist, mean_vel]
        ).astype(np.float32)
