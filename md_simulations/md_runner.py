"""
md_simulations/md_runner.py
Molecular Dynamics runner and trajectory feature extractor.

Supports:
  - Running OpenMM simulations (optional, requires openmm)
  - Running GROMACS simulations (optional, requires GROMACS installation)
  - Running CHARMM simulations (optional, requires CHARMM installation)
  - Loading pre-computed trajectories via MDTraj
  - Extracting CDR-loop inter-residue distances for MSM featurisation

MD software installation
------------------------
OpenMM  :  conda install -c conda-forge openmm
GROMACS :  conda install -c bioconda gromacs
           or build from source: https://manual.gromacs.org/current/install-guide
CHARMM  :  academic licence required — https://www.charmm.org/charmm/
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Feature extraction (MDTraj) ─────────────────────────────────────────────

class TrajectoryAnalyzer:
    """
    Loads an MD trajectory and computes structural features.

    Parameters
    ----------
    topology_file   : path to PDB topology
    trajectory_file : path to XTC / DCD trajectory (or None for PDB-only)
    """

    def __init__(self, topology_file: str, trajectory_file: Optional[str] = None):
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self._traj = None

    @property
    def traj(self):
        if self._traj is None:
            self._traj = self._load()
        return self._traj

    def _load(self):
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("mdtraj required: pip install mdtraj")

        if self.trajectory_file and Path(self.trajectory_file).exists():
            logger.info("Loading trajectory %s …", self.trajectory_file)
            return md.load(self.trajectory_file, top=self.topology_file)
        else:
            logger.info("No trajectory — loading topology only: %s",
                        self.topology_file)
            return md.load(self.topology_file)

    # ── Feature extraction ────────────────────────────────────────────────────

    def get_ca_distances(self, atom_pairs: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """
        Compute inter-Cα distances across all frames.

        Parameters
        ----------
        atom_pairs : (M, 2) integer array of CA atom indices.
                     If None, uses all unique Cα pairs (expensive for large proteins).

        Returns
        -------
        np.ndarray of shape (n_frames, n_pairs) in nm.
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("mdtraj required: pip install mdtraj")

        if atom_pairs is None:
            ca_idx = self.traj.topology.select("name CA")
            from itertools import combinations
            atom_pairs = np.array(list(combinations(ca_idx, 2)))

        logger.info("Computing %d CA-distance pairs across %d frames …",
                    len(atom_pairs), len(self.traj))
        distances = md.compute_distances(self.traj, atom_pairs)
        return distances

    def get_phi_psi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute backbone phi/psi dihedrals.

        Returns
        -------
        phi : (n_frames, n_residues-1)
        psi : (n_frames, n_residues-1)
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("mdtraj required: pip install mdtraj")

        _, phi = md.compute_phi(self.traj)
        _, psi = md.compute_psi(self.traj)
        return phi, psi

    def get_contact_map(self, cutoff_nm: float = 0.8) -> np.ndarray:
        """
        Binary contact maps (per frame) using Cα cutoff.

        Returns
        -------
        np.ndarray of shape (n_frames, n_residues, n_residues) bool
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("mdtraj required: pip install mdtraj")

        contacts, res_pairs = md.compute_contacts(self.traj, scheme="ca")
        n_res = self.traj.topology.n_residues
        maps = np.zeros((len(self.traj), n_res, n_res), dtype=bool)
        for t in range(len(self.traj)):
            for k, (i, j) in enumerate(res_pairs):
                if contacts[t, k] < cutoff_nm:
                    maps[t, i, j] = maps[t, j, i] = True
        return maps

    def featurize(self, method: str = "distances",
                  atom_pairs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Unified featurisation entry point.

        method : 'distances' | 'dihedrals' | 'combined'
        Returns np.ndarray of shape (n_frames, n_features).
        """
        if method == "distances":
            return self.get_ca_distances(atom_pairs)
        elif method == "dihedrals":
            phi, psi = self.get_phi_psi()
            return np.hstack([phi, psi])
        elif method == "combined":
            dist = self.get_ca_distances(atom_pairs)
            phi, psi = self.get_phi_psi()
            min_frames = min(dist.shape[0], phi.shape[0])
            return np.hstack([dist[:min_frames], phi[:min_frames],
                               psi[:min_frames]])
        else:
            raise ValueError(f"Unknown method: {method}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        t = self.traj
        return {
            "n_frames": len(t),
            "n_atoms": t.n_atoms,
            "n_residues": t.n_residues,
            "time_ns": float(t.time[-1] / 1000) if len(t) > 0 else 0.0,
        }


# ─── OpenMM Simulation runner ─────────────────────────────────────────────────

class OpenMMRunner:
    """
    Thin wrapper for running OpenMM simulations.

    Requires: openmm (pip install openmm)
    """

    def __init__(self, pdb_file: str, forcefield: str = "amber14-all.xml",
                 water_model: str = "amber14/tip3pfb.xml",
                 temperature_k: float = 300.0,
                 step_size_fs: float = 2.0,
                 output_dir: str = "md_simulations/output"):
        self.pdb_file = pdb_file
        self.forcefield = forcefield
        self.water_model = water_model
        self.temperature_k = temperature_k
        self.step_size_fs = step_size_fs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, n_steps: int = 1_000_000,
            report_interval: int = 1000) -> str:
        """
        Minimise, equilibrate, and produce NVT MD trajectory.

        Returns
        -------
        Path to output trajectory file.
        """
        try:
            from openmm.app import (PDBFile, ForceField, Simulation,
                                    DCDReporter, StateDataReporter,
                                    PME, HBonds)
            from openmm import LangevinMiddleIntegrator, Platform
            from openmm.unit import (kelvin, picosecond, femtosecond,
                                     nanometer, kilocalories_per_mole)
        except ImportError:
            raise ImportError("openmm required: pip install openmm")

        traj_path = str(self.output_dir / "trajectory.dcd")
        log_path = str(self.output_dir / "md_log.csv")

        logger.info("Setting up OpenMM simulation …")
        pdb = PDBFile(self.pdb_file)
        ff = ForceField(self.forcefield, self.water_model)
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds,
        )
        integrator = LangevinMiddleIntegrator(
            self.temperature_k * kelvin,
            1.0 / picosecond,
            self.step_size_fs * femtosecond,
        )
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        logger.info("Energy minimisation …")
        simulation.minimizeEnergy()

        simulation.reporters.append(DCDReporter(traj_path, report_interval))
        simulation.reporters.append(
            StateDataReporter(log_path, report_interval,
                              step=True, potentialEnergy=True, temperature=True)
        )

        logger.info("Running %d steps …", n_steps)
        simulation.step(n_steps)
        logger.info("Trajectory saved to %s", traj_path)
        return traj_path


# ─── GROMACS runner ───────────────────────────────────────────────────────────

class GROMACSRunner:
    """
    Thin wrapper for running GROMACS (gmx) molecular dynamics simulations.

    Requires GROMACS to be installed and the ``gmx`` binary on PATH.
    Install: conda install -c bioconda gromacs
             or build from source: https://manual.gromacs.org

    Workflow
    --------
    1. pdb2gmx   — convert PDB to GROMACS topology + coordinate files.
    2. editconf  — define simulation box.
    3. solvate   — add solvent.
    4. grompp    — pre-process to produce a .tpr run input file.
    5. mdrun     — execute the simulation.

    Parameters
    ----------
    pdb_file      : path to the input PDB structure.
    forcefield    : GROMACS force-field name (e.g. 'amber99sb-ildn').
    water_model   : water model name (e.g. 'tip3p').
    temperature_k : simulation temperature in Kelvin.
    output_dir    : directory where all GROMACS output files are written.
    gmx_binary    : name / path of the GROMACS executable.
    """

    _NVT_MDP = textwrap.dedent("""\
        ; NVT equilibration / production
        integrator  = md
        nsteps      = {nsteps}
        dt          = {dt}
        nstxout     = {nstxout}
        nstvout     = {nstxout}
        nstenergy   = {nstxout}
        nstlog      = {nstxout}
        continuation = no
        constraint_algorithm = lincs
        constraints = h-bonds
        lincs_iter  = 1
        lincs_order = 4
        cutoff-scheme = Verlet
        ns_ns_type  = grid
        rcoulomb    = 1.0
        rvdw        = 1.0
        DispCorr    = EnerPres
        coulombtype = PME
        pme_order   = 4
        fourierspacing = 0.16
        tcoupl      = V-rescale
        tc-grps     = Protein Non-Protein
        tau_t       = 0.1 0.1
        ref_t       = {temp} {temp}
        pcoupl      = no
        pbc         = xyz
        gen_vel     = yes
        gen_temp    = {temp}
        gen_seed    = -1
    """)

    def __init__(
        self,
        pdb_file: str,
        forcefield: str = "amber99sb-ildn",
        water_model: str = "tip3p",
        temperature_k: float = 300.0,
        output_dir: str = "md_simulations/output",
        gmx_binary: str = "gmx",
    ):
        self.pdb_file = pdb_file
        self.forcefield = forcefield
        self.water_model = water_model
        self.temperature_k = temperature_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gmx = gmx_binary

    def run(
        self,
        n_steps: int = 1_000_000,
        report_interval: int = 1000,
    ) -> str:
        """
        Run a GROMACS NVT simulation.

        Returns
        -------
        Path to the output trajectory (.xtc) file.
        """
        if not shutil.which(self.gmx):
            raise EnvironmentError(
                f"GROMACS binary '{self.gmx}' not found on PATH. "
                "Install: conda install -c bioconda gromacs"
            )

        out = self.output_dir
        tpr = str(out / "topol.tpr")
        traj = str(out / "trajectory.xtc")
        mdp = str(out / "nvt.mdp")
        gro = str(out / "conf.gro")
        top = str(out / "topol.top")

        # Write MDP file
        Path(mdp).write_text(self._NVT_MDP.format(
            nsteps=n_steps,
            dt=0.002,
            nstxout=report_interval,
            temp=int(self.temperature_k),
        ))

        logger.info("GROMACS: pdb2gmx …")
        self._run_gmx([
            "pdb2gmx", "-f", self.pdb_file,
            "-o", gro, "-p", top,
            "-ff", self.forcefield,
            "-water", self.water_model,
            "-ignh",
        ])

        logger.info("GROMACS: editconf (cubic box) …")
        boxed = str(out / "boxed.gro")
        self._run_gmx(["editconf", "-f", gro, "-o", boxed,
                       "-c", "-d", "1.0", "-bt", "cubic"])

        logger.info("GROMACS: solvate …")
        solvated = str(out / "solvated.gro")
        self._run_gmx(["solvate", "-cp", boxed, "-cs", "spc216.gro",
                       "-o", solvated, "-p", top])

        logger.info("GROMACS: grompp …")
        self._run_gmx(["grompp", "-f", mdp, "-c", solvated,
                       "-p", top, "-o", tpr])

        logger.info("GROMACS: mdrun (%d steps) …", n_steps)
        self._run_gmx([
            "mdrun", "-v", "-deffnm", str(out / "md"),
            "-s", tpr, "-x", traj,
        ])

        logger.info("GROMACS trajectory saved to %s", traj)
        return traj

    def _run_gmx(self, args: List[str]) -> None:
        cmd = [self.gmx] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"GROMACS command failed: {' '.join(cmd)}\n{result.stderr}"
            )


# ─── CHARMM runner ────────────────────────────────────────────────────────────

class CHARMMRunner:
    """
    Thin wrapper for running CHARMM molecular dynamics simulations.

    Requires a CHARMM binary (academic licence: https://www.charmm.org).
    CHARMM is driven by an input script (.inp); this class generates a
    minimal NVT script from a PDB file using the CHARMM36m force field.

    Parameters
    ----------
    pdb_file       : path to the input PDB structure.
    charmm_binary  : name / path of the CHARMM executable.
    forcefield_dir : directory containing CHARMM36m parameter files
                     (toppar/*.rtf, *.prm).  Download from mackerell.umaryland.edu.
    temperature_k  : simulation temperature in Kelvin.
    output_dir     : directory for CHARMM output files.
    """

    _CHARMM_SCRIPT = textwrap.dedent("""\
        * Minimal NVT MD — generated by RP1 pipeline
        *

        ! Read force field
        stream @toppardir/toppar_water_ions.str
        read rtf  card name @toppardir/top_all36_prot.rtf
        read param card flex name @toppardir/par_all36m_prot.prm

        ! Read structure
        read sequence pdb name @pdbfile
        generate PROT setup

        read coor pdb name @pdbfile resid

        ! Minimise
        mini sd nstep 1000

        ! Dynamics
        set nstep = @nsteps
        dynamics leap start timestep 0.002 nstep @nstep -
            firstt @temp finalt @temp tbath @temp -
            iunread -1 iunwri @dcdunit iuncrd -1 -
            nprint @nprint nsavc @nprint -
            iasors 0 iasvel 1

        stop
    """)

    def __init__(
        self,
        pdb_file: str,
        charmm_binary: str = "charmm",
        forcefield_dir: str = "toppar",
        temperature_k: float = 300.0,
        output_dir: str = "md_simulations/output",
    ):
        self.pdb_file = pdb_file
        self.charmm = charmm_binary
        self.forcefield_dir = forcefield_dir
        self.temperature_k = temperature_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_steps: int = 1_000_000,
        report_interval: int = 1000,
    ) -> str:
        """
        Run a CHARMM NVT simulation.

        Returns
        -------
        Path to the output DCD trajectory file.
        """
        if not shutil.which(self.charmm):
            raise EnvironmentError(
                f"CHARMM binary '{self.charmm}' not found on PATH. "
                "Obtain licence and install from https://www.charmm.org"
            )

        out = self.output_dir
        script_path = str(out / "md.inp")
        dcd_path = str(out / "trajectory.dcd")

        # Write CHARMM input script
        Path(script_path).write_text(self._CHARMM_SCRIPT.format(
            toppardir=self.forcefield_dir,
            pdbfile=self.pdb_file,
            nsteps=n_steps,
            temp=int(self.temperature_k),
            nprint=report_interval,
            dcdunit=30,
        ))

        logger.info("CHARMM: running %d steps …", n_steps)
        result = subprocess.run(
            [self.charmm, "-i", script_path, "-o", str(out / "md.log")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CHARMM simulation failed.\n{result.stderr}"
            )

        logger.info("CHARMM trajectory saved to %s", dcd_path)
        return dcd_path


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_md_runner(
    backend: str,
    pdb_file: str,
    temperature_k: float = 300.0,
    output_dir: str = "md_simulations/output",
    **kwargs,
):
    """
    Return the appropriate MD runner for the requested *backend*.

    Parameters
    ----------
    backend      : 'openmm' | 'gromacs' | 'charmm'.
    pdb_file     : path to the input PDB structure file.
    temperature_k: simulation temperature in Kelvin.
    output_dir   : output directory for trajectory files.
    **kwargs     : additional keyword arguments forwarded to the runner class.

    Returns
    -------
    OpenMMRunner | GROMACSRunner | CHARMMRunner
    """
    backend = backend.lower()
    if backend == "openmm":
        return OpenMMRunner(
            pdb_file=pdb_file,
            temperature_k=temperature_k,
            output_dir=output_dir,
            **kwargs,
        )
    elif backend == "gromacs":
        return GROMACSRunner(
            pdb_file=pdb_file,
            temperature_k=temperature_k,
            output_dir=output_dir,
            **kwargs,
        )
    elif backend == "charmm":
        return CHARMMRunner(
            pdb_file=pdb_file,
            temperature_k=temperature_k,
            output_dir=output_dir,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown MD backend '{backend}'. Choose: 'openmm', 'gromacs', 'charmm'."
        )


# ─── Mock trajectory (for testing without MD files) ──────────────────────────

def generate_mock_trajectory(n_frames: int = 500,
                             n_features: int = 100) -> np.ndarray:
    """
    Return a fake trajectory feature matrix for pipeline testing.
    Adds slow sinusoidal drift to simulate conformational transitions.
    """
    t = np.linspace(0, 4 * np.pi, n_frames)
    slow = np.outer(np.sin(t), np.ones(n_features // 4))
    fast = np.random.randn(n_frames, n_features - n_features // 4) * 0.1
    return np.hstack([slow, fast])
