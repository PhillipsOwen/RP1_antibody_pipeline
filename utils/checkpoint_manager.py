"""
checkpoint_manager.py — Save and load intermediate pipeline data at process milestones.

This module provides utilities to checkpoint pipeline outputs at each major stage,
enabling resume-from-checkpoint functionality and detailed analysis of intermediate results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of intermediate pipeline data.

    Each checkpoint includes:
    - Stage metadata (timestamp, stage name, config snapshot)
    - Primary outputs (sequences, scores, matrices, models)
    - Quality metrics (summary statistics)
    """

    def __init__(self, checkpoint_dir: str = "RP1_antibody_pipeline/experiments/checkpoints", run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : directory to store checkpoints
        run_id : unique identifier for this pipeline run (auto-generated if None)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.run_dir = self.checkpoint_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {self.run_dir}")

    def save_stage(self, stage_name: str, data: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint for a pipeline stage.

        Parameters
        ----------
        stage_name : identifier for the stage (e.g., "stage_0_escape_panel")
        data : dictionary containing stage outputs
        metadata : optional additional metadata

        Returns
        -------
        Path to the saved checkpoint directory
        """
        stage_dir = self.run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_time = datetime.now().isoformat()

        # Save metadata
        meta = {
            "stage_name": stage_name,
            "timestamp": checkpoint_time,
            "run_id": self.run_id,
        }
        if metadata:
            meta.update(metadata)

        with open(stage_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save data components
        for key, value in data.items():
            self._save_object(stage_dir, key, value)

        # Create summary
        summary = self._generate_summary(data)
        with open(stage_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {stage_dir}")
        return str(stage_dir)

    def _save_object(self, directory: Path, name: str, obj: Any):
        """Save individual object with appropriate format."""
        if obj is None:
            return

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            np.save(directory / f"{name}.npy", obj)

        # Lists of strings (sequences)
        elif isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            with open(directory / f"{name}.txt", "w") as f:
                f.write("\n".join(obj))

        # Lists of numbers (scores)
        elif isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            np.save(directory / f"{name}.npy", np.array(obj))

        # Dictionaries (JSON-serializable)
        elif isinstance(obj, dict):
            try:
                with open(directory / f"{name}.json", "w") as f:
                    json.dump(obj, f, indent=2, default=str)
            except (TypeError, ValueError):
                # Fall back to pickle for complex dicts
                with open(directory / f"{name}.pkl", "wb") as f:
                    pickle.dump(obj, f)

        # Lists of complex objects
        elif isinstance(obj, list):
            with open(directory / f"{name}.pkl", "wb") as f:
                pickle.dump(obj, f)

        # Generic objects (models, etc.)
        else:
            with open(directory / f"{name}.pkl", "wb") as f:
                pickle.dump(obj, f)

    def _generate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for checkpoint data."""
        summary = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                summary[key] = {
                    "type": "ndarray",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "mean": float(np.mean(value)) if value.size > 0 else None,
                    "std": float(np.std(value)) if value.size > 0 else None,
                    "min": float(np.min(value)) if value.size > 0 else None,
                    "max": float(np.max(value)) if value.size > 0 else None,
                }
            elif isinstance(value, list):
                summary[key] = {
                    "type": "list",
                    "length": len(value),
                    "sample": str(value[0]) if len(value) > 0 else None,
                }
            elif isinstance(value, dict):
                summary[key] = {
                    "type": "dict",
                    "keys": list(value.keys()),
                }
            else:
                summary[key] = {
                    "type": type(value).__name__,
                }

        return summary

    def load_stage(self, stage_name: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint for a pipeline stage.

        Parameters
        ----------
        stage_name : identifier for the stage
        run_id : run identifier (uses current run_id if None)

        Returns
        -------
        Dictionary containing stage outputs
        """
        if run_id is None:
            run_id = self.run_id

        stage_dir = self.checkpoint_dir / run_id / stage_name

        if not stage_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {stage_dir}")

        data = {}

        # Load all files in the stage directory
        for file_path in stage_dir.iterdir():
            if file_path.name in ["metadata.json", "summary.json"]:
                continue

            key = file_path.stem

            if file_path.suffix == ".npy":
                data[key] = np.load(file_path, allow_pickle=True)
            elif file_path.suffix == ".txt":
                with open(file_path, "r") as f:
                    data[key] = [line.strip() for line in f]
            elif file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    data[key] = json.load(f)
            elif file_path.suffix == ".pkl":
                with open(file_path, "rb") as f:
                    data[key] = pickle.load(f)

        logger.info(f"Checkpoint loaded: {stage_dir}")
        return data

    def list_stages(self, run_id: Optional[str] = None) -> List[str]:
        """List all saved stages for a run."""
        if run_id is None:
            run_id = self.run_id

        run_dir = self.checkpoint_dir / run_id

        if not run_dir.exists():
            return []

        return [d.name for d in run_dir.iterdir() if d.is_dir()]

    def list_runs(self) -> List[str]:
        """List all available run IDs."""
        return [d.name for d in self.checkpoint_dir.iterdir() if d.is_dir()]


def save_stage_0_escape_panel(manager: CheckpointManager, escape_panel: List, cfg) -> str:
    """Save Stage 0: Viral escape mutant panel."""
    data = {
        "escape_panel": escape_panel,
        "panel_size": len(escape_panel),
        "wildtype_sequence": cfg.viral_escape.antigen_sequence,
        "epitope_residues": cfg.viral_escape.epitope_residues,
    }
    metadata = {
        "stage_number": 0,
        "description": "Viral escape mutant panel generation",
        "config": {
            "panel_size": cfg.viral_escape.panel_size,
            "max_mutations": cfg.viral_escape.max_mutations,
        }
    }
    return manager.save_stage("stage_0_escape_panel", data, metadata)


def save_stage_1_bcr_repertoire(manager: CheckpointManager, repertoire, atlas, cfg) -> str:
    """Save Stage 1: BCR repertoire and atlas."""
    sequences = repertoire.get_sequences()
    data = {
        "sequences": sequences,
        "atlas_centroid": atlas.get("centroid"),
        "atlas_covariance": atlas.get("covariance"),
        "disease_label": atlas.get("disease_label"),
        "n_sequences": len(sequences),
    }
    metadata = {
        "stage_number": 1,
        "description": "BCR repertoire loading and atlas construction",
        "config": {
            "disease_label": cfg.bcr.disease_label,
            "max_sequences": cfg.bcr.max_sequences,
        }
    }
    return manager.save_stage("stage_1_bcr_repertoire", data, metadata)


def save_stage_2_lm_scoring(manager: CheckpointManager, sequences: List[str],
                             scores: List[float], cfg) -> str:
    """Save Stage 2: Language model scoring."""
    data = {
        "sequences": sequences,
        "lm_scores": scores,
        "n_sequences": len(sequences),
    }
    metadata = {
        "stage_number": 2,
        "description": "Antibody language model scoring and generation",
        "config": {
            "model_name": cfg.lm.model_name,
            "num_sequences": cfg.lm.num_sequences,
        },
        "statistics": {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
        }
    }
    return manager.save_stage("stage_2_lm_scoring", data, metadata)


def save_stage_2a_antigen_profile(manager: CheckpointManager, affinity_matrix: np.ndarray,
                                    antibody_seqs: List[str], antigen_seqs: List[str]) -> str:
    """Save Stage 2a: Antigen-ALM binding site profile."""
    data = {
        "affinity_matrix": affinity_matrix,
        "antibody_sequences": antibody_seqs,
        "antigen_sequences": antigen_seqs,
    }
    metadata = {
        "stage_number": "2a",
        "description": "Antigen-ALM binding site profiling",
        "statistics": {
            "matrix_shape": affinity_matrix.shape,
            "mean_affinity": float(affinity_matrix.mean()),
            "std_affinity": float(affinity_matrix.std()),
        }
    }
    return manager.save_stage("stage_2a_antigen_profile", data, metadata)


def save_stage_2b_md_binding(manager: CheckpointManager, binding_matrix: np.ndarray,
                              antibody_seqs: List[str], antigen_seqs: List[str]) -> str:
    """Save Stage 2b: MD binding prediction."""
    data = {
        "binding_matrix": binding_matrix,
        "antibody_sequences": antibody_seqs,
        "antigen_sequences": antigen_seqs,
    }
    metadata = {
        "stage_number": "2b",
        "description": "MD-based binding prediction",
        "statistics": {
            "matrix_shape": binding_matrix.shape,
            "mean_binding": float(binding_matrix.mean()),
            "max_binding": float(binding_matrix.max()),
        }
    }
    return manager.save_stage("stage_2b_md_binding", data, metadata)


def save_stage_2c_alm_finetune(manager: CheckpointManager, finetuner,
                                sequences: List[str], cfg) -> str:
    """Save Stage 2c: ALM fine-tuning."""
    data = {
        "finetuner": finetuner,
        "sequences": sequences,
    }
    metadata = {
        "stage_number": "2c",
        "description": "ALM fine-tuning with MD guidance",
        "config": {
            "learning_rate": cfg.alm_finetune.learning_rate,
            "n_epochs": cfg.alm_finetune.n_epochs,
        }
    }
    return manager.save_stage("stage_2c_alm_finetune", data, metadata)


def save_stage_2d_blind_spots(manager: CheckpointManager, blind_spot_report: dict) -> str:
    """Save Stage 2d: Immune blind spot analysis."""
    data = {
        "blind_spot_report": blind_spot_report,
    }
    metadata = {
        "stage_number": "2d",
        "description": "Immune blind spot analysis",
    }
    return manager.save_stage("stage_2d_blind_spots", data, metadata)


def save_stage_2_5_pathways(manager: CheckpointManager, pathway_result: dict) -> str:
    """Save Stage 2.5: Ag-Ab structural pathways."""
    data = {
        "pathway_msm": pathway_result.get("pathway_msm"),
        "pathway_timescales": pathway_result.get("pathway_timescales"),
        "free_energy": pathway_result.get("free_energy"),
        "n_complexes": len(pathway_result.get("complex_features", [])),
    }
    metadata = {
        "stage_number": "2.5",
        "description": "Ag-Ab structural pathways and binding MSM",
        "statistics": {
            "n_complexes": len(pathway_result.get("complex_features", [])),
            "timescales": pathway_result.get("pathway_timescales", []).tolist() if isinstance(pathway_result.get("pathway_timescales"), np.ndarray) else [],
        }
    }
    return manager.save_stage("stage_2_5_pathways", data, metadata)


def save_stage_3_structure(manager: CheckpointManager, latent_embeddings: np.ndarray,
                           sequences: List[str], cfg) -> str:
    """Save Stage 3: Structural modeling (VAE/GAN)."""
    data = {
        "latent_embeddings": latent_embeddings,
        "sequences": sequences,
    }
    metadata = {
        "stage_number": 3,
        "description": "Structural modeling with VAE and GAN",
        "config": {
            "latent_dim": cfg.vae.latent_dim,
            "vae_epochs": cfg.vae.epochs,
        },
        "statistics": {
            "latent_shape": latent_embeddings.shape,
        }
    }
    return manager.save_stage("stage_3_structure", data, metadata)


def save_stage_4_msm(manager: CheckpointManager, msm, cfg) -> str:
    """Save Stage 4: MD simulation and MSM."""
    data = {
        "msm": msm,
        "timescales": msm.timescales if hasattr(msm, 'timescales') else None,
        "stationary_distribution": msm.stationary_distribution if hasattr(msm, 'stationary_distribution') else None,
    }
    metadata = {
        "stage_number": 4,
        "description": "Molecular dynamics and Markov state model",
        "config": {
            "lag_time": cfg.msm.lag_time,
            "n_states": cfg.msm.n_states,
        }
    }
    return manager.save_stage("stage_4_msm", data, metadata)


def save_stage_5_evolution(manager: CheckpointManager, evolved_sequences: List[str],
                           scores: List[float], cfg) -> str:
    """Save Stage 5: Synthetic evolution."""
    data = {
        "evolved_sequences": evolved_sequences,
        "scores": scores,
    }
    metadata = {
        "stage_number": 5,
        "description": "Synthetic repertoire evolution",
        "config": {
            "n_generations": cfg.evolution.n_generations,
            "population_size": cfg.evolution.population_size,
        },
        "statistics": {
            "n_sequences": len(evolved_sequences),
            "mean_score": float(np.mean(scores)) if scores else None,
            "max_score": float(np.max(scores)) if scores else None,
        }
    }
    return manager.save_stage("stage_5_evolution", data, metadata)


def save_stage_6_screening(manager: CheckpointManager, top_sequences: List[str],
                           top_scores: List[float], cfg) -> str:
    """Save Stage 6: Repertoire screening."""
    data = {
        "top_sequences": top_sequences,
        "top_scores": top_scores,
    }
    metadata = {
        "stage_number": 6,
        "description": "Repertoire-scale parallel screening",
        "config": {
            "top_candidates": cfg.repertoire.top_candidates,
        }
    }
    return manager.save_stage("stage_6_screening", data, metadata)


def save_stage_7_cross_reactivity(manager: CheckpointManager, coverage_matrix: np.ndarray,
                                   adaptation_summary: dict, sequences: List[str],
                                   escape_panel: List) -> str:
    """Save Stage 7: Cross-reactivity analysis."""
    data = {
        "coverage_matrix": coverage_matrix,
        "adaptation_summary": adaptation_summary,
        "sequences": sequences,
        "escape_panel": escape_panel,
    }
    metadata = {
        "stage_number": 7,
        "description": "Cross-reactivity against escape panel",
        "statistics": adaptation_summary,
    }
    return manager.save_stage("stage_7_cross_reactivity", data, metadata)


def save_stage_8_vaccine_design(manager: CheckpointManager, vaccine_candidates: List[str],
                                cfg) -> str:
    """Save Stage 8: Vaccine candidate selection."""
    data = {
        "vaccine_candidates": vaccine_candidates,
    }
    metadata = {
        "stage_number": 8,
        "description": "Broadly neutralizing vaccine candidate selection",
        "config": {
            "min_coverage_fraction": cfg.vaccine_design.min_coverage_fraction,
            "top_candidates": cfg.vaccine_design.top_candidates,
        },
        "statistics": {
            "n_candidates": len(vaccine_candidates),
        }
    }
    return manager.save_stage("stage_8_vaccine_design", data, metadata)


def save_stage_9_validation(manager: CheckpointManager, sequences: List[str],
                            predicted_scores: List[float], report_path: str) -> str:
    """Save Stage 9: Experimental validation."""
    data = {
        "sequences": sequences,
        "predicted_scores": predicted_scores,
        "report_path": report_path,
    }
    metadata = {
        "stage_number": 9,
        "description": "Experimental validation comparison",
    }
    return manager.save_stage("stage_9_validation", data, metadata)


def save_stage_10_lab_loop(manager: CheckpointManager, lab_loop_result: dict) -> str:
    """Save Stage 10: Lab-in-the-loop."""
    data = {
        "lab_loop_result": lab_loop_result,
    }
    metadata = {
        "stage_number": 10,
        "description": "Laboratory-in-the-loop iterative refinement",
        "statistics": {
            "pearson_r_before": lab_loop_result.get("pearson_r_before"),
            "pearson_r_after": lab_loop_result.get("pearson_r_after"),
            "n_suggestions": len(lab_loop_result.get("suggested_next", [])),
        }
    }
    return manager.save_stage("stage_10_lab_loop", data, metadata)
