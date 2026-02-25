"""
experiments/validation.py
Experimental validation interface.

Compares computational predictions (binding scores, free energies)
against experimental measurements (SPR, BLI, ITC, ELISA).

Outputs:
  - Correlation plots (predicted vs. experimental)
  - Enrichment metrics (top-N recall, ROC-AUC)
  - Summary CSV / JSON reports
  - RP1: escape coverage heatmap + immune adaptation report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data container ───────────────────────────────────────────────────────────

class ValidationDataset:
    """
    Holds paired (predicted, experimental) binding data.

    Parameters
    ----------
    sequences    : antibody sequences
    predicted    : model-predicted scores (higher = better binder)
    experimental : measured binding values (e.g., -log10 KD, arbitrary units)
    labels       : optional human-readable labels per entry
    """

    def __init__(self, sequences: List[str],
                 predicted: List[float],
                 experimental: List[float],
                 labels: Optional[List[str]] = None):
        assert len(sequences) == len(predicted) == len(experimental)
        self.sequences = sequences
        self.predicted = np.asarray(predicted, dtype=float)
        self.experimental = np.asarray(experimental, dtype=float)
        self.labels = labels or [f"ab_{i}" for i in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    @classmethod
    def from_csv(cls, path: str) -> "ValidationDataset":
        """
        Load from CSV with columns: sequence, predicted, experimental[, label].
        """
        import csv
        sequences, predicted, experimental, labels = [], [], [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sequences.append(row["sequence"])
                predicted.append(float(row["predicted"]))
                experimental.append(float(row["experimental"]))
                labels.append(row.get("label", f"ab_{len(sequences)}"))
        return cls(sequences, predicted, experimental, labels)

    def to_csv(self, path: str) -> None:
        import csv
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "sequence", "predicted", "experimental"])
            for lab, seq, p, e in zip(self.labels, self.sequences,
                                      self.predicted, self.experimental):
                writer.writerow([lab, seq, p, e])
        logger.info("Saved validation dataset → %s", path)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def pearson_r(predicted: np.ndarray, experimental: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    from scipy.stats import pearsonr  # type: ignore
    r, _ = pearsonr(predicted, experimental)
    return float(r)


def spearman_rho(predicted: np.ndarray, experimental: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy.stats import spearmanr  # type: ignore
    rho, _ = spearmanr(predicted, experimental)
    return float(rho)


def top_n_recall(predicted: np.ndarray, experimental: np.ndarray,
                 n: int = 10, threshold_pct: float = 0.1) -> float:
    """
    Fraction of true top-*threshold_pct* binders found in top-*n* predicted.

    Parameters
    ----------
    n              : number of top predictions to evaluate
    threshold_pct  : fraction of data considered 'true positives' by experiment
    """
    n_true = max(1, int(len(experimental) * threshold_pct))
    true_top = set(np.argsort(experimental)[-n_true:])
    pred_top = set(np.argsort(predicted)[-n:])
    return len(true_top & pred_top) / n_true


def roc_auc(predicted: np.ndarray, experimental: np.ndarray,
            threshold_pct: float = 0.1) -> float:
    """
    ROC-AUC using binary labels derived from experimental top-*threshold_pct*.
    """
    from sklearn.metrics import roc_auc_score  # type: ignore
    n_true = max(1, int(len(experimental) * threshold_pct))
    labels = np.zeros(len(experimental), dtype=int)
    labels[np.argsort(experimental)[-n_true:]] = 1
    return float(roc_auc_score(labels, predicted))


def compute_all_metrics(ds: ValidationDataset) -> Dict[str, float]:
    """Return a dict of all validation metrics for a dataset."""
    metrics: Dict[str, float] = {}
    try:
        metrics["pearson_r"] = pearson_r(ds.predicted, ds.experimental)
        metrics["spearman_rho"] = spearman_rho(ds.predicted, ds.experimental)
    except ImportError:
        logger.warning("scipy not installed — skipping correlation metrics.")

    metrics["top10_recall"] = top_n_recall(
        ds.predicted, ds.experimental, n=10
    )
    metrics["top20_recall"] = top_n_recall(
        ds.predicted, ds.experimental, n=20
    )

    try:
        metrics["roc_auc"] = roc_auc(ds.predicted, ds.experimental)
    except ImportError:
        logger.warning("sklearn not installed — skipping ROC-AUC.")

    rmse = float(np.sqrt(np.mean((ds.predicted - ds.experimental) ** 2)))
    metrics["rmse"] = rmse
    return metrics


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_correlation(ds: ValidationDataset,
                     output_path: str = "experiments/output/validation_plot.png",
                     title: str = "Predicted vs. Experimental Binding") -> None:
    """
    Scatter plot: predicted score (x) vs. experimental value (y).
    Annotates with Pearson r.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ds.predicted, ds.experimental, alpha=0.6, edgecolors="k",
               linewidths=0.4, s=40)

    # Trend line
    coef = np.polyfit(ds.predicted, ds.experimental, 1)
    x_range = np.linspace(ds.predicted.min(), ds.predicted.max(), 100)
    ax.plot(x_range, np.polyval(coef, x_range), "r--", linewidth=1.5,
            label="Linear fit")

    try:
        r = pearson_r(ds.predicted, ds.experimental)
        ax.set_title(f"{title}\nPearson r = {r:.3f}")
    except Exception:
        ax.set_title(title)

    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Experimental binding")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Validation plot saved → %s", output_path)


def plot_score_distribution(sequences: List[str], scores: List[float],
                             output_path: str = "experiments/output/score_dist.png",
                             title: str = "Candidate Score Distribution") -> None:
    """Histogram of predicted binding scores for a candidate set."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(scores), color="red", linestyle="--",
               label=f"Mean = {np.mean(scores):.2f}")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Score distribution plot → %s", output_path)


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(ds: ValidationDataset,
                    output_dir: str = "experiments/output") -> str:
    """
    Compute all metrics, generate plots, and write a JSON summary report.

    Returns path to the report JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_all_metrics(ds)
    plot_correlation(ds, str(out / "correlation_plot.png"))
    plot_score_distribution(
        ds.sequences, ds.predicted.tolist(),
        str(out / "score_distribution.png")
    )
    ds.to_csv(str(out / "validation_data.csv"))

    report = {
        "n_samples": len(ds),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }

    import json
    report_path = str(out / "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report written → %s", report_path)
    for k, v in metrics.items():
        logger.info("  %-20s %.4f", k, v)

    return report_path


# ─── RP1: Escape Coverage & Vaccine Design Metrics ────────────────────────────

def escape_coverage_score(
    coverage_matrix: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Per-antibody escape panel coverage fraction.

    Parameters
    ----------
    coverage_matrix : (n_antibodies, n_variants) binding probability matrix
    threshold       : minimum binding score to count as 'covered'

    Returns
    -------
    np.ndarray of shape (n_antibodies,) — fraction of variants each antibody covers
    """
    return (coverage_matrix >= threshold).mean(axis=1)


def breadth_score(coverage_matrix: np.ndarray, threshold: float = 0.5) -> float:
    """
    Mean escape panel coverage across all antibodies in the matrix.

    Returns
    -------
    float in [0, 1] — higher = immune population covers more escape variants
    """
    return float(escape_coverage_score(coverage_matrix, threshold).mean())


def plot_cross_reactivity_heatmap(
    antibody_sequences: List[str],
    escape_panel,        # List[EscapeMutant]
    coverage_matrix: np.ndarray,
    output_path: str = "experiments/output/cross_reactivity_heatmap.png",
    title: str = "Antibody × Escape Variant Binding (RP1)",
    max_display: int = 40,
) -> None:
    """
    Heatmap of antibody (rows) vs. escape variant (columns) binding probabilities.

    Colour scale: 0 (no binding / immune escape) → 1 (strong binding).
    Rows and columns are ordered by hierarchical clustering for readability.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not installed — skipping heatmap.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Subsample for readability if there are many antibodies / variants
    n_ab = min(len(antibody_sequences), max_display)
    n_var = min(int(coverage_matrix.shape[1]), max_display)
    mat = coverage_matrix[:n_ab, :n_var]

    # X-axis labels: short mutation string of each escape variant
    x_labels = [
        ep.mutation_string() if hasattr(ep, "mutation_string") else f"V{i}"
        for i, ep in enumerate(escape_panel[:n_var])
    ]
    # Y-axis labels: abbreviated antibody sequences
    y_labels = [
        f"Ab{i+1}({seq[:6]}…)" for i, seq in enumerate(antibody_sequences[:n_ab])
    ]

    fig_h = max(4.0, n_ab * 0.3)
    fig_w = max(6.0, n_var * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.2,
        linecolor="gray",
        cbar_kws={"label": "Binding probability", "shrink": 0.7},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Escape variant (mutation)", fontsize=9)
    ax.set_ylabel("Antibody candidate", fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Cross-reactivity heatmap → %s", output_path)


def generate_escape_report(
    antibody_sequences: List[str],
    escape_panel,           # List[EscapeMutant]
    coverage_matrix: np.ndarray,
    adaptation_summary: Dict,
    output_dir: str = "experiments/output",
) -> str:
    """
    Generate the RP1 escape coverage report:
      - Cross-reactivity heatmap
      - Per-antibody coverage CSV
      - JSON summary with breadth score and immune adaptation stats

    Returns path to the JSON report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Heatmap
    plot_cross_reactivity_heatmap(
        antibody_sequences,
        escape_panel,
        coverage_matrix,
        str(out / "cross_reactivity_heatmap.png"),
    )

    # 2. Per-antibody coverage CSV
    cov_scores = escape_coverage_score(coverage_matrix)
    mean_binding = coverage_matrix.mean(axis=1)
    cov_path = str(out / "escape_coverage.csv")
    Path(cov_path).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(cov_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["antibody_id", "sequence_prefix",
                         "coverage_fraction", "mean_binding_score"])
        for i, seq in enumerate(antibody_sequences):
            writer.writerow([
                f"ab_{i+1}", seq[:12] + "…",
                round(float(cov_scores[i]), 4),
                round(float(mean_binding[i]), 4),
            ])
    logger.info("Escape coverage CSV → %s", cov_path)

    # 3. JSON summary
    n_variants = len(escape_panel)
    report: Dict[str, Any] = {
        "rp1_goal": "Predicting Antibody Responses to Viral Escape Mutants",
        "n_antibodies_evaluated": len(antibody_sequences),
        "n_escape_variants": n_variants,
        "breadth_score": round(breadth_score(coverage_matrix), 4),
        "fraction_broadly_neutralising": round(
            float(adaptation_summary.get("fraction_broadly_neutralising", 0.0)), 4
        ),
        "mean_panel_coverage": round(
            float(adaptation_summary.get("mean_panel_coverage", 0.0)), 4
        ),
        "most_vulnerable_variant_indices": adaptation_summary.get(
            "most_vulnerable_variants", []
        ),
        "most_vulnerable_variants": [
            escape_panel[i].mutation_string()
            for i in adaptation_summary.get("most_vulnerable_variants", [])
            if i < n_variants
        ],
        "top_broadly_neutralising_antibodies": [
            f"ab_{i+1}" for i in
            adaptation_summary.get("most_resistant_antibody_indices", [])
        ],
    }

    report_path = str(out / "escape_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("RP1 escape report → %s", report_path)
    logger.info(
        "  breadth_score=%.3f  broadly_neutralising=%.1f%%  panel_coverage=%.3f",
        report["breadth_score"],
        report["fraction_broadly_neutralising"] * 100,
        report["mean_panel_coverage"],
    )
    return report_path
