"""
utils/helpers.py
Shared utilities: parallel evaluation, data I/O, embeddings, logging setup.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO",
                  log_file: Optional[str] = None) -> None:
    """Configure root logger with console (and optional file) handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ─── Parallel evaluation ──────────────────────────────────────────────────────

def parallel_map(fn: Callable, items: List[Any],
                 n_workers: int = 4) -> List[Any]:
    """
    Apply *fn* to every item in *items* using a process pool.

    Falls back to serial execution if n_workers <= 1 or if *fn* cannot be
    pickled (e.g. local closures), which is required by multiprocessing.Pool.
    """
    if n_workers <= 1 or len(items) == 0:
        return [fn(item) for item in items]
    try:
        pickle.dumps(fn)
    except Exception:
        logger.debug("parallel_map: fn is not picklable — falling back to serial execution.")
        return [fn(item) for item in items]
    with Pool(n_workers) as pool:
        return pool.map(fn, items)


def batched(iterable: Iterable, batch_size: int):
    """Yield successive batches of size *batch_size* from *iterable*."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ─── Data I/O ─────────────────────────────────────────────────────────────────

def save_sequences(sequences: List[str], path: str,
                   scores: Optional[List[float]] = None) -> None:
    """Save sequences (and optional scores) to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if scores is not None:
            writer.writerow(["sequence", "score"])
            for seq, sc in zip(sequences, scores):
                writer.writerow([seq, sc])
        else:
            writer.writerow(["sequence"])
            for seq in sequences:
                writer.writerow([seq])
    logger.info("Saved %d sequences → %s", len(sequences), path)


def load_sequences(path: str) -> tuple[List[str], Optional[List[float]]]:
    """
    Load sequences from CSV.

    Returns (sequences, scores) — scores is None if not present.
    """
    sequences, scores = [], []
    has_scores = False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["sequence"])
            if "score" in row:
                scores.append(float(row["score"]))
                has_scores = True
    return sequences, (scores if has_scores else None)


def save_numpy(array: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    logger.debug("Saved array %s → %s", array.shape, path)


def load_numpy(path: str) -> np.ndarray:
    return np.load(path)


def save_pickle(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ─── Sequence utilities ───────────────────────────────────────────────────────

def one_hot_encode(sequences: List[str],
                   alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
                   max_len: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a list of amino acid sequences.

    Returns
    -------
    np.ndarray of shape (N, max_len, len(alphabet))
    """
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    encoded = np.zeros((len(sequences), max_len, len(alphabet)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            if aa in aa_to_idx:
                encoded[i, j, aa_to_idx[aa]] = 1.0
    return encoded


def deduplicate(sequences: List[str],
                scores: Optional[List[float]] = None
                ) -> tuple[List[str], Optional[List[float]]]:
    """Remove duplicate sequences, keeping the highest-scored duplicate."""
    seen: Dict[str, float] = {}
    for i, seq in enumerate(sequences):
        s = scores[i] if scores else 0.0
        if seq not in seen or s > seen[seq]:
            seen[seq] = s

    if scores is None:
        return list(seen.keys()), None

    unique_seqs = list(seen.keys())
    unique_scores = [seen[seq] for seq in unique_seqs]
    return unique_seqs, unique_scores


# ─── FASTA / antigen sequence utilities ───────────────────────────────────────

_CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Spike gene coordinates in Wuhan reference NC_045512.2 (0-indexed, half-open)
_SPIKE_NT_START = 21562
_SPIKE_NT_END   = 25384   # 3822 nt → 1274 codons including stop


def _translate(nt_seq: str) -> str:
    """Translate nucleotide sequence to amino acids; stop at first stop codon."""
    aa: List[str] = []
    for i in range(0, len(nt_seq) - 2, 3):
        codon = nt_seq[i:i + 3].upper()
        if "N" in codon:
            aa.append("X")
            continue
        residue = _CODON_TABLE.get(codon, "X")
        if residue == "*":
            break
        aa.append(residue)
    return "".join(aa)


def _parse_fasta_first(fasta_path: Path) -> Optional[str]:
    """Return the concatenated nucleotide sequence of the first FASTA record."""
    seq_lines: List[str] = []
    in_first = False
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if in_first:
                    break          # second record — stop reading
                in_first = True
            elif in_first:
                seq_lines.append(line)
    return "".join(seq_lines) if seq_lines else None


def _find_spike_orf(nt: str) -> Optional[str]:
    """
    Locate the spike protein ORF in a nucleotide genome sequence by scanning
    all three reading frames for the first ORF >800 aa that contains the
    conserved RBD motif 'GVYYPDKVFR'.  Returns the amino acid sequence or
    None if not found.
    """
    _RBD_MOTIF = "GVYYPDKVFR"
    _SEARCH_WIN = (20000, min(len(nt), 26000))

    for frame in range(3):
        region = nt[_SEARCH_WIN[0]:_SEARCH_WIN[1]]
        # Collect ORFs in this frame
        orf_start: Optional[int] = None
        orf_aa: List[str] = []
        best: Optional[str] = None

        for i in range(frame, len(region) - 2, 3):
            codon = region[i:i + 3].upper()
            if "N" in codon:
                aa = "X"
            else:
                aa = _CODON_TABLE.get(codon, "X")

            if aa == "*":
                if orf_start is not None and len(orf_aa) > 800:
                    candidate = "".join(orf_aa)
                    if _RBD_MOTIF in candidate:
                        best = candidate
                        break
                orf_start = None
                orf_aa = []
            else:
                if aa == "M" and orf_start is None:
                    orf_start = i
                if orf_start is not None:
                    orf_aa.append(aa)

        if best is not None:
            return best
    return None


def load_spike_from_fasta(
    fasta_path: Optional[str] = None,
) -> str:
    """
    Extract and translate the SARS-CoV-2 spike protein from a nucleotide FASTA.

    Strategy (biopython preferred, pure-Python fallback):
      1. Parse the first record with Bio.SeqIO (or the built-in pure-Python
         parser if biopython is unavailable).
      2. Try extracting the spike gene at Wuhan reference coordinates
         (nt 21562–25384).  If the result is <100 aa or has >20 % ambiguous
         residues, fall through to step 3.
      3. Scan all three reading frames for an ORF >800 aa that contains the
         conserved RBD motif 'GVYYPDKVFR'.
      4. If all else fails, return the hardcoded RBD seed sequence.

    Falls back gracefully — the pipeline will always receive a usable
    antigen sequence regardless of genome assembly quality.
    """
    _FALLBACK = (
        "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSTSFSTFKCYGVSPTKLNDLCFTNVYADSFVIRG"
    )

    if fasta_path is None:
        fasta_path = str(
            Path(__file__).parent.parent / "data" / "SARS-CoV-2_sequences.fasta"
        )

    try:
        # Prefer biopython for robust FASTA parsing
        try:
            from Bio import SeqIO
            record = next(SeqIO.parse(fasta_path, "fasta"))
            nt = str(record.seq).upper()
        except (ImportError, StopIteration):
            nt_raw = _parse_fasta_first(Path(fasta_path))
            if not nt_raw:
                return _FALLBACK
            nt = nt_raw.upper()

        # Try reference coordinates first
        if len(nt) >= _SPIKE_NT_END:
            aa = _translate(nt[_SPIKE_NT_START:_SPIKE_NT_END])
            x_frac = aa.count("X") / max(len(aa), 1)
            if len(aa) >= 100 and x_frac <= 0.20:
                logger.info("Loaded spike from reference coords: %d aa (%.0f%% X)",
                            len(aa), 100 * x_frac)
                return aa

        # Fall back to ORF scan
        spike_aa = _find_spike_orf(nt)
        if spike_aa is not None:
            logger.info("Loaded spike via ORF scan: %d aa", len(spike_aa))
            return spike_aa

        logger.warning("Spike ORF not found; using fallback antigen sequence.")
        return _FALLBACK

    except Exception as exc:
        logger.warning("Could not load spike from FASTA (%s); using fallback.", exc)
        return _FALLBACK


def load_all_spike_sequences_from_fasta(
    fasta_path: Optional[str] = None,
    max_records: Optional[int] = None,
) -> List[str]:
    """
    Extract spike protein amino acid sequences from all records in a FASTA.

    Uses the same two-stage strategy as load_spike_from_fasta (reference
    coordinates then ORF scan) for each record independently.  Records from
    which a spike protein cannot be extracted are skipped.

    Parameters
    ----------
    fasta_path  : path to a nucleotide FASTA file.  Defaults to the bundled
                  ``data/SARS-CoV-2_sequences.fasta``.
    max_records : stop after extracting this many spike sequences (None = all).

    Returns
    -------
    List of amino acid sequence strings, one per successfully parsed record.
    """
    if fasta_path is None:
        fasta_path = str(
            Path(__file__).parent.parent / "data" / "SARS-CoV-2_sequences.fasta"
        )

    results: List[str] = []

    try:
        from Bio import SeqIO
        records = SeqIO.parse(fasta_path, "fasta")
    except ImportError:
        logger.warning("biopython not installed; falling back to pure-Python FASTA parser.")
        records = _iter_fasta_records(Path(fasta_path))  # type: ignore[assignment]

    for record in records:
        nt = str(record.seq).upper() if hasattr(record, "seq") else record.upper()

        spike_aa: Optional[str] = None

        # Try reference coordinates
        if len(nt) >= _SPIKE_NT_END:
            aa = _translate(nt[_SPIKE_NT_START:_SPIKE_NT_END])
            x_frac = aa.count("X") / max(len(aa), 1)
            if len(aa) >= 100 and x_frac <= 0.20:
                spike_aa = aa

        # ORF scan fallback
        if spike_aa is None:
            spike_aa = _find_spike_orf(nt)

        if spike_aa is not None:
            results.append(spike_aa)
            if max_records is not None and len(results) >= max_records:
                break

    logger.info(
        "load_all_spike_sequences_from_fasta: extracted %d spike sequences from %s",
        len(results), fasta_path,
    )
    return results


def _iter_fasta_records(fasta_path: Path):
    """Pure-Python FASTA record iterator (fallback when biopython unavailable)."""
    seq_lines: List[str] = []
    header = ""
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if seq_lines:
                    yield "".join(seq_lines)
                    seq_lines = []
                header = line
            else:
                seq_lines.append(line)
    if seq_lines:
        yield "".join(seq_lines)


# ─── Embedding utilities ──────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 10,
                       method: str = "kmeans") -> np.ndarray:
    """
    Cluster *embeddings* and return integer cluster assignments.

    method : 'kmeans' | 'agglomerative'
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering  # type: ignore
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        return km.fit_predict(embeddings)
    elif method == "agglomerative":
        ag = AgglomerativeClustering(n_clusters=n_clusters)
        return ag.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap",
                      n_components: int = 2) -> np.ndarray:
    """
    Reduce high-dimensional embeddings for visualisation.

    method : 'umap' | 'tsne' | 'pca'
    """
    if method == "umap":
        try:
            from umap import UMAP  # type: ignore
            return UMAP(n_components=n_components, random_state=42
                        ).fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not found — falling back to PCA.")
            method = "pca"
    if method == "tsne":
        from sklearn.manifold import TSNE  # type: ignore
        return TSNE(n_components=n_components, random_state=42
                    ).fit_transform(embeddings)
    # PCA
    from sklearn.decomposition import PCA  # type: ignore
    return PCA(n_components=n_components).fit_transform(embeddings)
