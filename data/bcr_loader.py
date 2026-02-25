"""
data/bcr_loader.py

BCR (B-Cell Receptor) repertoire data loading.

Supports two data sources
--------------------------
1. OAS (Observed Antibody Space) — public bulk-download CSV files.
   Data portal : https://opig.stats.ox.ac.uk/webapps/oas/
   Download    : use the OAS web interface to select units by species,
                 chain, isotype, and disease, then bulk-download as CSV.
   Format      : AIRR-Community standard columns including sequence_id,
                 sequence_aa, v_call, j_call, junction_aa, isotype.

2. Private BCR data — researcher-supplied CSV or FASTA files.
   CSV  : must include an amino-acid sequence column (default: 'sequence_aa').
          Optional columns: sequence_id, cdr3 / junction_aa, v_gene, isotype.
   FASTA: standard single-letter AA FASTA (biopython SeqIO).

Usage
-----
    from RP1_antibody_pipeline.data.bcr_loader import load_repertoire

    # Public OAS data
    repertoire = load_repertoire(
        source="oas",
        path="data/oas/",
        disease_label="COVID-19",
        max_sequences=5000,
    )

    # Private CSV
    repertoire = load_repertoire(
        source="csv",
        path="data/private_bcr.csv",
        disease_label="influenza",
        sequence_col="vh_sequence",
    )
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class BCRSequence:
    """
    A single BCR / antibody sequence with metadata.

    Attributes
    ----------
    sequence_id  : unique identifier (from OAS or assigned if absent).
    sequence_aa  : full-length amino acid sequence (VH or VL).
    v_gene       : V-gene call (e.g. 'IGHV1-2*02').
    j_gene       : J-gene call (e.g. 'IGHJ4*02').
    junction_aa  : CDR3 amino acid sequence.
    isotype      : antibody isotype (e.g. 'IgG', 'IgM').
    disease      : disease label attached at load time.
    source       : 'oas' | 'private'.
    """
    sequence_id: str
    sequence_aa: str
    v_gene: str = ""
    j_gene: str = ""
    junction_aa: str = ""
    isotype: str = ""
    disease: str = ""
    source: str = ""


@dataclass
class BCRRepertoire:
    """
    A collection of BCR sequences representing a repertoire.

    Attributes
    ----------
    sequences     : list of BCRSequence objects.
    disease_label : disease context for this repertoire.
    """
    sequences: List[BCRSequence]
    disease_label: str
    n_sequences: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_sequences = len(self.sequences)

    def get_sequences(self) -> List[str]:
        """Return non-empty full-length AA sequences."""
        return [s.sequence_aa for s in self.sequences if s.sequence_aa]

    def get_cdr3s(self) -> List[str]:
        """Return non-empty CDR3 / junction_aa sequences."""
        return [s.junction_aa for s in self.sequences if s.junction_aa]

    def filter_by_length(self, min_len: int = 50,
                          max_len: int = 200) -> "BCRRepertoire":
        """Return a new repertoire keeping only sequences within length bounds."""
        kept = [s for s in self.sequences
                if min_len <= len(s.sequence_aa) <= max_len]
        return BCRRepertoire(sequences=kept, disease_label=self.disease_label)

    def summary(self) -> Dict:
        seqs = self.get_sequences()
        lengths = [len(s) for s in seqs]
        return {
            "disease": self.disease_label,
            "n_sequences": self.n_sequences,
            "mean_length": sum(lengths) / len(lengths) if lengths else 0,
            "isotypes": list({s.isotype for s in self.sequences if s.isotype}),
        }


# ─── OAS loader ───────────────────────────────────────────────────────────────

class OASLoader:
    """
    Loads BCR repertoire data from OAS bulk-download CSV files.

    OAS CSV format (AIRR-Community standard):
        sequence_id, sequence_aa, v_call, j_call, junction_aa, isotype, …

    Each OAS 'unit' is a single CSV file corresponding to one study / sample.
    Point the loader at a directory containing one or more such CSVs.

    Parameters
    ----------
    disease_label : label to attach to all loaded sequences.
    """

    # Column name mappings (OAS uses AIRR-C standard names)
    _SEQ_AA  = "sequence_aa"
    _SEQ_ID  = "sequence_id"
    _V_CALL  = "v_call"
    _J_CALL  = "j_call"
    _CDR3    = "junction_aa"
    _ISOTYPE = "isotype"

    def __init__(self, disease_label: str = "unknown"):
        self.disease_label = disease_label

    def load(
        self,
        csv_path: str,
        max_sequences: Optional[int] = None,
    ) -> BCRRepertoire:
        """
        Load one OAS unit CSV file.

        Parameters
        ----------
        csv_path      : path to the OAS CSV file.
        max_sequences : stop after this many rows (None = load all).
        """
        sequences = list(self._iter_csv(csv_path, max_sequences))
        logger.info("OAS: loaded %d sequences from %s", len(sequences), csv_path)
        return BCRRepertoire(sequences=sequences,
                             disease_label=self.disease_label)

    def load_multiple(
        self,
        csv_paths: List[str],
        max_per_file: Optional[int] = None,
    ) -> BCRRepertoire:
        """
        Load and merge multiple OAS unit CSV files.

        Parameters
        ----------
        csv_paths    : list of paths to OAS CSV files.
        max_per_file : cap per file (None = no cap).
        """
        all_seqs: List[BCRSequence] = []
        for path in csv_paths:
            seqs = list(self._iter_csv(path, max_per_file))
            all_seqs.extend(seqs)
        logger.info("OAS: loaded %d sequences total from %d files",
                    len(all_seqs), len(csv_paths))
        return BCRRepertoire(sequences=all_seqs,
                             disease_label=self.disease_label)

    def load_directory(
        self,
        directory: str,
        max_per_file: Optional[int] = None,
    ) -> BCRRepertoire:
        """
        Load all CSV files in *directory* as OAS units.

        Parameters
        ----------
        directory    : directory containing OAS bulk-download CSVs.
        max_per_file : cap per file (None = no cap).
        """
        csv_files = sorted(Path(directory).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in OAS directory: %s", directory)
            return BCRRepertoire(sequences=[], disease_label=self.disease_label)
        paths = [str(f) for f in csv_files]
        return self.load_multiple(paths, max_per_file=max_per_file)

    # ── Private ────────────────────────────────────────────────────────────

    def _iter_csv(
        self,
        csv_path: str,
        max_rows: Optional[int],
    ) -> Iterator[BCRSequence]:
        """Yield BCRSequence objects from one OAS CSV file."""
        count = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            # OAS CSVs sometimes have a metadata header line before the column row
            first_line = fh.readline()
            if not first_line.startswith(self._SEQ_ID) and \
               not first_line.startswith(self._SEQ_AA):
                # Likely a metadata line — skip it and re-read
                pass
            else:
                fh.seek(0)

            reader = csv.DictReader(fh)
            for row in reader:
                seq_aa = row.get(self._SEQ_AA, "").strip()
                if not seq_aa:
                    continue
                seq_id = row.get(self._SEQ_ID, f"oas_{count}").strip()
                yield BCRSequence(
                    sequence_id=seq_id,
                    sequence_aa=seq_aa,
                    v_gene=row.get(self._V_CALL, "").strip(),
                    j_gene=row.get(self._J_CALL, "").strip(),
                    junction_aa=row.get(self._CDR3, "").strip(),
                    isotype=row.get(self._ISOTYPE, "").strip(),
                    disease=self.disease_label,
                    source="oas",
                )
                count += 1
                if max_rows is not None and count >= max_rows:
                    break


# ─── Private BCR loader ───────────────────────────────────────────────────────

class PrivateBCRLoader:
    """
    Loads BCR sequences from private researcher-supplied files.

    Supports CSV (with configurable column names) and FASTA (biopython).

    Parameters
    ----------
    disease_label : label to attach to all loaded sequences.
    sequence_col  : name of the amino-acid sequence column in CSV input.
    id_col        : name of the sequence ID column (optional).
    cdr3_col      : name of the CDR3 column (optional).
    v_gene_col    : name of the V-gene column (optional).
    isotype_col   : name of the isotype column (optional).
    """

    def __init__(
        self,
        disease_label: str = "unknown",
        sequence_col: str = "sequence_aa",
        id_col: str = "sequence_id",
        cdr3_col: str = "junction_aa",
        v_gene_col: str = "v_gene",
        isotype_col: str = "isotype",
    ):
        self.disease_label = disease_label
        self.sequence_col = sequence_col
        self.id_col = id_col
        self.cdr3_col = cdr3_col
        self.v_gene_col = v_gene_col
        self.isotype_col = isotype_col

    def load_csv(
        self,
        csv_path: str,
        max_sequences: Optional[int] = None,
    ) -> BCRRepertoire:
        """
        Load BCR sequences from a private CSV file.

        The file must contain a column named *sequence_col* with amino acid
        sequences.  All other columns are optional.
        """
        sequences: List[BCRSequence] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                seq_aa = row.get(self.sequence_col, "").strip()
                if not seq_aa:
                    continue
                sequences.append(BCRSequence(
                    sequence_id=row.get(self.id_col, f"private_{i}").strip(),
                    sequence_aa=seq_aa,
                    v_gene=row.get(self.v_gene_col, "").strip(),
                    junction_aa=row.get(self.cdr3_col, "").strip(),
                    isotype=row.get(self.isotype_col, "").strip(),
                    disease=self.disease_label,
                    source="private",
                ))
                if max_sequences is not None and len(sequences) >= max_sequences:
                    break

        logger.info("Private CSV: loaded %d sequences from %s",
                    len(sequences), csv_path)
        return BCRRepertoire(sequences=sequences,
                             disease_label=self.disease_label)

    def load_fasta(
        self,
        fasta_path: str,
        max_sequences: Optional[int] = None,
    ) -> BCRRepertoire:
        """
        Load BCR sequences from a FASTA file (amino acid sequences).

        Requires biopython.  The FASTA record ID becomes the sequence_id.
        """
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("biopython required: pip install biopython")

        sequences: List[BCRSequence] = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_aa = str(record.seq).upper().replace("*", "")
            if not seq_aa:
                continue
            sequences.append(BCRSequence(
                sequence_id=record.id,
                sequence_aa=seq_aa,
                disease=self.disease_label,
                source="private",
            ))
            if max_sequences is not None and len(sequences) >= max_sequences:
                break

        logger.info("Private FASTA: loaded %d sequences from %s",
                    len(sequences), fasta_path)
        return BCRRepertoire(sequences=sequences,
                             disease_label=self.disease_label)


# ─── Factory ──────────────────────────────────────────────────────────────────

def load_repertoire(
    source: str,
    path: str,
    disease_label: str = "unknown",
    max_sequences: Optional[int] = None,
    sequence_col: str = "sequence_aa",
) -> BCRRepertoire:
    """
    Load a BCR repertoire from a file or directory.

    Parameters
    ----------
    source        : 'oas'  — OAS bulk-download directory or single CSV.
                    'csv'  — private CSV file.
                    'fasta'— private FASTA file.
    path          : file or directory path.
    disease_label : label attached to all loaded sequences.
    max_sequences : maximum number of sequences to load (None = all).
    sequence_col  : for CSV sources — name of the AA sequence column.

    Returns
    -------
    BCRRepertoire
    """
    source = source.lower()

    if source == "oas":
        loader = OASLoader(disease_label=disease_label)
        p = Path(path)
        if p.is_dir():
            return loader.load_directory(str(p), max_per_file=max_sequences)
        return loader.load(str(p), max_sequences=max_sequences)

    elif source == "csv":
        loader = PrivateBCRLoader(
            disease_label=disease_label,
            sequence_col=sequence_col,
        )
        return loader.load_csv(path, max_sequences=max_sequences)

    elif source == "fasta":
        loader = PrivateBCRLoader(disease_label=disease_label)
        return loader.load_fasta(path, max_sequences=max_sequences)

    else:
        raise ValueError(
            f"Unknown source '{source}'. Choose from: 'oas', 'csv', 'fasta'."
        )
