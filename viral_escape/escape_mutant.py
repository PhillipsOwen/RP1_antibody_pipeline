"""
viral_escape/escape_mutant.py

Viral escape mutant generation and escape potential scoring.

Models how viruses accumulate mutations in antibody-binding epitopes to evade
immune detection — the antigen side of RP1: Predicting Antibody Responses to
Viral Escape Mutants.

Workflow
--------
1. Provide a wildtype antigen sequence (e.g., spike RBD, HA head domain).
2. Define the epitope residues — positions the antibody physically contacts.
3. Call generate_panel() to produce single/double/triple mutants at those sites.
4. Each mutant is scored for escape potential: fraction of epitope positions
   that were disrupted.
5. Use generate_known_variants() to model real-world variants (Alpha, Delta,
   Omicron, etc.) from published mutation tables.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class EscapeMutant:
    """
    A viral antigen variant with one or more mutations relative to wildtype.

    Attributes
    ----------
    wildtype_sequence  : full wildtype antigen amino acid sequence
    mutant_sequence    : full mutated antigen sequence
    mutations          : list of (position, wt_aa, mut_aa) — 0-indexed
    epitope_mutations  : number of mutations that fall inside the epitope
    escape_score       : fraction of epitope positions disrupted (0–1).
                         1.0 → every epitope residue changed (full escape);
                         0.0 → no epitope residues changed (no escape).
    """
    wildtype_sequence: str
    mutant_sequence: str
    mutations: List[Tuple[int, str, str]] = field(default_factory=list)
    epitope_mutations: int = 0
    escape_score: float = 0.0

    @property
    def n_mutations(self) -> int:
        return len(self.mutations)

    def mutation_string(self) -> str:
        """Human-readable label, e.g. 'K417N/E484K/N501Y'."""
        return "/".join(
            f"{wt}{pos + 1}{mut}" for pos, wt, mut in self.mutations
        ) or "WT"


# ─── Generator ────────────────────────────────────────────────────────────────

class EscapeMutantGenerator:
    """
    Generates a panel of viral escape mutants by mutating epitope residues.

    The panel covers:
      - All single-residue substitutions at epitope positions (up to panel_size)
      - Randomly sampled double / triple mutants to fill remaining slots

    Parameters
    ----------
    wildtype_sequence  : str
        Full wildtype antigen amino acid sequence.
    epitope_residues   : List[int]
        0-indexed residue positions contacted by the antibody of interest.
        If empty, the generator treats the full sequence as the epitope.
    panel_size         : int
        Maximum number of escape mutants to return from generate_panel().
    max_mutations      : int
        Maximum simultaneous point mutations per variant (1–3 recommended).
    random_seed        : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        wildtype_sequence: str,
        epitope_residues: Optional[List[int]] = None,
        panel_size: int = 50,
        max_mutations: int = 3,
        random_seed: int = 42,
        lm=None,
    ):
        self.wildtype = wildtype_sequence
        self.epitope = list(epitope_residues) if epitope_residues else list(
            range(len(wildtype_sequence))
        )
        self.panel_size = panel_size
        self.max_mutations = max_mutations
        self.lm = lm  # optional AntibodyLM for PLM-guided escape (SA2)
        random.seed(random_seed)
        np.random.seed(random_seed)

    # ── Panel generation ──────────────────────────────────────────────────────

    def generate_panel(self) -> List[EscapeMutant]:
        """
        Generate a diverse panel of escape mutants, sorted by escape_score
        descending (highest escape potential first).

        Strategy
        --------
        1. Enumerate all single-position substitutions at epitope sites.
        2. Randomly sample double/triple mutants until panel_size is reached.

        Returns
        -------
        List[EscapeMutant]
        """
        panel: List[EscapeMutant] = []

        # 1. Single-mutant sweep across epitope positions
        for pos in self.epitope:
            wt_aa = self.wildtype[pos] if pos < len(self.wildtype) else None
            if wt_aa is None:
                continue
            for mut_aa in AA_ALPHABET:
                if mut_aa == wt_aa:
                    continue
                mutant = self._apply_mutations([(pos, wt_aa, mut_aa)])
                panel.append(mutant)
                if len(panel) >= self.panel_size:
                    break
            if len(panel) >= self.panel_size:
                break

        # 2. Multi-mutant fill
        seen_seqs = {m.mutant_sequence for m in panel}
        attempts = 0
        while len(panel) < self.panel_size and attempts < self.panel_size * 20:
            attempts += 1
            n_muts = random.randint(2, min(self.max_mutations, len(self.epitope)))
            positions = random.sample(
                [p for p in self.epitope if p < len(self.wildtype)],
                k=min(n_muts, len(self.epitope)),
            )
            muts = []
            for pos in positions:
                wt_aa = self.wildtype[pos]
                mut_aa = random.choice([aa for aa in AA_ALPHABET if aa != wt_aa])
                muts.append((pos, wt_aa, mut_aa))
            mutant = self._apply_mutations(muts)
            if mutant.mutant_sequence not in seen_seqs:
                panel.append(mutant)
                seen_seqs.add(mutant.mutant_sequence)

        panel.sort(key=lambda m: m.escape_score, reverse=True)
        top_score = panel[0].escape_score if panel else 0.0
        logger.info(
            "Generated %d escape mutants (max escape_score=%.3f)",
            len(panel), top_score,
        )
        return panel

    def generate_known_variants(
        self,
        variant_mutations: List[List[Tuple[int, str, str]]],
    ) -> List[EscapeMutant]:
        """
        Build EscapeMutants from pre-defined mutation sets (known viral variants).

        Use this to model real-world strains (e.g., Alpha B.1.1.7, Omicron BA.1)
        using published mutation tables.

        Parameters
        ----------
        variant_mutations : list of mutation lists, each entry is:
            [(0-indexed position, wildtype_aa, mutant_aa), ...]

        Returns
        -------
        List[EscapeMutant] in the same order as *variant_mutations*.
        """
        variants = [self._apply_mutations(muts) for muts in variant_mutations]
        logger.info("Created %d known-variant escape mutants.", len(variants))
        return variants

    # ── PLM-guided escape panel generation (SA2) ────────────────────────────────

    def generate_lm_guided_panel(
        self, n_samples: int = 50, top_k: int = 10
    ) -> "List[EscapeMutant]":
        """
        Generate escape mutants guided by a protein language model (SA2).

        SA2 requires PLM-driven forward prediction of pathogen variants that
        evade the existing BCR repertoire — not just exhaustive combinatorial
        mutagenesis.  This method uses the LM's masked-token predictions at
        epitope positions to propose amino acid substitutions that are both:
          1. Structurally plausible (high LM log-probability for the mutant).
          2. Located at epitope residues (maximising immune escape potential).

        Falls back to generate_panel() if no LM is configured.

        Parameters
        ----------
        n_samples : number of LM-guided mutant sequences to generate.
        top_k     : top-k sampling width passed to lm.generate_mutations().

        Returns
        -------
        List[EscapeMutant] sorted by escape_score descending.
        """
        if self.lm is None:
            logger.warning(
                "No LM configured for PLM-guided escape generation — "
                "falling back to combinatorial generate_panel()."
            )
            return self.generate_panel()

        panel: "List[EscapeMutant]" = []
        seen_seqs = set()

        # Use LM to generate high-fitness mutations; restrict sampling to
        # epitope positions by selecting only those mutation proposals that
        # fall at epitope residues.
        epitope_set = set(self.epitope)
        candidates = self.lm.generate_mutations(
            seed_sequence=self.wildtype,
            n_mutations=min(self.max_mutations, len(self.epitope)),
            n_samples=n_samples * 4,  # oversample; filter to epitope hits
            top_k=top_k,
        )

        for mutant_seq, _lm_score in candidates:
            if mutant_seq in seen_seqs or mutant_seq == self.wildtype:
                continue
            # Identify which positions changed
            muts = [
                (i, self.wildtype[i], mutant_seq[i])
                for i in range(min(len(self.wildtype), len(mutant_seq)))
                if mutant_seq[i] != self.wildtype[i]
            ]
            # Keep only if at least one mutation is in the epitope
            if not any(pos in epitope_set for pos, _, _ in muts):
                continue
            escape = self._apply_mutations(
                [(pos, wt, mut) for pos, wt, mut in muts if pos in epitope_set]
            )
            panel.append(escape)
            seen_seqs.add(mutant_seq)
            if len(panel) >= self.panel_size:
                break

        # Fill remaining slots with combinatorial panel if LM did not produce enough
        if len(panel) < self.panel_size:
            combo = self.generate_panel()
            for m in combo:
                if m.mutant_sequence not in seen_seqs and len(panel) < self.panel_size:
                    panel.append(m)
                    seen_seqs.add(m.mutant_sequence)

        panel.sort(key=lambda m: m.escape_score, reverse=True)
        logger.info(
            "LM-guided escape panel: %d mutants generated (top escape_score=%.3f)",
            len(panel), panel[0].escape_score if panel else 0.0,
        )
        return panel

    # ── Escape scoring ────────────────────────────────────────────────────────

    def score_escape(self, mutant_sequence: str) -> float:
        """
        Estimate escape score: fraction of epitope positions that changed.

        Score = 0  → no epitope residues mutated (antibody likely still binds)
        Score = 1  → every epitope residue mutated (full immune escape)
        """
        if not self.epitope:
            return 0.0
        changed = sum(
            1 for pos in self.epitope
            if pos < len(mutant_sequence)
            and pos < len(self.wildtype)
            and mutant_sequence[pos] != self.wildtype[pos]
        )
        return changed / len(self.epitope)

    def mutation_hotspots(self, n_top: int = 10) -> List[Tuple[int, float]]:
        """
        Rank epitope positions by their likelihood of being escape hotspots.

        Heuristic: positions where substitution changes the charge/polarity
        class are harder for antibodies to tolerate → higher hotspot score.

        Returns
        -------
        List of (position, hotspot_score) sorted descending.
        """
        scores = []
        for pos in self.epitope:
            if pos >= len(self.wildtype):
                continue
            wt_aa = self.wildtype[pos]
            # Count substitutions that change the physicochemical class
            distinct = sum(
                1 for aa in AA_ALPHABET
                if aa != wt_aa
                and _charge_group(aa) != _charge_group(wt_aa)
            )
            scores.append((pos, distinct / len(AA_ALPHABET)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_top]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_mutations(
        self, mutations: List[Tuple[int, str, str]]
    ) -> EscapeMutant:
        """Apply (pos, wt, mut) mutations to the wildtype and return EscapeMutant."""
        seq = list(self.wildtype)
        applied: List[Tuple[int, str, str]] = []
        epitope_set = set(self.epitope)

        for pos, wt_aa, mut_aa in mutations:
            if pos < len(seq) and seq[pos] == wt_aa:
                seq[pos] = mut_aa
                applied.append((pos, wt_aa, mut_aa))

        mutant_seq = "".join(seq)
        epitope_muts = sum(1 for pos, _, _ in applied if pos in epitope_set)
        escape = self.score_escape(mutant_seq)

        return EscapeMutant(
            wildtype_sequence=self.wildtype,
            mutant_sequence=mutant_seq,
            mutations=applied,
            epitope_mutations=epitope_muts,
            escape_score=escape,
        )


# ─── Physicochemical grouping ─────────────────────────────────────────────────

def _charge_group(aa: str) -> str:
    """Coarse charge/polarity class for amino acid *aa*."""
    if aa in "KRH":
        return "positive"
    if aa in "DE":
        return "negative"
    if aa in "NQSTY":
        return "polar"
    return "nonpolar"