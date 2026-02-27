"""
synthetic_evolution/evolution.py
Synthetic repertoire evolution — simulates antibody affinity maturation.

Models:
  - Random somatic hypermutation
  - LM-guided (scored) mutation
  - Selection based on predicted binding score
  - Multi-generation evolutionary loop
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

# CDR loop positions (approximate, based on IMGT numbering for VH domain)
# These are residue index ranges: CDR-H1, CDR-H2, CDR-H3
CDR_REGIONS = [(26, 35), (50, 66), (95, 102)]

# SHM hotspot motif positions (SA2).
# AID (activation-induced deaminase) preferentially deaminates cytosines in
# WRCY (W=A/T, R=A/G, C=C, Y=C/T) and its complement RGYW motifs.
# These are encoded here as a per-position relative elevation factor.
# The lookup maps each residue to its SHM mutability weight (normalised):
#   CDR positions already receive elevation via CDR_REGIONS;
#   J-gene junction positions (approx. 90-110 for VH) receive an additional boost.
#   Framework residues receive a baseline weight of 1.0.
def _shm_position_weights(sequence_length: int) -> np.ndarray:
    """
    Return per-position SHM mutability weights (SA2).

    Weights reflect:
    - CDR loop elevation (5x) -- same bias as cdr_focused_mutate
    - J-gene junction elevation (3x) -- positions 90-110 in VH
    - Baseline framework weight (1.0)

    The weights are NOT normalised to sum to 1; they are used as per-position
    mutation probability multipliers relative to the base mutation_rate.
    """
    weights = np.ones(sequence_length, dtype=np.float32)
    # CDR loops
    for start, end in CDR_REGIONS:
        weights[start:min(end, sequence_length)] = 5.0
    # J-gene junction region (approximate VH positions 90-110)
    j_start, j_end = 90, min(110, sequence_length)
    weights[j_start:j_end] = np.maximum(weights[j_start:j_end], 3.0)
    return weights


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Antibody:
    sequence: str
    score: float = 0.0          # predicted binding score (higher = better)
    generation: int = 0
    parent: Optional[str] = None


@dataclass
class Generation:
    index: int
    population: List[Antibody]
    best_score: float = 0.0
    mean_score: float = 0.0

    def __post_init__(self):
        scores = [ab.score for ab in self.population]
        self.best_score = max(scores) if scores else 0.0
        self.mean_score = float(np.mean(scores)) if scores else 0.0


# ─── Mutation operators ───────────────────────────────────────────────────────

def random_mutate(sequence: str, mutation_rate: float = 0.05) -> str:
    """Apply random point mutations across the whole sequence."""
    seq = list(sequence)
    for i in range(len(seq)):
        if random.random() < mutation_rate:
            seq[i] = random.choice(AA_ALPHABET)
    return "".join(seq)


def cdr_focused_mutate(sequence: str, mutation_rate: float = 0.1,
                       cdr_bias: float = 5.0) -> str:
    """
    Mutation with higher rate at CDR loop positions.

    *cdr_bias* multiplies the mutation probability inside CDR regions.
    """
    seq = list(sequence)
    for i in range(len(seq)):
        in_cdr = any(start <= i < end for start, end in CDR_REGIONS
                     if end <= len(seq))
        rate = mutation_rate * cdr_bias if in_cdr else mutation_rate
        if random.random() < rate:
            seq[i] = random.choice(AA_ALPHABET)
    return "".join(seq)


def shm_hotspot_mutate(sequence: str, mutation_rate: float = 0.05) -> str:
    """
    Position-dependent SHM mutation using AID hotspot weights (SA2).

    SA2 requires mutation rates parameterised by known somatic hypermutation
    (SHM) hotspot positions (WRCY/RGYW motifs) rather than a uniform rate.
    This function applies the per-position weight vector from
    _shm_position_weights() so that CDR loops and J-gene junction positions
    mutate at elevated rates consistent with known SHM biology.

    Per-position mutation probability = mutation_rate * weight[i]
    where weight[i] is 5� in CDRs, 3� at the J-junction, and 1� elsewhere.

    Parameters
    ----------
    sequence     : antibody amino acid sequence.
    mutation_rate: base per-residue mutation probability (before weighting).

    Returns
    -------
    Mutated sequence string.
    """
    seq = list(sequence)
    weights = _shm_position_weights(len(seq))
    for i in range(len(seq)):
        if random.random() < mutation_rate * float(weights[i]):
            seq[i] = random.choice(AA_ALPHABET)
    return "".join(seq)


def guided_mutate(sequence: str, scorer: Callable[[List[str]], List[float]],
                  n_candidates: int = 20, n_positions: int = 3) -> str:
    """
    Greedy single-position mutation guided by a scoring function.

    Tries *n_candidates* random single-point mutations, keeps the best.
    """
    best_seq, best_score = sequence, scorer([sequence])[0]
    positions = random.sample(range(len(sequence)), k=min(n_positions, len(sequence)))
    for pos in positions:
        candidates = []
        for aa in random.sample(AA_ALPHABET, k=min(n_candidates, len(AA_ALPHABET))):
            mut = list(sequence)
            mut[pos] = aa
            candidates.append("".join(mut))
        scores = scorer(candidates)
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_seq, best_score = candidates[idx], scores[idx]
    return best_seq


# ─── Selection ────────────────────────────────────────────────────────────────

def tournament_select(population: List[Antibody], n_select: int,
                      tournament_size: int = 3) -> List[Antibody]:
    """Tournament selection — returns *n_select* winners."""
    selected = []
    for _ in range(n_select):
        contestants = random.sample(population, k=min(tournament_size,
                                                       len(population)))
        winner = max(contestants, key=lambda ab: ab.score)
        selected.append(winner)
    return selected


def top_k_select(population: List[Antibody], k: int) -> List[Antibody]:
    """Truncation selection — return top *k* individuals by score."""
    return sorted(population, key=lambda ab: ab.score, reverse=True)[:k]


# ─── Evolutionary engine ──────────────────────────────────────────────────────

class RepertoireEvolver:
    """
    Simulates rounds of somatic hypermutation + selection (affinity maturation).

    Parameters
    ----------
    scorer           : callable that maps List[str] → List[float].
                       Typically an antibody LM score or binding energy predictor.
    mutation_fn      : mutation operator ('random', 'cdr', 'guided')
    mutation_rate    : per-residue mutation probability
    n_generations    : total evolutionary rounds
    population_size  : sequences per generation
    top_fraction     : fraction of parents selected each round
    """

    def __init__(self,
                 scorer: Callable[[List[str]], List[float]],
                 mutation_fn: str = "cdr",
                 mutation_rate: float = 0.05,
                 n_generations: int = 10,
                 population_size: int = 500,
                 top_fraction: float = 0.2):
        self.scorer = scorer
        self.mutation_fn = mutation_fn
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population_size = population_size
        self.top_fraction = top_fraction

    def _mutate(self, seq: str) -> str:
        if self.mutation_fn == "random":
            return random_mutate(seq, self.mutation_rate)
        elif self.mutation_fn == "cdr":
            return cdr_focused_mutate(seq, self.mutation_rate)
        elif self.mutation_fn == "guided":
            return guided_mutate(seq, self.scorer)
        elif self.mutation_fn == "shm":
            return shm_hotspot_mutate(seq, self.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation_fn: {self.mutation_fn}")

    def _score_population(self, antibodies: List[Antibody]) -> None:
        seqs = [ab.sequence for ab in antibodies]
        scores = self.scorer(seqs)
        for ab, s in zip(antibodies, scores):
            ab.score = s

    def run(self, seed_sequences: List[str]) -> List[Generation]:
        """
        Run the evolutionary loop starting from *seed_sequences*.

        Returns
        -------
        List of Generation objects (one per generation + initial).
        """
        # Initialise population
        population: List[Antibody] = [
            Antibody(sequence=s, generation=0) for s in seed_sequences
        ]
        self._score_population(population)
        history = [Generation(index=0, population=population)]
        logger.info("Gen 0 | best=%.4f  mean=%.4f",
                    history[0].best_score, history[0].mean_score)

        for gen in range(1, self.n_generations + 1):
            n_parents = max(1, int(len(population) * self.top_fraction))
            parents = top_k_select(population, k=n_parents)

            # Breed next generation
            next_pop: List[Antibody] = list(parents)  # elitism
            while len(next_pop) < self.population_size:
                parent = random.choice(parents)
                child_seq = self._mutate(parent.sequence)
                next_pop.append(Antibody(
                    sequence=child_seq,
                    generation=gen,
                    parent=parent.sequence,
                ))

            self._score_population(next_pop)
            population = next_pop
            g = Generation(index=gen, population=population)
            history.append(g)
            logger.info("Gen %d | best=%.4f  mean=%.4f  pop=%d",
                        gen, g.best_score, g.mean_score, len(population))

        return history

    def top_candidates(self, history: List[Generation],
                       n: int = 100) -> List[Antibody]:
        """
        Return the top *n* unique sequences (by score) across all generations.
        """
        all_ab = {ab.sequence: ab
                  for gen in history
                  for ab in gen.population}
        return sorted(all_ab.values(), key=lambda ab: ab.score,
                      reverse=True)[:n]


# ─── Diversity metrics ────────────────────────────────────────────────────────

def hamming_distance(seq_a: str, seq_b: str) -> int:
    return sum(a != b for a, b in zip(seq_a, seq_b))


def diversity_score(sequences: List[str]) -> float:
    """Mean pairwise Hamming distance (normalised by length)."""
    if len(sequences) < 2:
        return 0.0
    n = len(sequences[0])
    dists = []
    sample = sequences[:200]  # cap for speed
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dists.append(hamming_distance(sample[i], sample[j]) / max(n, 1))
    return float(np.mean(dists))
