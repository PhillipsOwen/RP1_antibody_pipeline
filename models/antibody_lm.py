"""
models/antibody_lm.py
Antibody Language Model wrapper using ESM2 (or any HuggingFace masked LM).

Responsibilities:
  - Score sequences by pseudo-log-likelihood
  - Generate mutation candidates via masked token sampling
  - Produce sequence embeddings for downstream clustering / MSM featurization
  - Build disease-specific reference atlases from BCR repertoire embeddings
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Amino acid alphabet (standard 20)
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class AntibodyLM:
    """
    Wrapper around a HuggingFace masked protein language model (e.g. ESM-2).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model and tokenizer on first use."""
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            logger.info("Loading %s …", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Model loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required: "
                "pip install transformers torch"
            )

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, sequences: List[str]) -> np.ndarray:
        """
        Return mean-pooled residue embeddings for a list of sequences.

        Returns
        -------
        np.ndarray of shape (N, hidden_dim)
        """
        embeddings = []
        for seq in sequences:
            enc = self.tokenizer(seq, return_tensors="pt",
                                 truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            # Last hidden state, mean over residue dimension (exclude BOS/EOS)
            hidden = out.hidden_states[-1][0, 1:-1, :]  # (L, D)
            embeddings.append(hidden.mean(dim=0).cpu().numpy())
        return np.array(embeddings)

    def score(self, sequences: List[str]) -> List[float]:
        """
        Compute pseudo-log-likelihood score for each sequence.
        Higher = more 'natural' according to the model.

        Returns
        -------
        List of float scores, one per sequence.
        """
        scores = []
        for seq in sequences:
            score = self._pseudo_log_likelihood(seq)
            scores.append(score)
        return scores

    def generate_mutations(self, seed_sequence: str,
                           n_mutations: int = 1,
                           n_samples: int = 50,
                           top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate point-mutated variants of *seed_sequence* using masked
        token sampling from the LM.

        Parameters
        ----------
        seed_sequence : str
            Starting antibody sequence (single-letter AA codes).
        n_mutations : int
            Number of simultaneous point mutations per sample.
        n_samples : int
            How many mutant sequences to return.
        top_k : int
            Sample from top-k token predictions at each masked position.

        Returns
        -------
        List of (mutant_sequence, lm_score) sorted by score descending.
        """
        results: List[Tuple[str, float]] = []
        seq_list = list(seed_sequence)

        for _ in range(n_samples):
            positions = random.sample(range(len(seq_list)), k=n_mutations)
            candidate = seq_list.copy()

            for pos in positions:
                candidate[pos] = self.tokenizer.mask_token  # type: ignore[index]

            masked_str = "".join(
                c if c != self.tokenizer.mask_token else self.tokenizer.mask_token
                for c in candidate
            )

            enc = self.tokenizer(masked_str, return_tensors="pt",
                                 truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits[0]  # (seq_len, vocab)

            # Fill masked positions with top-k sampled tokens
            filled = seq_list.copy()
            for pos in positions:
                token_idx = pos + 1  # account for BOS token
                probs = F.softmax(logits[token_idx], dim=-1)
                top_ids = torch.topk(probs, top_k).indices
                chosen_id = top_ids[random.randint(0, top_k - 1)].item()
                token_str = self.tokenizer.convert_ids_to_tokens([chosen_id])[0]
                # ESM tokens are single AA letters or special tokens
                if token_str in AA_ALPHABET:
                    filled[pos] = token_str

            mutant = "".join(filled)
            score = self._pseudo_log_likelihood(mutant)
            results.append((mutant, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Atlas construction ────────────────────────────────────────────────────

    def build_individual_atlases(
        self,
        sequences_by_individual: Dict[str, List[str]],
        batch_size: int = 32,
    ) -> Dict[str, dict]:
        """
        Build per-individual embedding atlases from subject-stratified BCR data.

        SA3b: each donor's BCR repertoire is summarised as a separate atlas
        (centroid + covariance), enabling per-individual B-cell fingerprinting
        and protection estimation against HIV-1 variants.

        SA3a: atlases built per-donor and per-time-point can be differenced or
        correlated with disease outcomes for variant-susceptibility modelling.

        Parameters
        ----------
        sequences_by_individual : dict mapping individual_id → list of AA seqs.
        batch_size              : embedding batch size.

        Returns
        -------
        dict mapping individual_id → atlas dict (same schema as build_atlas()).
        Each atlas additionally includes:
          'individual_id' : the donor identifier.
          'n_sequences'   : number of sequences embedded for this donor.
        """
        atlases: Dict[str, dict] = {}
        for ind_id, seqs in sequences_by_individual.items():
            if not seqs:
                logger.warning("Individual '%s' has no sequences — skipping.", ind_id)
                continue
            atlas = self.build_atlas(seqs, disease_label=ind_id, batch_size=batch_size)
            atlas["individual_id"] = ind_id
            atlases[ind_id] = atlas
            logger.info(
                "Individual atlas built: id=%s  n=%d  dim=%d",
                ind_id, atlas["n_sequences"], atlas["centroid"].shape[0],
            )
        return atlases

    def compute_individual_fingerprint(
        self,
        individual_atlas: dict,
        category_reference_atlases: Dict[str, dict],
    ) -> Dict[str, float]:
        """
        Compute a per-individual B-cell immune fingerprint (SA3b).

        Measures each donor's BCR repertoire similarity to each known
        neutralization-mechanism category (e.g., V1/V2, V3, gp41 binders)
        by computing cosine similarity between the individual's atlas centroid
        and each category reference centroid.

        Parameters
        ----------
        individual_atlas            : atlas dict from build_individual_atlases().
        category_reference_atlases  : dict mapping category_name → reference atlas.
                                      Build one atlas per neutralization category
                                      (V1/V2, V3, gp41, etc.) from known antibodies
                                      in that category.

        Returns
        -------
        dict mapping category_name → cosine similarity score in [0, 1].
        Higher score = individual's repertoire is more similar to that category.
        """
        ind_centroid = individual_atlas["centroid"]
        fingerprint: Dict[str, float] = {}
        for category, ref_atlas in category_reference_atlases.items():
            ref_centroid = ref_atlas["centroid"]
            norm_i = np.linalg.norm(ind_centroid)
            norm_r = np.linalg.norm(ref_centroid)
            if norm_i == 0 or norm_r == 0:
                fingerprint[category] = 0.0
            else:
                cos_sim = float(
                    np.dot(ind_centroid, ref_centroid) / (norm_i * norm_r)
                )
                fingerprint[category] = (cos_sim + 1.0) / 2.0  # map [-1,1] → [0,1]
        return fingerprint

    def build_atlas(
        self,
        sequences: List[str],
        disease_label: str = "unknown",
        batch_size: int = 32,
    ) -> dict:
        """
        Build a disease-specific reference embedding atlas from a BCR repertoire.

        Embeds all sequences in batches and stores the centroid, covariance,
        and per-sequence embeddings as a named atlas.

        Parameters
        ----------
        sequences     : list of antibody AA sequences (from BCRRepertoire).
        disease_label : label for the atlas (e.g. 'COVID-19').
        batch_size    : embedding batch size.

        Returns
        -------
        dict with keys:
          'disease'     : disease_label
          'centroid'    : mean embedding vector  (hidden_dim,)
          'std'         : per-dimension std       (hidden_dim,)
          'embeddings'  : (N, hidden_dim) array of all per-sequence embeddings
          'n_sequences' : number of sequences embedded
        """
        all_embs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            embs = self.embed(batch)           # (B, hidden_dim)
            all_embs.append(embs)

        embeddings = np.vstack(all_embs)       # (N, hidden_dim)
        centroid = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)

        atlas = {
            "disease": disease_label,
            "centroid": centroid,
            "std": std,
            "embeddings": embeddings,
            "n_sequences": len(sequences),
        }
        logger.info(
            "Atlas built: disease=%s  n=%d  embedding_dim=%d",
            disease_label, len(sequences), centroid.shape[0],
        )
        return atlas

    def atlas_similarity(self, query_sequence: str, atlas: dict) -> float:
        """
        Compute the similarity of a query antibody sequence to a disease atlas.

        Returns cosine similarity between the query embedding and the atlas
        centroid, scaled to [0, 1].

        Parameters
        ----------
        query_sequence : antibody AA sequence to evaluate.
        atlas          : atlas dict returned by build_atlas().

        Returns
        -------
        float in [0, 1]
        """
        query_emb = self.embed([query_sequence])[0]         # (hidden_dim,)
        centroid = atlas["centroid"]

        norm_q = np.linalg.norm(query_emb)
        norm_c = np.linalg.norm(centroid)
        if norm_q == 0 or norm_c == 0:
            return 0.0
        cos_sim = float(np.dot(query_emb, centroid) / (norm_q * norm_c))
        return (cos_sim + 1.0) / 2.0                        # map [-1,1] → [0,1]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pseudo_log_likelihood(self, sequence: str) -> float:
        """
        Approximate pseudo-log-likelihood by summing log probabilities of
        each token when masked one at a time.
        """
        total_log_prob = 0.0
        seq_tokens = self.tokenizer.encode(sequence, add_special_tokens=False)

        for i, true_token in enumerate(seq_tokens):
            masked_ids = seq_tokens.copy()
            masked_ids[i] = self.tokenizer.mask_token_id
            input_ids = torch.tensor(
                [[self.tokenizer.cls_token_id] + masked_ids
                 + [self.tokenizer.eos_token_id]]
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0]

            log_probs = F.log_softmax(logits[i + 1], dim=-1)
            total_log_prob += log_probs[true_token].item()

        return total_log_prob / max(len(seq_tokens), 1)


# ── Lightweight fallback (no GPU / transformers) ──────────────────────────────

class RandomAntibodyLM:
    """
    Dummy LM that generates random mutations and random scores.
    Useful for pipeline testing without downloading large models.
    """

    def embed(self, sequences: List[str]) -> np.ndarray:
        return np.random.randn(len(sequences), 64)

    def score(self, sequences: List[str]) -> List[float]:
        return [random.gauss(-10, 2) for _ in sequences]

    def generate_mutations(self, seed_sequence: str,
                           n_mutations: int = 1,
                           n_samples: int = 50,
                           top_k: int = 10) -> List[Tuple[str, float]]:
        results = []
        for _ in range(n_samples):
            seq = list(seed_sequence)
            for pos in random.sample(range(len(seq)), k=min(n_mutations, len(seq))):
                seq[pos] = random.choice(AA_ALPHABET)
            score = random.gauss(-10, 2)
            results.append(("".join(seq), score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def build_atlas(
        self,
        sequences: List[str],
        disease_label: str = "unknown",
        batch_size: int = 32,
    ) -> dict:
        """Mock atlas: random embeddings with a stable centroid."""
        embeddings = np.random.randn(len(sequences), 64)
        return {
            "disease": disease_label,
            "centroid": embeddings.mean(axis=0),
            "std": embeddings.std(axis=0),
            "embeddings": embeddings,
            "n_sequences": len(sequences),
        }

    def build_individual_atlases(
        self,
        sequences_by_individual: Dict[str, List[str]],
        batch_size: int = 32,
    ) -> Dict[str, dict]:
        """Mock per-individual atlases: random embeddings per donor."""
        atlases: Dict[str, dict] = {}
        for ind_id, seqs in sequences_by_individual.items():
            if not seqs:
                continue
            embeddings = np.random.randn(len(seqs), 64)
            atlases[ind_id] = {
                "individual_id": ind_id,
                "disease": ind_id,
                "centroid": embeddings.mean(axis=0),
                "std": embeddings.std(axis=0),
                "embeddings": embeddings,
                "n_sequences": len(seqs),
            }
        return atlases

    def compute_individual_fingerprint(
        self,
        individual_atlas: dict,
        category_reference_atlases: Dict[str, dict],
    ) -> Dict[str, float]:
        """Mock fingerprint: random similarity per category."""
        return {cat: random.uniform(0.3, 0.7) for cat in category_reference_atlases}

    def atlas_similarity(self, query_sequence: str, atlas: dict) -> float:
        """Mock atlas similarity: uniform random in [0.4, 0.8]."""
        return random.uniform(0.4, 0.8)


def get_lm(model_name: Optional[str] = None,
           device: str = "cpu",
           use_mock: bool = False) -> AntibodyLM | RandomAntibodyLM:
    """Factory: return real or mock LM based on *use_mock*."""
    if use_mock:
        logger.info("Using RandomAntibodyLM (mock mode).")
        return RandomAntibodyLM()
    name = model_name or "facebook/esm2_t33_650M_UR50D"
    return AntibodyLM(model_name=name, device=device)
