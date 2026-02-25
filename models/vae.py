"""
models/vae.py
Variational Autoencoder for antibody conformational space exploration.

The VAE learns a continuous latent representation of antibody residue features
(e.g. distance matrices, dihedral angles, contact maps).  Sampling from the
latent space generates novel conformational states beyond MD sampling limits.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_layer(h), self.log_var_layer(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AntibodyVAE(nn.Module):
    """
    VAE for antibody conformation features.

    Input  : flattened feature vector (distances, dihedrals, contact map)
    Latent : continuous Gaussian distribution z ~ N(mu, sigma²)
    Output : reconstructed feature vector

    Parameters
    ----------
    input_dim  : dimensionality of input feature vector
    hidden_dim : width of hidden layers
    latent_dim : dimensionality of latent space
    beta       : weight on KL divergence term (beta-VAE, beta=1 → standard VAE)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512,
                 latent_dim: int = 64, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    # ── Core forward ─────────────────────────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        recon : reconstructed x
        mu    : latent mean
        log_var : latent log-variance
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean (deterministic) for a batch."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Sample *n* novel conformations from the prior N(0,I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(self, x: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss

    # ── Training helper ───────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, epochs: int = 100,
            batch_size: int = 32, lr: float = 1e-3,
            device: str = "cpu") -> list[float]:
        """
        Train VAE on a numpy array of conformation features.

        Parameters
        ----------
        features : np.ndarray of shape (N, input_dim)
        epochs   : training epochs
        batch_size, lr, device : standard training params

        Returns
        -------
        List of per-epoch losses.
        """
        self.to(device)
        tensor = torch.tensor(features, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor),
                            batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epoch_losses = []

        for epoch in range(1, epochs + 1):
            total = 0.0
            for (x,) in loader:
                x = x.to(device)
                recon, mu, log_var = self(x)
                loss = self.loss(x, recon, mu, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
            avg = total / len(tensor)
            epoch_losses.append(avg)
            if epoch % 10 == 0:
                logger.info("VAE epoch %d/%d  loss=%.4f", epoch, epochs, avg)

        return epoch_losses

    def get_latent_embeddings(self, features: np.ndarray,
                              device: str = "cpu") -> np.ndarray:
        """Encode features → latent means (numpy)."""
        self.eval()
        tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            z = self.encode(tensor)
        return z.cpu().numpy()

    def generate_structures(self, n: int = 100,
                            device: str = "cpu") -> np.ndarray:
        """Sample *n* synthetic conformations from prior."""
        self.eval()
        with torch.no_grad():
            samples = self.sample(n, device=device)
        return samples.cpu().numpy()

    # ── Checkpoint persistence ────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """
        Save model weights and architecture config to *path* (.pt file).

        The checkpoint stores everything needed to reconstruct the model via
        :meth:`load_checkpoint` without requiring the original constructor args.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.encoder.net[0].in_features,
                "hidden_dim": self.encoder.net[0].out_features,
                "latent_dim": self.latent_dim,
                "beta": self.beta,
            },
            path,
        )
        logger.info("VAE checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str,
                        device: str = "cpu") -> "AntibodyVAE":
        """
        Reconstruct an :class:`AntibodyVAE` from a checkpoint file.

        Parameters
        ----------
        path   : path to the .pt checkpoint produced by :meth:`save_checkpoint`.
        device : torch device string (e.g. 'cpu', 'cuda').

        Returns
        -------
        AntibodyVAE with weights restored, moved to *device*.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            beta=checkpoint["beta"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        logger.info("VAE checkpoint loaded from %s", path)
        return model
