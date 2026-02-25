"""
models/gan.py
Generative Adversarial Network for realistic antibody structure generation.

The Generator produces synthetic structural feature vectors.
The Discriminator distinguishes real (MD / PDB) structures from generated ones.
Training drives the Generator to produce structurally realistic samples.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─── Building blocks ─────────────────────────────────────────────────────────

def _mlp(dims: list[int], activation: type = nn.LeakyReLU,
         final_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation(0.2, inplace=True)
                          if activation is nn.LeakyReLU
                          else activation())
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


# ─── Generator ───────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps Gaussian noise z → synthetic structural feature vector.

    noise_dim  → hidden → output_dim
    """

    def __init__(self, noise_dim: int = 128, hidden_dim: int = 256,
                 output_dim: int = 256):
        super().__init__()
        self.net = _mlp(
            [noise_dim, hidden_dim, hidden_dim, output_dim],
            final_activation=nn.Tanh()  # features normalised to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── Discriminator ───────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Classifies feature vectors as real (1) or generated (0).
    Uses spectral normalisation for training stability.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── AntibodyGAN ─────────────────────────────────────────────────────────────

class AntibodyGAN:
    """
    High-level wrapper combining Generator and Discriminator.

    Training uses standard non-saturating GAN losses with separate
    optimizers for G and D (Adam).

    Parameters
    ----------
    noise_dim   : latent noise dimension
    hidden_dim  : hidden layer width (shared for G and D)
    output_dim  : structural feature vector size (must match real data)
    lr          : learning rate for both optimizers
    device      : 'cpu' or 'cuda'
    """

    def __init__(self, noise_dim: int = 128, hidden_dim: int = 256,
                 output_dim: int = 256, lr: float = 2e-4,
                 device: str = "cpu"):
        self.noise_dim = noise_dim
        self.device = device
        self.G = Generator(noise_dim, hidden_dim, output_dim).to(device)
        self.D = Discriminator(output_dim, hidden_dim).to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, real_features: np.ndarray, epochs: int = 200,
            batch_size: int = 32) -> Tuple[list[float], list[float]]:
        """
        Train GAN on *real_features* (numpy array, shape N × output_dim).

        Returns
        -------
        g_losses, d_losses : per-epoch loss lists
        """
        real_tensor = torch.tensor(
            _normalize(real_features), dtype=torch.float32
        )
        loader = DataLoader(TensorDataset(real_tensor),
                            batch_size=batch_size, shuffle=True,
                            drop_last=True)
        g_losses, d_losses = [], []

        for epoch in range(1, epochs + 1):
            g_total = d_total = 0.0

            for (real_batch,) in loader:
                bs = real_batch.size(0)
                real_batch = real_batch.to(self.device)
                ones = torch.ones(bs, 1, device=self.device)
                zeros = torch.zeros(bs, 1, device=self.device)

                # ── Train Discriminator ──
                z = torch.randn(bs, self.noise_dim, device=self.device)
                fake = self.G(z).detach()
                d_loss = (
                    self.criterion(self.D(real_batch), ones)
                    + self.criterion(self.D(fake), zeros)
                )
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                # ── Train Generator ──
                z = torch.randn(bs, self.noise_dim, device=self.device)
                g_loss = self.criterion(self.D(self.G(z)), ones)
                self.opt_G.zero_grad()
                g_loss.backward()
                self.opt_G.step()

                g_total += g_loss.item()
                d_total += d_loss.item()

            n_batches = max(len(loader), 1)
            g_losses.append(g_total / n_batches)
            d_losses.append(d_total / n_batches)

            if epoch % 20 == 0:
                logger.info(
                    "GAN epoch %d/%d  G_loss=%.4f  D_loss=%.4f",
                    epoch, epochs, g_losses[-1], d_losses[-1]
                )

        return g_losses, d_losses

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(self, n: int = 100) -> np.ndarray:
        """
        Generate *n* synthetic structural feature vectors.

        Returns
        -------
        np.ndarray of shape (n, output_dim)
        """
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.noise_dim, device=self.device)
            fake = self.G(z)
        return fake.cpu().numpy()

    def discriminator_score(self, features: np.ndarray) -> np.ndarray:
        """
        Return discriminator confidence (0–1) that each sample is 'real'.
        Use as a structural plausibility score.
        """
        self.D.eval()
        tensor = torch.tensor(
            _normalize(features), dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            scores = self.D(tensor)
        return scores.cpu().numpy().flatten()


# ─── Utility ─────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [-1, 1] column-wise."""
    mn, mx = x.min(axis=0), x.max(axis=0)
    denom = np.where((mx - mn) == 0, 1.0, mx - mn)
    return 2 * (x - mn) / denom - 1
