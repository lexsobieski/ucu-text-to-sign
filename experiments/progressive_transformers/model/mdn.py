"""
Mixture Density Network head + loss, following Saunders et al., IJCV 2021.

Output per step: K mixture components, each a diagonal Gaussian over D target dims.
Raw output shape: (B, T, K*(2D+1)) -> (log_pi, mu, log_sigma).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_LOG_2PI = math.log(2.0 * math.pi)


def mdn_output_size(n_components: int, trg_size: int) -> int:
    """Raw linear-layer output size for an MDN with K components over D dims."""
    return n_components * (2 * trg_size + 1)


def split_mdn_output(raw: Tensor, n_components: int, trg_size: int,
                     log_sigma_min: float = -5.0,
                     log_sigma_max: float = 2.0):
    """Split raw MDN output into (log_pi, mu, log_sigma).

    raw: (B, T, K*(2D+1))
    returns:
        log_pi:    (B, T, K)
        mu:        (B, T, K, D)
        log_sigma: (B, T, K, D), clamped for numerical stability
    """
    K, D = n_components, trg_size
    B, T, _ = raw.shape

    pi_logits = raw[..., : K]
    mu = raw[..., K : K + K * D].reshape(B, T, K, D)
    log_sigma = raw[..., K + K * D :].reshape(B, T, K, D)
    log_sigma = log_sigma.clamp(log_sigma_min, log_sigma_max)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    return log_pi, mu, log_sigma


def mdn_nll(log_pi: Tensor, mu: Tensor, log_sigma: Tensor,
            target: Tensor, target_pad: float = 0.0) -> Tensor:
    """Negative log-likelihood of target under Gaussian mixture, averaged over non-pad dims.

    log_pi:    (B, T, K)
    mu:        (B, T, K, D)
    log_sigma: (B, T, K, D)
    target:    (B, T, D)
    """
    target = target.unsqueeze(2)  # (B, T, 1, D)
    sigma = log_sigma.exp()

    # Per-component log-likelihood per dim, summed over dims:
    #   log N(y | mu, sigma) = -0.5 * ((y - mu)/sigma)^2 - log_sigma - 0.5*log(2pi)
    z = (target - mu) / sigma
    comp_logprob = -0.5 * z.pow(2) - log_sigma - 0.5 * _LOG_2PI  # (B, T, K, D)
    comp_logprob = comp_logprob.sum(dim=-1)                      # (B, T, K)

    # Mixture log-likelihood
    log_prob = torch.logsumexp(log_pi + comp_logprob, dim=-1)    # (B, T)

    # Mask out pad frames (where target is entirely pad value)
    frame_mask = (target.squeeze(2) != target_pad).any(dim=-1).float()  # (B, T)
    nll = -(log_prob * frame_mask).sum() / frame_mask.sum().clamp(min=1.0)
    return nll


def mdn_to_pose(log_pi: Tensor, mu: Tensor, log_sigma: Tensor,
                mode: str = "argmax") -> Tensor:
    """Convert a mixture into a deterministic pose for autoregressive feedback.

    mode:
      "argmax" -- return mean of most-probable component (deterministic, preferred)
      "mean"   -- return mixture-weighted mean (can blur, behaves like MSE)
      "sample" -- sample a component and sample from its Gaussian

    Returns: (B, T, D) or (B, 1, D) depending on input time dim.
    """
    if mode == "argmax":
        k_star = log_pi.argmax(dim=-1, keepdim=True)  # (B, T, 1)
        k_star = k_star.unsqueeze(-1).expand(-1, -1, 1, mu.shape[-1])
        return mu.gather(dim=2, index=k_star).squeeze(2)

    if mode == "mean":
        pi = log_pi.exp().unsqueeze(-1)  # (B, T, K, 1)
        return (pi * mu).sum(dim=2)

    if mode == "sample":
        pi = log_pi.exp()
        dist = torch.distributions.Categorical(probs=pi)
        k = dist.sample().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.shape[-1])
        chosen_mu = mu.gather(dim=2, index=k).squeeze(2)
        chosen_log_sigma = log_sigma.gather(dim=2, index=k).squeeze(2)
        eps = torch.randn_like(chosen_mu)
        return chosen_mu + eps * chosen_log_sigma.exp()

    raise ValueError(f"Unknown mdn_to_pose mode: {mode}")


class MDNLoss(nn.Module):
    """Thin nn.Module wrapper around mdn_nll for use in the training loop."""

    def __init__(self, n_components: int, trg_size: int, target_pad: float = 0.0):
        super().__init__()
        self.n_components = n_components
        self.trg_size = trg_size
        self.target_pad = target_pad

    def forward(self, raw: Tensor, target: Tensor) -> Tensor:
        log_pi, mu, log_sigma = split_mdn_output(raw, self.n_components, self.trg_size)
        return mdn_nll(log_pi, mu, log_sigma, target, target_pad=self.target_pad)
