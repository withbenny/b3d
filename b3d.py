"""
b3d.py — Black-box Backdoor Detection (B3D) core algorithm.

Implements Algorithm 1 and B3D-SS from:
  Dong et al. (2021) "Black-box Detection of Backdoor Attacks with
  Limited Information and Data", ICCV 2021.

Key idea: use Natural Evolution Strategies (NES) to reverse-engineer
a potential trigger (m, p) for every class c by minimising

    F(m, p; c) = E_{x~X}[ CE(c, f(A(x, m, p))) ] + lambda * ||g(theta_m)||_1

where A(x, m, p) = (1-m)*x + m*p, m in {0,1}^d, p in [0,1]^d.

Critical implementation details (from reference implementation):
  - theta_m is initialised to -5.0  =>  g(-5) ~= 0.007  (nearly empty mask)
  - L1 norm is computed on the SOFT mask g(theta_m), not the binary mask
  - Lambda update uses a 10-step rolling-mean ASR (gentler than per-step EMA)
  - ASR is averaged over 2k samples (k mask-loop + k pattern-loop)
  - Lambda multipliers: x1.2 up / x0.8 down  (gentler than x1.5/x0.85)

Paper hyperparameters (Appendix B): k=50, sigma=0.1, lr=0.05. batch_size not stated.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Tuple

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def g(theta: torch.Tensor) -> torch.Tensor:
    """Normalisation: g(theta) = 0.5*(tanh(theta)+1), mapping R -> [0,1]."""
    return 0.5 * (torch.tanh(theta) + 1.0)


def apply_trigger(x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Trigger-stamping: x' = (1-m)*x + m*p, clipped to [0,1]  (Eq. 1)."""
    return ((1.0 - m) * x + m * p).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# B3D optimiser — Algorithm 1
# ---------------------------------------------------------------------------

class B3DOptimizer:
    """
    Reverse-engineer the potential trigger for every class using NES.

    Paper defaults: k=50, sigma=0.1, lr=0.05, batch_size=128.

    Args:
        model_fn:       Black-box model. Input: float [N, *image_shape] in [0,1].
                        Output: logit tensor [N, num_classes].
        num_classes:    Number of classes C.
        image_shape:    Shape of ONE image, e.g. (3, 32, 32).
        k:              NES samples per iteration (paper: 50).
        sigma:          Gaussian std for pattern NES (paper: 0.1).
        lr:             Adam learning rate (paper: 0.05).
        lam_init:       Initial lambda for L1 regularisation.
        lam_up:         Lambda multiplier when rolling-mean ASR >= threshold.
        lam_down:       Lambda multiplier when rolling-mean ASR < threshold.
        lam_max:        Hard upper bound on lambda.
        lam_min:        Hard lower bound on lambda.
        asr_window:     Rolling-window size for ASR smoothing (default 10).
        asr_threshold:  Target ASR for adaptive lambda (default 0.99).
        num_iterations: Optimisation steps T.
        batch_size:     Minibatch size (paper: 128).
        device:         Torch device string.
    """

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        num_classes: int,
        image_shape: Tuple[int, ...],
        k: int = 50,
        sigma: float = 0.1,
        lr: float = 0.05,
        lam_init: float = 0.01,
        lam_up: float = 1.2,
        lam_down: float = 0.8,
        lam_max: float = 1e2,
        lam_min: float = 1e-5,
        asr_window: int = 10,
        asr_threshold: float = 0.99,
        num_iterations: int = 1000,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.k = k
        self.sigma = sigma
        self.lr = lr
        self.lam_init = lam_init
        self.lam_up = lam_up
        self.lam_down = lam_down
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.asr_window = asr_window
        self.asr_threshold = asr_threshold
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.device = device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _vectorized_loss_and_asr_mask(
        self,
        x_batch: torch.Tensor,
        gm: torch.Tensor,
        gp: torch.Tensor,
        target_class: int,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Vectorized mask gradient step: sample k masks simultaneously.

        Stacks k Bernoulli mask samples with the x_batch into a single
        forward pass of size [k * batch_size, C, H, W], replacing the
        Python for-loop with one GPU call.

        Returns:
            losses_k: [k] per-sample loss values
            m_k:      [k, *image_shape] sampled masks
            mean_asr: scalar, mean ASR over all k*batch_size predictions
        """
        B = x_batch.shape[0]
        # Sample k masks at once: [k, *image_shape]
        m_k = torch.bernoulli(gm.unsqueeze(0).expand(self.k, *self.image_shape))

        # Expand x_batch to [k*B, C, H, W] and masks to [k*B, C, H, W]
        x_rep = x_batch.unsqueeze(0).expand(self.k, *x_batch.shape).reshape(self.k * B, *self.image_shape)
        m_rep = m_k.unsqueeze(1).expand(self.k, B, *self.image_shape).reshape(self.k * B, *self.image_shape)
        gp_rep = gp.unsqueeze(0).expand(self.k * B, *self.image_shape)

        x_trig = apply_trigger(x_rep, m_rep, gp_rep)
        logits = self.model_fn(x_trig)

        tgt = torch.full((self.k * B,), target_class, dtype=torch.long, device=self.device)
        # Per-sample CE loss (reduction='none'), then mean over batch for each k
        ce_per_sample = F.cross_entropy(logits, tgt, reduction="none")  # [k*B]
        ce_per_k = ce_per_sample.view(self.k, B).mean(dim=1)            # [k]
        l1_per_k = m_k.view(self.k, -1).sum(dim=1)                     # [k]
        losses_k = ce_per_k + lam * l1_per_k                            # [k]

        mean_asr = (logits.argmax(1) == tgt).float().mean().item()
        return losses_k, m_k, mean_asr

    @torch.no_grad()
    def _vectorized_loss_and_asr_pattern(
        self,
        x_batch: torch.Tensor,
        gm: torch.Tensor,
        theta_p: torch.Tensor,
        target_class: int,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Vectorized pattern gradient step: sample k epsilon perturbations simultaneously.

        Returns:
            losses_k: [k] per-sample loss values
            eps_k:    [k, *image_shape] sampled noise vectors
            mean_asr: scalar
        """
        B = x_batch.shape[0]
        # Sample k epsilon vectors: [k, *image_shape]
        eps_k = torch.randn(self.k, *self.image_shape, device=self.device)
        # Perturbed patterns: [k, *image_shape]
        p_k = g(theta_p.unsqueeze(0) + self.sigma * eps_k)

        x_rep  = x_batch.unsqueeze(0).expand(self.k, *x_batch.shape).reshape(self.k * B, *self.image_shape)
        gm_rep = gm.unsqueeze(0).expand(self.k * B, *self.image_shape)
        p_rep  = p_k.unsqueeze(1).expand(self.k, B, *self.image_shape).reshape(self.k * B, *self.image_shape)

        x_trig = apply_trigger(x_rep, gm_rep, p_rep)
        logits = self.model_fn(x_trig)

        tgt = torch.full((self.k * B,), target_class, dtype=torch.long, device=self.device)
        ce_per_sample = F.cross_entropy(logits, tgt, reduction="none")   # [k*B]
        ce_per_k = ce_per_sample.view(self.k, B).mean(dim=1)             # [k]
        losses_k = ce_per_k + lam * gm.abs().sum()                       # scalar broadcast to [k]

        mean_asr = (logits.argmax(1) == tgt).float().mean().item()
        return losses_k, eps_k, mean_asr

    # ------------------------------------------------------------------
    def _update_lambda(
        self,
        lam: float,
        asr_history: deque,
        current_asr: float,
    ) -> float:
        """
        Adaptive lambda: update only after asr_window samples accumulate.
        Uses a rolling-window mean (same as reference implementation).
        """
        asr_history.append(current_asr)
        if len(asr_history) < asr_history.maxlen:
            return lam  # not enough history yet
        mean_asr = float(np.mean(asr_history))
        if mean_asr < self.asr_threshold:
            lam = lam * self.lam_down
        else:
            lam = lam * self.lam_up
        return max(self.lam_min, min(lam, self.lam_max))

    # ------------------------------------------------------------------
    def optimize_class(
        self,
        X: torch.Tensor,
        target_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run Algorithm 1 for one target class c.

        Args:
            X:            Clean images [N, *image_shape] in [0, 1].
            target_class: Target class index c.

        Returns:
            m_soft:  Soft mask g(theta_m) [*image_shape] in [0, 1].
            p:       Trigger pattern g(theta_p) [*image_shape] in [0, 1].
            l1_norm: L1 norm of the soft mask (used for outlier detection).
        """
        N = X.shape[0]
        lam = self.lam_init
        asr_history: deque = deque(maxlen=self.asr_window)

        # ---- Initialise theta_m = -5  =>  g(-5) ~= 0.007 (nearly empty mask)
        # This is the critical initialisation: starting near-zero forces the
        # optimiser to GROW the mask only where it reduces loss, rather than
        # pruning a random 50%-mask (which never converges to a small trigger).
        theta_m = torch.nn.Parameter(
            torch.full(self.image_shape, -5.0, device=self.device)
        )
        theta_p = torch.nn.Parameter(
            torch.zeros(self.image_shape, device=self.device)
        )

        optimizer_m = optim.Adam([theta_m], lr=self.lr)
        optimizer_p = optim.Adam([theta_p], lr=self.lr)

        for t in range(self.num_iterations):
            # Draw minibatch X_t  (Algorithm 1 line 4)
            idx = torch.randperm(N, device=self.device)[: self.batch_size]
            X_t = X[idx]

            gm = g(theta_m.detach())   # g(theta_m) in [0,1]
            gp = g(theta_p.detach())   # g(theta_p) in [0,1]

            optimizer_m.zero_grad()
            optimizer_p.zero_grad()

            # ---- Vectorized gradient for theta_m  (Eq. 5) -----------------
            # ĝ_m = (1/k) * sum_j  F(m_j, p; c) * 2*(m_j - g(theta_m))
            losses_m, m_k, asr_m = self._vectorized_loss_and_asr_mask(
                X_t, gm, gp, target_class, lam)
            # losses_m: [k], m_k: [k, *image_shape]
            grad_m = (losses_m.view(-1, *([1] * len(self.image_shape))) * 2.0 * (m_k - gm)).mean(dim=0)

            # ---- Vectorized gradient for theta_p  (Eq. 6) -----------------
            # ĝ_p = (1/k*sigma) * sum_j  F(g(theta_m), p_j; c) * eps_j
            losses_p, eps_k, asr_p = self._vectorized_loss_and_asr_pattern(
                X_t, gm, theta_p.detach(), target_class, lam)
            # losses_p: [k], eps_k: [k, *image_shape]
            grad_p = (losses_p.view(-1, *([1] * len(self.image_shape))) * eps_k).mean(dim=0) / self.sigma

            # ---- Apply gradients via Adam  (Algorithm 1 lines 13-14) ------
            theta_m.grad = grad_m
            theta_p.grad = grad_p
            optimizer_m.step()
            optimizer_p.step()

            # ---- Adaptive lambda  -----------------------------------------
            mean_asr = (asr_m + asr_p) / 2.0
            lam = self._update_lambda(lam, asr_history, mean_asr)

            if (t + 1) % 100 == 0:
                soft_l1 = g(theta_m.detach()).sum().item()
                logger.info(
                    "class %d  iter %4d/%d  ASR=%.3f  lam=%.6f  soft_L1=%.2f",
                    target_class, t + 1, self.num_iterations,
                    mean_asr, lam, soft_l1,
                )

        # Final values  — use SOFT mask for both visualisation and L1 norm
        m_soft = g(theta_m.detach())
        p_soft = g(theta_p.detach())
        l1_norm = m_soft.sum().item()
        return m_soft, p_soft, l1_norm

    # ------------------------------------------------------------------
    def run(
        self,
        X: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Reverse-engineer triggers for all C classes.

        Args:
            X: Clean images [N, *image_shape] in [0, 1].

        Returns:
            masks:    C soft masks    [*image_shape] in [0,1].
            patterns: C trigger patterns [*image_shape] in [0,1].
            l1_norms: C L1 norms of soft masks (used for outlier detection).
        """
        masks:    List[torch.Tensor] = []
        patterns: List[torch.Tensor] = []
        l1_norms: List[float] = []

        for c in range(self.num_classes):
            logger.info("B3D - optimising class %d / %d ...", c, self.num_classes)
            m, p, l1 = self.optimize_class(X, c)
            masks.append(m)
            patterns.append(p)
            l1_norms.append(l1)

        return masks, patterns, l1_norms


# ---------------------------------------------------------------------------
# B3D-SS: generate synthetic samples via NES  (Section 3.4, Eq. 7)
# ---------------------------------------------------------------------------

def generate_synthetic_samples(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    image_shape: Tuple[int, ...],
    n_per_class: int = 100,
    k: int = 50,
    sigma: float = 0.1,
    lr: float = 0.05,
    num_iterations: int = 500,
    device: str = "cpu",
) -> torch.Tensor:
    """
    B3D-SS: synthesise class-representative images (Section 3.4, Eq. 7).

    For each class c, draws n random images and optimises them toward class c
    using the NES gradient estimator (no model gradients required):

        x_i^c <- x_i^c - lr * (1/k*sigma) * sum_j  CE(c, f(x + sigma*delta_j)) * delta_j

    Returns:
        X: [num_classes * n_per_class, *image_shape] synthetic images in [0,1].
    """
    all_images: List[torch.Tensor] = []
    n_img_dims = len(image_shape)

    for c in range(num_classes):
        logger.info("B3D-SS - synthesising class %d / %d ...", c, num_classes)
        x = torch.rand(n_per_class, *image_shape, device=device)

        for t in range(num_iterations):
            nes_grad = torch.zeros_like(x)
            for _ in range(k):
                delta = torch.randn_like(x)
                x_p = (x + sigma * delta).clamp(0.0, 1.0)
                with torch.no_grad():
                    logits = model_fn(x_p)
                tgt = torch.full((n_per_class,), c, dtype=torch.long, device=device)
                ce = F.cross_entropy(logits, tgt, reduction="none")
                nes_grad.add_(ce.view(-1, *([1] * n_img_dims)) * delta)
            nes_grad.div_(k * sigma)
            x = (x - lr * nes_grad).clamp(0.0, 1.0)

            if (t + 1) % 100 == 0:
                with torch.no_grad():
                    preds = model_fn(x).argmax(dim=1)
                acc = (preds == c).float().mean().item()
                logger.info(
                    "  class %d  iter %4d/%d  p(pred=c)=%.2f",
                    c, t + 1, num_iterations, acc,
                )

        all_images.append(x.detach())

    return torch.cat(all_images, dim=0)
