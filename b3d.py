from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class B3D:
    def __init__(
        self,
        model: nn.Module,
        normalizer: nn.Module,
        input_shape: torch.Size,
        num_classes: int,
        k: int = 50,
        lr: float = 0.05,
        sigma: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.normalizer = normalizer
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.k = k
        self.lr = lr
        self.sigma = sigma
        self.device = device

        self.theta_m = None
        self.theta_p = None
        self.optimizer_m = None
        self.optimizer_p = None
        self.lambda_reg = 0.01
        self.asr_history = deque(maxlen=10)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.tanh(x) + 1)

    def apply_trigger(
        self, x_batch: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor
    ) -> torch.Tensor:
        return (1.0 - mask) * x_batch + mask * pattern

    @torch.no_grad()
    def calculate_loss(
        self,
        x_batch: torch.Tensor,
        target_class: int,
        mask: torch.Tensor,
        pattern: torch.Tensor,
    ) -> tuple:
        x_batch = x_batch.to(self.device)
        mask = mask.to(self.device)
        pattern = pattern.to(self.device)

        x_poisoned = self.apply_trigger(x_batch, mask, pattern)
        x_poisoned = torch.clamp(x_poisoned, 0.0, 1.0)

        x_poisoned_norm = self.normalizer(x_poisoned)
        outputs = self.model(x_poisoned_norm)

        target_class = torch.full(
            (x_batch.size(0),), target_class, dtype=torch.long, device=self.device
        )
        l_ce = F.cross_entropy(outputs, target_class)
        l_mask = torch.sum(mask)

        total_loss = l_ce + self.lambda_reg * l_mask
        preds = torch.argmax(outputs, dim=1)
        asr = (preds == target_class).float().mean().item()
        return total_loss, asr

    def update_lambda(self, current_asr: float) -> None:
        self.asr_history.append(current_asr)
        if len(self.asr_history) < self.asr_history.maxlen:
            return

        mean_asr = np.mean(self.asr_history)
        if mean_asr < 0.99:
            self.lambda_reg = max(self.lambda_reg * 0.8, 1e-5)
        else:
            self.lambda_reg = min(self.lambda_reg * 1.2, 1e2)

    def run_detection(
        self, dataloader: DataLoader, target_class: int, num_iterations: int
    ) -> float:
        self.theta_m = nn.Parameter(
            torch.full(self.input_shape, -5.0, device=self.device)
        )

        self.theta_p = nn.Parameter(torch.zeros(self.input_shape, device=self.device))

        self.optimizer_m = optim.Adam([self.theta_m], lr=self.lr)
        self.optimizer_p = optim.Adam([self.theta_p], lr=self.lr)

        self.lambda_reg = 0.01
        self.asr_history.clear()
        data_iterator = iter(dataloader)

        pbar = tqdm(
            range(num_iterations), desc=f"Optimizing for Class {target_class:2d}"
        )
        for _ in pbar:
            try:
                x_batch, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                x_batch, _ = next(data_iterator)

            x_batch = x_batch.to(self.device)

            self.optimizer_m.zero_grad()
            self.optimizer_p.zero_grad()

            g_m = self.g(self.theta_m.detach())
            g_p = self.g(self.theta_p.detach())

            grad_m_sum = torch.zeros_like(self.theta_m)
            grad_p_sum = torch.zeros_like(self.theta_p)
            total_asr_m = 0.0
            total_asr_p = 0.0

            for _ in range(self.k):
                m_j = torch.bernoulli(g_m)
                loss_j, asr_j = self.calculate_loss(x_batch, target_class, m_j, g_p)
                grad_m_sum += loss_j * 2.0 * (m_j - g_m)
                total_asr_m += asr_j

            for _ in range(self.k):
                epsilon_j = torch.randn_like(self.theta_p)
                p_j = self.g(self.theta_p.detach() + self.sigma * epsilon_j)
                loss_j, asr_j = self.calculate_loss(x_batch, target_class, g_m, p_j)
                grad_p_sum += loss_j * epsilon_j
                total_asr_p += asr_j

            mean_grad_m = grad_m_sum / self.k
            mean_grad_p = grad_p_sum / (self.k * self.sigma)

            self.theta_m.grad = mean_grad_m
            self.theta_p.grad = mean_grad_p

            self.optimizer_m.step()
            self.optimizer_p.step()

            mean_asr = (total_asr_m + total_asr_p) / (2.0 * self.k)
            self.update_lambda(mean_asr)

            pbar.set_postfix(
                {"ASR": f"{mean_asr:.4f}", "Lambda": f"{self.lambda_reg:.6f}"}
            )
        final_mask = self.g(self.theta_m.detach())
        return final_mask.sum().item()
