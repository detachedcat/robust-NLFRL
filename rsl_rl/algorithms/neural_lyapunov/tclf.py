from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rsl_rl.algorithms.neural_lyapunov.utils import bound_loss, build_nn


class InputAmplifierBase:
    """Optional input amplifier for small Lyapunov state changes."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TwinControlLyapunovFunction(nn.Module):
    """Twin Control Lyapunov Function (LF + LQF)."""

    def __init__(
        self,
        lf_structure: List[int],
        lqf_structure: List[int],
        ub: float,
        sink: List[float],
        input_amplifier: Optional[InputAmplifierBase] = None,
        lie_derivative_upper: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.lf = build_nn(lf_structure)
        self.lqf = build_nn(lqf_structure)
        self.sink = torch.tensor(sink, dtype=torch.float32, device=device)
        self.ub = ub
        self.input_amplifier = input_amplifier
        self.lie_derivative_upper = lie_derivative_upper
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_lf(x), self.forward_lqf(x, a)

    def forward_lf(self, s: torch.Tensor) -> torch.Tensor:
        inpt = self.input_amplifier(s) if self.input_amplifier is not None else s
        return self.lf(inpt)

    def forward_lqf(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inpt = self.input_amplifier(s) if self.input_amplifier is not None else s
        return self.lqf(torch.cat([inpt, a], dim=-1))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        inpt = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.forward_lf(inpt)
        return v.cpu().detach().numpy()

    def loss(
        self,
        s0: torch.Tensor,
        a: torch.Tensor,
        s1: torch.Tensor,
        current_q: torch.Tensor,
        not_done: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            if mask is None:
                return x.mean()
            w = mask.float()
            if w.dim() < x.dim():
                for _ in range(x.dim() - w.dim()):
                    w = w.unsqueeze(-1)
            denom = w.sum().clamp_min(1.0)
            return (x * w).sum() / denom

        lf_v0, lqf_v0 = self.forward(s0, a)
        lf_v1 = self.forward_lf(s1)

        lqf_loss = _masked_mean((lqf_v0 - lf_v1.detach()).pow(2), not_done)

        alpha = self.lie_derivative_upper
        desired_decay = alpha * lf_v0
        violation = F.relu(lf_v1 - (lf_v0 - desired_decay))
        lie_der = violation

        if current_q.dim() == 2:
            lie_der_weight = (current_q - current_q.min(dim=1, keepdim=True)[0]) / (
                current_q.max(dim=1, keepdim=True)[0] - current_q.min(dim=1, keepdim=True)[0] + 1e-8
            )
            lie_der_weight = lie_der_weight.mean(dim=1)
        else:
            lie_der_weight = (current_q - current_q.min()) / (current_q.max() - current_q.min() + 1e-8)

        if lie_der.dim() > 1:
            lie_der = lie_der.mean(dim=1)

        lie_der_loss = _masked_mean(lie_der * lie_der_weight, not_done)
        sink_loss = torch.abs(self.forward_lf(self.sink.view(-1, self.sink.shape[-1]))).mean()

        lf_bound_loss0 = _masked_mean(bound_loss(lf_v0, 0, self.ub), not_done)
        lf_bound_loss1 = _masked_mean(bound_loss(lf_v1, 0, self.ub), not_done)
        lqf_bound_loss = _masked_mean(bound_loss(lqf_v0, 0, self.ub), not_done)
        two_sides_bound_loss = lf_bound_loss0 + lf_bound_loss1 + lqf_bound_loss

        loss_sum = lqf_loss + lie_der_loss + sink_loss + two_sides_bound_loss
        return {
            "loss_sum": loss_sum,
            "lqf_loss": lqf_loss,
            "lie_der_loss": lie_der_loss,
            "sink_loss": sink_loss,
            "two_sides_bound_loss": two_sides_bound_loss,
        }
