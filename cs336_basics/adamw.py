from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params, 
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t += 1
                
                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data = (1 - lr * weight_decay) * p.data

                state["t"] = t
                state["m"] = m
                state["v"] = v
        
        return loss


