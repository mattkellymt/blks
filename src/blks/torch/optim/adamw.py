import math
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step_param(self, p, group):
        if p.grad is None:
            return
        lr = group["lr"]
        b1, b2 = group["beta1"], group["beta2"]
        eps = group["eps"]
        wd = group["weight_decay"]
        grad = p.grad
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        t = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

        alpha_val = 1 - b1
        beta_val = 1 - b2
        exp_avg.mul_(b1).add_(grad, alpha=alpha_val)
        exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=beta_val)

        step_size = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))
        denom = exp_avg_sq.sqrt().add_(eps)

        neg_step_size = -step_size
        p.mul_(1 - lr * wd)
        p.addcdiv_(exp_avg, denom, value=neg_step_size)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)
