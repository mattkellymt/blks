import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def init_state(self, state, p, amsgrad):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(p)

    def denom(self, state, amsgrad, bias_correction2_sqrt, eps):
        exp_avg_sq = state["exp_avg_sq"]
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            exp_avg_sq = max_exp_avg_sq
        return (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    def step_param(self, p, group):
        if p.grad is None:
            return
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        grad = p.grad
        state = self.state[p]

        if "step" not in state:
            self.init_state(state, p, amsgrad)

        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Decoupled weight decay: shrink the parameter directly, not via the gradient.
        p.mul_(1 - lr * weight_decay)

        # Exponential moving averages of the gradient and its square.
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        step_size = lr / bias_correction1

        denom = self.denom(state, amsgrad, bias_correction2_sqrt, eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)
