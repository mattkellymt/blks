import math
import torch
from blks.torch.optimizers.adam import Adam


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer implementation with Newton-Schulz orthogonalization.
    """

    def __init__(
        self,
        params,
        lr,
        weight_decay,
        momentum,
        nesterov,
        eps,
        ns_steps,
        adam_lr,
        adam_betas,
        adam_eps,
        adam_wd,
    ):
        params_list = list(params)
        muon_params = [p for p in params_list if p.ndim == 2]
        adam_params = [p for p in params_list if p.ndim != 2]
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
        )
        super().__init__(muon_params, defaults)
        self.adam = Adam(adam_params, adam_lr, adam_betas, adam_eps, adam_wd)

    def step_newton_schulz(self, update, a, b, c):
        g = update @ update.T
        g_upd = torch.addmm(g, g, g, beta=b, alpha=c)
        alpha_one = 1.0
        update_next = torch.addmm(update, g_upd, update, beta=a, alpha=alpha_one)
        return update_next

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        update = grad.bfloat16()
        is_transposed = grad.size(0) > grad.size(1)
        if is_transposed:
            update = update.T
        update.div_(update.norm().clamp(eps))
        for step_idx in range(steps):
            update = self.step_newton_schulz(update, a, b, c)
        if is_transposed:
            update = update.T
        return update

    def step_param(self, p, group):
        if p.grad is None:
            return
        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        nest = group["nesterov"]
        eps = group["eps"]
        steps = group["ns_steps"]
        grad = p.grad
        state = self.state[p]
        if "buf" not in state:
            state["buf"] = torch.zeros_like(grad)
        buf = state["buf"]
        mom_weight = 1 - mom
        buf.lerp_(grad, mom_weight)
        update = grad.lerp(buf, mom) if nest else buf
        update = self.newton_schulz(update, eps, steps)
        ratio = max(1, p.shape[0] / p.shape[1])
        adj_lr = lr * math.sqrt(ratio)
        neg_adj_lr = -adj_lr
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=neg_adj_lr)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)
        self.adam.step()
