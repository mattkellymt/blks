import math
import torch


class Muon(torch.optim.Optimizer):
    """
    Muon (Jordan et al.)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    def newton_schulz_iter(self, ortho, a, b, c):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        return torch.addmm(ortho, gram_update, ortho, beta=a)

    def newton_schulz(self, grad, ns_coefficients, ns_steps, eps):
        a, b, c = ns_coefficients
        transpose = grad.size(0) > grad.size(1)
        ortho = grad.bfloat16()
        if transpose:
            ortho = ortho.T
        ortho = ortho.div(ortho.norm().clamp(min=eps))
        for _ in range(ns_steps):
            ortho = self.newton_schulz_iter(ortho, a, b, c)
        if transpose:
            ortho = ortho.T
        return ortho

    def adjust_lr(self, lr, adjust_lr_fn, shape):
        a, b = shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            ratio = math.sqrt(max(1, a / b))
        elif adjust_lr_fn == "match_rms_adamw":
            ratio = 0.2 * math.sqrt(max(a, b))
        else:
            ratio = 1.0
        return lr * ratio

    def step_param(self, p, group):
        if p.grad is None:
            return
        if p.ndim != 2:
            raise ValueError(
                f"Muon only supports 2D parameters; got shape {tuple(p.shape)}"
            )
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_coefficients = group["ns_coefficients"]
        eps = group["eps"]
        ns_steps = group["ns_steps"]
        adjust_lr_fn = group["adjust_lr_fn"]
        grad = p.grad
        state = self.state[p]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]

        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        update = self.newton_schulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = self.adjust_lr(lr, adjust_lr_fn, p.shape)
        p.mul_(1 - lr * weight_decay)
        p.add_(update, alpha=-adjusted_lr)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)
