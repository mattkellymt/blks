import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float | None = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.ones(normalized_shape, device=device, dtype=dtype)
            self.weight = nn.Parameter(self.weight)
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out
