import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self,
        shape: int | list[int] | tuple[int, ...],
        eps: float | None = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.ones(shape, device=device, dtype=dtype)
            self.weight = nn.Parameter(self.weight)
            if bias:
                self.bias = torch.zeros(shape, device=device, dtype=dtype)
                self.bias = nn.Parameter(self.bias)
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)

        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
            
        return out
