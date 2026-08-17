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
        return F.layer_norm(
            x, self.shape, self.weight, self.bias, self.eps
        )
