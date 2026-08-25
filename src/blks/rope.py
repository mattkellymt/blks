import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(
        self,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        half_dim = head_dim // 2

        exponents = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (self.theta ** exponents)
        seq_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        angles = torch.outer(seq_idx, inv_freq)

        emb = torch.cat((angles, angles), -1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)

        pos_x1 = +x[..., :half_dim]
        neg_x2 = -x[..., half_dim:]
        rotate_half = torch.cat((neg_x2, pos_x1), -1)
        out = (x * cos) + (rotate_half * sin)
        return out
