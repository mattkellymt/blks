import torch
import torch.nn as nn
import torch.nn.functional as F

from blks.rope import Rope


def create_param(shape, dtype, device):
    param = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    param = nn.ParameterDict({'weight': param})
    return param


class Attention(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        num_attention_heads = config['num_attention_heads']
        num_key_value_heads = config['num_key_value_heads']
        head_dim = config['head_dim']
        rope_theta = config['rope_theta']
        dtype = config['dtype']
        device = config['device']

        expected_kv_dim = num_key_value_heads * head_dim
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.q_dim = num_attention_heads * head_dim
        self.rope_theta = rope_theta
        self.dtype = dtype
        self.device = device

        q_shape = (self.q_dim, hidden_size)
        kv_shape = (expected_kv_dim, hidden_size)
        o_shape = (hidden_size, self.q_dim)

        self.q_proj = create_param(q_shape, dtype, device)
        self.k_proj = create_param(kv_shape, dtype, device)
        self.v_proj = create_param(kv_shape, dtype, device)
        self.o_proj = create_param(o_shape, dtype, device)
        self.rope = Rope(theta=rope_theta)

    def gqa(self, q, k, v):
        batch_size, num_heads, seq_len, head_dim = q.shape
        n_rep = self.num_attention_heads // self.num_key_value_heads
        q_gqa = q.view(batch_size, self.num_key_value_heads, n_rep, seq_len, self.head_dim)
        k_gqa = k.unsqueeze(2)
        v_gqa = v.unsqueeze(2)
        attn_mask = None
        dropout_p = 0.0
        is_causal = True
        attn_out = F.scaled_dot_product_attention(q_gqa, k_gqa, v_gqa, attn_mask, dropout_p, is_causal)
        out = attn_out.reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)
        return out

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        q = F.linear(x, self.q_proj.weight).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, self.k_proj.weight).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, self.v_proj.weight).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        out = self.gqa(q, k, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.q_dim)
        out = F.linear(out, self.o_proj.weight)
        return out
