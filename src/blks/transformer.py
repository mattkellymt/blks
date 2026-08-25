from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from blks import RMSNorm, AdamW, Muon, Attention


def create_param(shape, dtype, device):
    param = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    param = nn.ParameterDict({'weight': param})
    return param


class MLP(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        dtype = config['dtype']
        device = config['device']

        gate_shape = (intermediate_size, hidden_size)
        up_shape = (intermediate_size, hidden_size)
        down_shape = (hidden_size, intermediate_size)

        self.gate_proj = create_param(gate_shape, dtype, device)
        self.up_proj = create_param(up_shape, dtype, device)
        self.down_proj = create_param(down_shape, dtype, device)

    def forward(self, x):
        gate = F.linear(x, self.gate_proj.weight)
        up = F.linear(x, self.up_proj.weight)
        out = F.linear(F.silu(gate) * up, self.down_proj.weight)
        return out


class Block(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        rms_norm_eps = config['rms_norm_eps']
        dtype = config['dtype']
        device = config['device']
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.self_attn = Attention(**config)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.mlp = MLP(**config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        num_hidden_layers = config['num_hidden_layers']
        tie_word_embeddings = config['tie_word_embeddings']
        rms_norm_eps = config['rms_norm_eps']
        dtype = config['dtype']
        device = config['device']

        embed_shape = (vocab_size, hidden_size)

        self.model = nn.ModuleDict({
            'embed_tokens': create_param(embed_shape, dtype, device),
            'norm': RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype),
            'layers': nn.ModuleList(Block(**config) for layer_idx in range(num_hidden_layers))
        })
        if tie_word_embeddings:
            self.lm_head = self.model['embed_tokens']
        else:
            self.lm_head = create_param(embed_shape, dtype, device)

    def forward(self, inputs):
        x = self.model.embed_tokens.weight[inputs]
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = F.linear(x, self.lm_head.weight)
        return logits

    @torch.no_grad()
    def init_params(self):
        std_val = 0.02
        mean_val = 0.0
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean_val, std_val)
            else:
                nn.init.ones_(p)
                

def save_config(config, path):
    config_dict = {k: str(v) for k, v in config.items()}
    indent_val = 2
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=indent_val)


def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def save_model(model, path):
    save_safetensors(model.state_dict(), path)


def load_model(model, path, device):
    dev_str = str(device)
    sd = load_safetensors(path, dev_str)
    model_sd = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if f"{k}.weight" in model_sd:
            new_sd[f"{k}.weight"] = v
        elif k in model_sd:
            new_sd[k] = v
    strict_flag = False
    model.load_state_dict(new_sd, strict_flag)


def train_step(model, optimizers, inputs, targets):
    for optimizer in optimizers:
        optimizer.zero_grad()
    logits = model(inputs)
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    loss_val = loss.item()
    return loss_val


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature):
    device = next(model.parameters()).device
    messages = [{'role': 'user', 'content': prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(formatted_prompt)
    prompt_tokens = torch.tensor([prompt_ids], device=device)
    prompt_len = prompt_tokens.shape[1]
    input_shape = (1, prompt_len + max_new_tokens)
    input_ids = torch.empty(input_shape, dtype=torch.long, device=device)
    input_ids[:, :prompt_len] = prompt_tokens

    for step_idx in range(max_new_tokens):
        current_len = prompt_len + step_idx
        logits = model(input_ids[:, :current_len])
        dim_val = -1
        if temperature <= 0.0:
            next_token = torch.argmax(logits[:, -1, :], dim_val)
        else:
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim_val)
            num_samples_val = 1
            next_token = torch.multinomial(probs, num_samples_val).squeeze(dim_val)
        input_ids[0, current_len] = next_token

    response_tokens = input_ids[0, prompt_len:].tolist()
    skip_special_val = True
    response_text = tokenizer.decode(response_tokens, skip_special_val)
    return response_text


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    repo_id = f"unsloth/Llama-3.2-1B-Instruct"
    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")

    max_new_tokens = 128
    temperature = 0.0
    nesterov_val = True
    eps_val = 1e-7
    ns_steps_val = 5
    betas_val = (0.9, 0.999)
    adam_eps_val = 1e-8

    config = load_config(config_path)
    config.update({
        'device': device,
        'dtype': torch.bfloat16,
        'lr': 0.01,
        'weight_decay': 0.1,
        'momentum': 0.95,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'config_path': config_path,
        'weights_path': weights_path,
    })

    model = Model(**config)
    load_model(model, weights_path, device)

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adamw_params = [p for p in model.parameters() if p.ndim != 2]

    muon = Muon(
        muon_params,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        momentum=config['momentum'],
        nesterov=nesterov_val,
        eps=eps_val,
        ns_steps=ns_steps_val,
    )
    adamw = AdamW(
        adamw_params,
        lr=config['lr'],
        betas=betas_val,
        eps=adam_eps_val,
        weight_decay=config['weight_decay'],
    )
    optimizers = [muon, adamw]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    prompt = "Explain how a transformer model uses multi-head self-attention to process text."
    print(f"Prompt: {prompt}")
    response_before = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(f"Response:\n{response_before}\n")


if __name__ == "__main__":
    main()