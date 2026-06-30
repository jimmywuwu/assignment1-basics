import torch
from torch import nn
import math
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, Optional
import numpy.typing as npt
import numpy as np
import os


class MyLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2.0 / (in_features + out_features))

        torch.nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
    
    def forward(self, x:torch.Tensor):
        return torch.einsum("...i,oi->...o", x, self.weights)


class MyEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        self.weights = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        
        std = 1
        torch.nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=1,
            a=-3 * std,
            b=3 * std,
        )

    
    def forward(self, x:torch.Tensor):
        return self.weights[x]

class MyRMSNorm(torch.nn.Module):

    def __init__(self, d_model:int , eps:float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )
    
    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        rms_norm = x/rms * self.gain
        
        return rms_norm.to(in_dtype)

class MySiLU(torch.nn.Module):

    def __init__(self, device=None, dtype=None):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        return x * torch.sigmoid(x)


class MySwiGLU(torch.nn.Module):

    def __init__(self, d_model:int ,d_ff:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.silu = MySiLU()
        self.w1 = nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
            )
        )
        self.w3 = nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )
    
        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.w1, self.w2, self.w3]:
            out_features, in_features = weight.shape

            std = math.sqrt(
                2.0 / (in_features + out_features)
            )

            torch.nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
    
    def forward(self, x:torch.Tensor):
        gate = self.silu.forward(torch.einsum("fd,btd->btf",self.w1, x))
        linear = torch.einsum("fd,btd->btf",self.w3, x)
        return torch.einsum("df,btf->btd",self.w2, gate * linear)

class MyRoPE(nn.Module):

    def __init__(self, theta:float, d_k: int, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.idx = torch.arange(0, self.d_k, 2)
        inv_freq = self.theta ** (-self.idx / self.d_k)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x, token_position):
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        angle = token_position[:, None] * self.inv_freq[None, :]
        cos = angle.cos()
        sin = angle.sin()
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        y = torch.empty_like(x)
        y[..., ::2] = y_even
        y[..., 1::2] = y_odd
        return y


def my_softmax(x, i):
    max_val = torch.max(x)
    return torch.exp(x-max_val)/torch.sum(torch.exp(x-max_val),dim=i, keepdim=True)


def my_scale_dot_product_attention(Q, K ,V, mask):
    scaled_dot_product = torch.einsum("bsnd,bsmd->bsnm", Q,K)/math.sqrt(Q.shape[-1])
    masked = scaled_dot_product.masked_fill(~mask, float("-inf"))
    softmax_res = my_softmax(masked,-1)
    return torch.einsum("bsnm,bsmd->bsnd", softmax_res, V)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Wq = MyLinear(d_model, d_model)
        self.Wk = MyLinear(d_model, d_model)
        self.Wv = MyLinear(d_model, d_model)
        self.Wo = MyLinear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        H = self.num_heads

        q = self.Wq(x).view(B, S, H, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, S, H, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, S, H, self.d_k).transpose(1, 2)
        mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))[None, None, :, :]
        out = my_scale_dot_product_attention(q, k, v, mask)
        out = out.transpose(1,2).contiguous()
        out = out.view(B,S,D)
        return self.Wo(out)

class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len,theta=10000.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Wq = MyLinear(d_model, d_model)
        self.Wk = MyLinear(d_model, d_model)
        self.Wv = MyLinear(d_model, d_model)
        self.Wo = MyLinear(d_model, d_model)

        self.rope = MyRoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x):
        B, S, D = x.shape
        H = self.num_heads

        q = self.Wq(x).view(B, S, H, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, S, H, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, S, H, self.d_k).transpose(1, 2)
        
        positions = torch.arange(S, device=x.device)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        mask = torch.tril(
            torch.ones(S, S, device=x.device, dtype=torch.bool)
        )[None, None, :, :]

        out = my_scale_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, S, D)
        return self.Wo(out)

class MyTransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ffn = MySwiGLU(d_model, d_ff)
        self.ln1 = MyRMSNorm(d_model)
        self.ln2 = MyRMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len,theta)

    
    def forward(self, x):
        y_hat = x+ self.attn(self.ln1(x))
        y_hat = y_hat + self.ffn(self.ln2(y_hat))
        return y_hat

class MyLLM(nn.Module):

    def __init__(self, vocab_size, d_model, context_length, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.embedding = MyEmbedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            MyTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            )
            for _ in range(num_layers)
        ])

        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = MyLinear(d_model, vocab_size)
    
    def forward(self, input_ids):

        x = self.embedding(input_ids) 
        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits



def my_cross_entropy(inputs, targets):

    B = inputs.shape[0]

    loss = (
        torch.logsumexp(
            inputs,
            dim=1,
        )
        -
        inputs[
            torch.arange(B),
            targets,
        ]
    )

    return loss.mean()



class MyAdamw(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0001, weight_decay=0, eps=0, betas=(0.9,0.99)):
        
        defaults = { "lr":lr, "eps":eps, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1 = group['betas'][0]
            beta_2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']

            for pa in group["params"]:
                if pa.grad is None:
                    continue
                
                g = pa.grad.data # Get the gradient of loss with respect to p.
                
                state = self.state[pa] # Get state associated with p.
                t = state.get("t", 1)
                lr_t = lr * math.sqrt((1 - beta_2** t)) / (1 - beta_1**t)
                
                pa.data = pa.data - lr * weight_decay * pa.data
                m = state.get("m", torch.zeros_like(pa)) 
                v = state.get("v", torch.zeros_like(pa))
                
                m.data = beta_1*m.data + (1 - beta_1) * g
                v.data = beta_2*v.data + (1 - beta_2) * g**2
                pa.data = pa.data - lr_t * m.data / (torch.sqrt(v.data) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
    
def my_lr_cosine_learning_schedule(t, lr_max, lr_min, Tw, Tc):
    if t < Tw:
        return t/Tw*lr_max
    elif Tw<=t and t <= Tc:
        return lr_min + 0.5 *(1+math.cos((t-Tw)/(Tc-Tw)*math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

def my_gradient_clipping(parameters, max_l2_norm):
    parameters = list(parameters)

    grads = [
        p.grad
        for p in parameters
        if p.grad is not None
    ]

    if len(grads) == 0:
        return

    total_norm = torch.sqrt(
        sum(
            g.pow(2).sum()
            for g in grads
        )
    )

    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

def my_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    idx = starts[:, None] + np.arange(context_length)

    x = torch.as_tensor(
        dataset[idx],
        dtype=torch.long,
        device=device,
    )

    y = torch.as_tensor(
        dataset[idx + 1],
        dtype=torch.long,
        device=device,
    )
    return x, y

def my_save_checkpoint(model: torch.nn.Module,optimizer: torch.optim.Optimizer,iteration: int,out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(
        checkpoint,
        out,
    )

def my_load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],model: torch.nn.Module,optimizer: torch.optim.Optimizer,) -> int:

    checkpoint = torch.load(src)

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    return checkpoint["iteration"]
  