import torch
from torch import nn
import math


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
        return torch.einsum("bsi,oi->bso", x, self.weights)


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
            torch.empty(
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
    
    def forward(self, x:torch.Tensor):
        gate = self.silu.forward(torch.einsum("fd,btd->btf",self.w1, x))
        linear = torch.einsum("fd,btd->btf",self.w3, x)
        return torch.einsum("df,btf->btd",self.w2, gate * linear)