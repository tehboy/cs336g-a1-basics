import math
import torch

from einops import einsum, rearrange
from torch.nn import init, Module, Parameter


def _initialize_tensor_from_dimensions(
    dim_a: int,
    dim_b: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    variance = 2 / (dim_a + dim_b)
    std_dev = math.sqrt(variance)
    bounds = 3 * std_dev
    tensor = torch.empty((dim_a, dim_b), device=device, dtype=dtype)
    init.trunc_normal_(tensor, std=std_dev, a=-bounds, b=bounds)
    return tensor


class Linear(Module):
    weights: Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(
            _initialize_tensor_from_dimensions(
                out_features, in_features, dtype=dtype, device=device
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... din, dout din -> ... dout")


class Embedding(Module):
    weights: Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: (torch.device | None) = None,
        dtype: (torch.dtype | None) = None,
    ):
        super().__init__()
        self.weights = Parameter(
            _initialize_tensor_from_dimensions(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(Module):
    weights: Parameter

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weights
        return x.to(in_dtype)


class SwiGLU(Module):
    # d_ff d_model
    w1: Linear
    # d_model dff
    w2: Linear
    # dff d_model
    w3: Linear

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    # x: ... d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w3_x = self.w3(x)
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        silu_w3_x = silu * w3_x
        return self.w2(silu_w3_x)
