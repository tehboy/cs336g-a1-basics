import math
import torch

from torch import Tensor

from jaxtyping import Float

from einops import einsum, rearrange
from torch.nn import init, Module, Parameter


def _initialize_tensor_from_dimension(
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    variance = 1 / dim
    std_dev = math.sqrt(variance)
    bounds = 3 * std_dev
    tensor = torch.empty((dim,), device=device, dtype=dtype)
    init.trunc_normal_(tensor, std=std_dev, a=-bounds, b=bounds)
    return tensor


def _initialize_tensor_from_dimensions(
    dim_a: int,
    dim_b: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weights[token_ids]


class RMSNorm(Module):
    weights: Parameter

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = Parameter(
            _initialize_tensor_from_dimension(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
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
    def forward(self, x: Tensor) -> Tensor:
        w3_x = self.w3(x)
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        silu_w3_x = silu * w3_x
        return self.w2(silu_w3_x)


def _theta_i_ks(
    theta: float, d_k: int, max_seq_len: int, device: (torch.device | None) = None
) -> tuple[Tensor, Tensor]:
    i_s = torch.arange(max_seq_len, device=device).unsqueeze(1)
    k_s = torch.arange(0, d_k, 2, device=device).unsqueeze(0)
    theta_t_k = torch.pow(theta, k_s / d_k)
    theta_i_k = i_s / theta_t_k
    return torch.sin(theta_i_k), torch.cos(theta_i_k)


class RotaryPositionalEmbedding(Module):
    """
    Run RoPE for a given input tensor.
    """

    sines: Tensor
    cosines: Tensor
    d_k: int

    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: (torch.device | None) = None
    ):
        """

        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        """
        super().__init__()
        sines, cosines = _theta_i_ks(theta, d_k, max_seq_len, device=device)
        self.register_buffer("sines", sines)
        self.register_buffer("cosines", cosines)
        self.d_k = d_k

    def forward(self, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
        """
        Args:
            in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        sines = self.sines[token_positions]
        cosines = self.cosines[token_positions]

        x1 = in_query_or_key[..., ::2]
        x2 = in_query_or_key[..., 1::2]

        rotated_x1 = x1 * cosines - x2 * sines
        rotated_x2 = x2 * cosines + x1 * sines

        out = rearrange(torch.stack((rotated_x1, rotated_x2), dim=-1), "... d pair -> ... (d pair)")
        return out


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # softmax(v_i) is exp(v_i - max(v)) / sum(exp(v - max(v)))
    max_values = in_features.max(dim=dim, keepdim=True)[0]
    max_adjusted = in_features - max_values
    exps = max_adjusted.exp()
    sum_exps = exps.sum(dim=dim, keepdim=True)
    return exps / sum_exps
