import math
import torch

from torch import Tensor

from jaxtyping import Float, Int

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
    weight: Parameter

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
        self.weight = Parameter(
            _initialize_tensor_from_dimensions(
                out_features, in_features, dtype=dtype, device=device
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... din, dout din -> ... dout")


class Embedding(Module):
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    weight: Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = Parameter(
            _initialize_tensor_from_dimensions(
                num_embeddings, embedding_dim, dtype=dtype, device=device
            )
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(Module):
    weight: Parameter

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = Parameter(
            _initialize_tensor_from_dimension(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        return x.to(in_dtype)


class SwiGLU(Module):
    # d_ff d_model
    w1: Linear
    # d_model dff
    w2: Linear
    # dff d_model
    w3: Linear

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    # x: ... d_model
    def forward(self, x: Tensor) -> Tensor:
        w3_x = self.w3(x)
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        silu_w3_x = silu * w3_x
        return self.w2(silu_w3_x)


def _theta_i_ks(
    theta: float,
    d_k: int,
    max_seq_len: int,
    device: (torch.device | None) = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    i_s = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
    k_s = torch.arange(0, d_k, 2, device=device, dtype=dtype).unsqueeze(0)
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
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: (torch.device | None) = None,
        dtype: torch.dtype | None = None,
    ):
        """

        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        """
        super().__init__()
        sines, cosines = _theta_i_ks(theta, d_k, max_seq_len, device=device, dtype=dtype)
        self.register_buffer("sines", sines, persistent=False)
        self.register_buffer("cosines", cosines, persistent=False)
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


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    qt_k = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    qt_k_div = qt_k * torch.rsqrt(torch.tensor(d_k, device=Q.device))
    if mask is not None:
        qt_k_div.masked_fill_(~mask, float("-inf"))
    sm = softmax(qt_k_div, -1)
    return sm @ V


class MultiHeadSelfAttention(Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.

    Weights:
        q_proj (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj (Float[Tensor, "d_k d_in"]): Weights for the V projection
        output_proj (Float[Tensor, "d_model d_v"]): Weights for the output projection
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear

    d_model: int
    d_k: int
    d_v: int
    num_heads: int

    rope: RotaryPositionalEmbedding | None

    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if rope:
            assert self.d_k == rope.d_k
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Args:
            x (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
            token_positions (Float[Tensor, "... sequence_length"])

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            assert token_positions is not None
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        else:
            assert token_positions is None
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attention_output = scaled_dot_product_attention(q, k, v, causal_mask)

        combined_heads = rearrange(attention_output, "... h s d -> ... s (h d)")
        return self.output_proj(combined_heads)


class PreNormTransformer(Module):
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Weights:
        - `attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is (d_model, d_model).
        - `ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `ffn.w3.weight`
            Weight of the third linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).

    """

    attn: MultiHeadSelfAttention
    ln1: RMSNorm
    ln2: RMSNorm
    ffn: SwiGLU

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        device=None,
        dtype=None,
    ):
        """

        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
        """
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, attn_in: Tensor):
        """
        Args:
            in_features (Float[Tensor, "batch sequence_length d_model"]):

        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """
        batch_size, sequence_length, _ = attn_in.shape
        attn_norm = self.ln1(attn_in)
        token_positions = torch.arange(sequence_length, device=attn_in.device).expand(
            batch_size, -1
        )
        attn_out = self.attn(attn_norm, token_positions)
        ff_in = attn_in + attn_out
        ffn_norm = self.ln2(ff_in)
        ffn_out = self.ffn(ffn_norm)
        return ff_in + ffn_out


class TransformerLanguageModel(Module):
    """
    Implementation of the Transformer Language Model.

    Weights:
        State dict of our reference implementation. {num_layers} refers to an
        integer between `0` and `num_layers - 1` (the layer index).
        The keys of this dictionary are:
        - `token_embeddings.weight`
            Token embedding matrix. Shape is (vocab_size, d_model).
        - `layers.{num_layers}.attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is ((d_model / num_heads) * num_heads, d_model).
        - `layers.{num_layers}.ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `layers.{num_layers}.ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `layers.{num_layers}.ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `layers.{num_layers}.ffn.w3.weight`
            Weight of the third linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `layers.{num_layers}.ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ln_final.weight`
            Weights of affine transform for RMSNorm applied to the output of the final transformer block.
            Shape is (d_model, ).
        - `lm_head.weight`
            Weights of the language model output embedding.
            Shape is (vocab_size, d_model).
    """

    token_embeddings: Embedding
    layers: torch.nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear
    rope: RotaryPositionalEmbedding

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            num_layers (int): The number of Transformer layers to use.
        """
        super().__init__()
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_k, context_length, device=device, dtype=dtype
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            PreNormTransformer(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor):
        """
        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        # Embedding
        embedded_in = self.token_embeddings(in_indices.to(torch.long))
        passed_through = embedded_in
        # Transformers
        for layer in self.layers:
            passed_through = layer(passed_through)
        # Norm
        normed_output = self.ln_final(passed_through)
        return self.lm_head(normed_output)
