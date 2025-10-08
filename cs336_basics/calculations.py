# Some back of the envelope weight formulas.
BYTES_PER_WORD = 4

def weight(words):
    return words * BYTES_PER_WORD


def embedding_weight(d_m, d_v):
    return d_m * d_v


def ff_weight(d_ff, d_m):
    return 3 * d_ff * d_m


def mhsa_weight(d_m):
    return 4 * d_m * d_m


def norm_weight(d_m):
    return d_m


def transformer_block_weights(d_ff, d_m):
    return ff_weight(d_ff, d_m) + 2 * norm_weight(d_m) + mhsa_weight(d_m)

def rope_cache(d_m, context_length, heads):
    return context_length * (d_m // heads)

def transformer_model_weights(d_ff, d_m, d_v, context_length, heads, layers):
    return (
        2 * embedding_weight(d_m, d_v)
        + norm_weight(d_m)
        + layers * transformer_block_weights(d_ff, d_m)
        + rope_cache(d_m, context_length, heads)
    )

def transformer_input_size(batch_size, context_length, d_m):
    return batch_size * context_length * d_m

def input_size(batch_size, context_length):
    return batch_size * context_length

def matmul_flops(i, j, k):
    return 2 * i * j * k


def ff_flops(d_ff, d_m):
    return 3 * d_ff * d_m

def transformer_block_flops():
    return ff_flops(d_ff, d_m) + 2 * norm_flops(d_m) + mhsa_flops(d_m)
