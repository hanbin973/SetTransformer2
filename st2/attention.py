from typing import Any, Optional

import jax
import jax.numpy as jnp

import flax
from flax.linen.linear import PrecisionLike
from flax.linen.dtypes import promote_dtype

Array = Any
Dtype = Any
PRNGKey = Any

def euclidean_attention_weights(query: Array,
                          key: Array,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          broadcast_dropout: bool = True,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: Optional[Dtype] = None,
                          precision: PrecisionLike = None) -> Array:
    """Computes Euclidean attention weights given query and key.

    The logits are computed as:
      logits = -||q - k||² / sqrt(depth),
    where the pairwise squared Euclidean distances are computed between query and key.

    Args:
      query: queries with shape `[batch..., q_length, num_heads, depth]`.
      key: keys with shape `[batch..., kv_length, num_heads, depth]`.
      bias: bias for the attention weights. Should be broadcastable to `[batch..., num_heads, q_length, kv_length]`.
      mask: mask for the attention weights. Should be broadcastable to `[batch..., num_heads, q_length, kv_length]`.
      broadcast_dropout: boolean indicating whether to use broadcasted dropout along batch dimensions.
      dropout_rng: PRNGKey to be used for dropout.
      dropout_rate: dropout rate.
      deterministic: if True, dropout is not applied.
      dtype: the computation dtype; if None, inferred from `query`.
      precision: numerical precision for dot-products.

    Returns:
      Attention weights of shape `[batch..., num_heads, q_length, kv_length]`.
    """
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    # Basic shape assertions.
    assert query.ndim == key.ndim, 'q and k must have the same rank.'
    assert query.shape[-2] == key.shape[-2], 'q and k must have the same number of heads.'
    assert query.shape[-1] == key.shape[-1], 'q and k must have the same depth.'
    
    # Compute the squared norm of the queries and keys.
    # q_norm_sq will have shape: [..., num_heads, q_length]
    q_norm_sq = jnp.einsum('...qhd,...qhd->...hq', query, query)
    # k_norm_sq will have shape: [..., num_heads, kv_length]
    k_norm_sq = jnp.einsum('...khd,...khd->...hk', key, key)
    
    # Compute the inner product between queries and keys.
    # This computes the dot product over the 'depth' dimension and produces an array
    # of shape [..., num_heads, q_length, kv_length]
    cross_term = jnp.einsum('...qhd,...khd->...hqk', query, key)
    
    # Now combine these terms using broadcasting to get the squared distances:
    # q_norm_sq[..., :, None, :] has shape [..., num_heads, q_length, 1]
    # k_norm_sq[..., None, :, :] has shape [..., num_heads, 1, kv_length]
    # Their sum minus 2 * cross_term gives shape [..., num_heads, q_length, kv_length]
    sq_distances = q_norm_sq[..., :, :, None] + k_norm_sq[..., :, None, :] - 2.0 * cross_term
    
    # Scaling factor is √depth
    depth = query.shape[-1]
    logits = - sq_distances / jnp.sqrt(depth).astype(dtype)  # shape: [..., q_length, kv_length, num_heads]

    # Apply bias and mask if provided.
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        # Where mask is False, set logits to a very large negative number.
        big_neg = jnp.finfo(dtype).min
        logits = jnp.where(mask, logits, big_neg)

    # Normalize via softmax.
    attn_weights = jax.nn.softmax(logits, axis=-1).astype(dtype)

    # Optionally apply dropout.
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # Dropout is broadcast across all dimensions except the last two.
            dropout_shape = (1,) * (key.ndim - 2) + attn_weights.shape[-2:]
        else:
            dropout_shape = attn_weights.shape
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def euclidean_attention(query: Array,
                  key: Array,
                  value: Array,
                  bias: Optional[Array] = None,
                  mask: Optional[Array] = None,
                  broadcast_dropout: bool = True,
                  dropout_rng: Optional[PRNGKey] = None,
                  dropout_rate: float = 0.,
                  deterministic: bool = False,
                  dtype: Optional[Dtype] = None,
                  precision: PrecisionLike = None) -> Array:
    """Computes Euclidean attention given query, key, and value.

    The attention weights are computed based on an RBF kernel applied to the pairwise squared Euclidean
    distances between query and key, and then the values are aggregated using these weights.

    Args:
      query: queries with shape `[batch..., q_length, num_heads, depth]`.
      key: keys with shape `[batch..., kv_length, num_heads, depth]`.
      value: values with shape `[batch..., kv_length, num_heads, v_depth]`.
      bias: bias for the attention weights. Should be broadcastable to `[batch..., num_heads, q_length, kv_length]`.
      mask: mask for the attention weights. Should be broadcastable to `[batch..., num_heads, q_length, kv_length]`.
      broadcast_dropout: boolean indicating whether to use broadcasted dropout along batch dimensions.
      dropout_rng: PRNGKey to be used for dropout.
      dropout_rate: dropout rate.
      deterministic: if True, dropout is not applied.
      dtype: the computation dtype; if None, inferred from `query`.
      precision: numerical precision for the computation.

    Returns:
      The attention output of shape `[batch..., q_length, num_heads, v_depth]`.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype

    attn_weights = euclidean_attention_weights(query, key, bias, mask,
                                         broadcast_dropout, dropout_rng,
                                         dropout_rate, deterministic,
                                         dtype, precision)
    # Aggregate the value vectors.
    # This einsum uses the convention: '...hqk,...khd -> ...qhd', where the attention weights have shape
    # [batch..., num_heads, q_length, kv_length] and the values have shape [batch..., kv_length, num_heads, v_depth].
    output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value, precision=precision)
    return output