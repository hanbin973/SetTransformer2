import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from typing import Any, Callable, Sequence, Optional, Union, Tuple
from jax import Array
from jax.typing import ArrayLike

from .config import ModuleConfig

Dtype = Any


class Norm(nn.Module):
    norm_type: str
    epsilon: float = 1e-6
    use_scale: bool = False
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike, mask: Optional[ArrayLike] = None) -> Array:
        
        eps_norm, use_scale, use_bias = self.epsilon, self.use_scale, self.use_bias

        if self.norm_type == "set":
            return SetNorm(epsilon=eps_norm, use_bias=use_bias, use_scale=use_scale)(x, mask)
        elif self.norm_type == "layer":
            return nn.LayerNorm(epsilon=eps_norm, use_bias=use_bias, use_scale=use_scale)(x)
        else:
            raise NotImplementedError("This normalization has not been implemented.")

class SetNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_scale: bool = False
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike, mask: Optional[ArrayLike] = None) -> Array:

        num_tokens, num_features = x.shape[1:]

        if mask is not None:
            num_tokens = jnp.maximum(jnp.sum(mask, axis=1), 1)
            mean = jnp.sum(x * mask[..., None], axis=(1, 2)) / num_tokens / num_features
            mean2 = jnp.sum(jnp.square(x) * mask[..., None], axis=(1, 2)) / num_tokens / num_features
        else:
            mean = jnp.sum(x, axis=(1, 2)) / num_tokens / num_features
            mean2 = jnp.sum(jnp.square(x), axis=(1, 2)) / num_tokens / num_features
        variance = jnp.maximum(0., mean2 - jnp.square(mean))

        normalized_x = (x - mean[:, None, None]) / jnp.sqrt(variance[:, None, None] + self.epsilon)

        return normalized_x

"""
import st2.normalization as normalization

m1 = np.sum(x * mask[..., None], axis=(1,2)) / np.sum(mask, axis=1) / 20
m2 = np.sum(x ** 2 * mask[..., None], axis=(1,2)) / np.sum(mask, axis=1) / 20
m2p = np.sum((x - m1[:, None, None]) ** 2 * mask[..., None], axis=(1,2)) / np.sum(mask, axis=1) / 20

norm = normalization.SetNorm()
params = norm.init(rng_init, x, mask)
jnp.allclose(norm.apply(params, x, mask), (x - m1[:, None, None]) / np.sqrt(m2p[:, None, None] + 1e-6), rtol=1e-2)
"""
