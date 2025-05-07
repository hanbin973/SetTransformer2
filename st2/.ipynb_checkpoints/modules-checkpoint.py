import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from typing import Any, Callable, Sequence, Optional, Union
from jax import Array
from jax.typing import ArrayLike

from .attention import euclidean_attention

class HiddenPadding(nn.Module):
    num_hidden: int

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        num_batches, num_samples, num_orig = x.shape
        padding = jnp.zeros((num_batches, num_samples, self.num_hidden - num_orig))
        return jnp.concatenate((x , padding), axis=-1)
        
class Embedding(nn.Module):
    num_hidden: int
    
    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        return nn.Dense(self.num_hidden)(x)

class MAB2(nn.Module):
    num_hidden: int
    num_attn_heads: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            y: ArrayLike,
            mask: ArrayLike = None,
            ) -> Array:

        num_batches, num_tokens = x.shape[0], x.shape[1]

        # attention 
        attn_head_dim = self.num_hidden // self.num_attn_heads

        q = nn.DenseGeneral(features = (self.num_attn_heads, attn_head_dim))(x)
        k = nn.DenseGeneral(features = (self.num_attn_heads, attn_head_dim))(y)
        v = nn.DenseGeneral(features = (self.num_attn_heads, attn_head_dim))(y)

        if self.layer_norm:
            attn = nn.dot_product_attention(nn.LayerNorm()(q), nn.LayerNorm()(k), v, mask=mask)
        else:
            attn = nn.dot_product_attention(q, nn.LayerNorm()(k), v, mask=mask)
        h = x + attn.reshape(num_batches, num_tokens, self.num_hidden)

        # feed-forward
        h = h + nn.relu(nn.Dense(self.num_hidden)(nn.LayerNorm()(h)))

        return h

class SAB2(nn.Module):
    num_hidden: int
    num_attn_heads: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: ArrayLike = None,
            ) -> Array:
        return MAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(x, x, mask)    

class ISAB2(nn.Module):
    num_hidden: int
    num_attn_heads: int
    num_induced: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        num_batches = x.shape[0]

        i = self.param("induced_points", nn.initializers.xavier_uniform(), (1, self.num_induced, self.num_hidden))
        i = jnp.repeat(i, num_batches, axis=0).reshape(num_batches, self.num_induced, self.num_hidden)

        h = MAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(i, x)
        h = MAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(x, h)
        return h
        
class PMA(nn.Module):
    num_hidden: int
    num_attn_heads: int
    num_seeds: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        num_batches = x.shape[0]

        s = self.param("seeds", nn.initializers.xavier_uniform(), (1, self.num_seeds, self.num_hidden))
        s = jnp.repeat(s, num_batches, axis=0).reshape(num_batches, self.num_seeds, self.num_hidden)

        x = MAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(s, x)
        return x

class LinearEstimator(nn.Module):
    num_seeds: int

    @nn.compact
    def __call__(
        self,
        x: ArrayLike
        ) -> Array:
        num_samples = x.shape[1]
        s = self.param("seeds", nn.initializers.xavier_uniform(), (num_samples, self.num_seeds))
        return jnp.einsum("np,bnj->bpj", s, x)

class SetEncoder2(nn.Module):
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        for _ in range(self.num_layers):
            x = SAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(x)
        return x

class InducedSetEncoder2(nn.Module):
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    num_induced: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        for _ in range(self.num_layers):
            x = ISAB2(self.num_hidden, self.num_attn_heads, self.num_induced, self.layer_norm)(x)
        return x

class SetAggregator(nn.Module):
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    num_seeds: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        x = PMA(self.num_hidden, self.num_attn_heads, self.num_seeds, self.layer_norm)(x)
        for _ in range(self.num_layers):
            x = SAB2(self.num_hidden, self.num_attn_heads, self.layer_norm)(x)
        return x

class SetTransformer2(nn.Module):
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    num_agg_layers: int
    num_seeds: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        x = SetEncoder2(self.num_layers, self.num_hidden, self.num_attn_heads, self.layer_norm)(x)
        x = SetAggregator(self.num_agg_layers, self.num_hidden, self.num_attn_heads, self.num_seeds, self.layer_norm)(x)
        return x

class InducedSetTransformer2(nn.Module):
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    num_induced: int
    num_agg_layers: int
    num_seeds: int
    layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x: ArrayLike
            ) -> Array:
        x = InducedSetEncoder2(self.num_layers, self.num_hidden, self.num_attn_heads, self.num_induced, self.layer_norm)(x)
        x = SetAggregator(self.num_agg_layers, self.num_hidden, self.num_attn_heads, self.num_seeds, self.layer_norm)(x)
        return x
        

if __name__ == "__main__":
    num_layers, num_agg_layers = 3, 2
    num_batches, num_tokens, num_hidden, num_attn_heads = 100, 1000, 256, 8
    num_induced, num_seeds = 50, 96
    
    # define model
    model = InducedSetEncoder2(num_layers, num_hidden, num_attn_heads, num_induced)
    agg = SetAggregator(num_agg_layers, num_hidden, num_attn_heads, num_seeds)

    # set random key
    rng = jax.random.PRNGKey(0)
    rng, rng_data, rng_init, rng_permute = jax.random.split(rng, 4)

    # generate data
    inputs = jax.random.normal(rng_data, (num_batches, num_tokens, num_hidden)) 
    
    # init model
    print("Initialize induced point set transformer")
    params = model.init(rng_init, inputs)

    # shuffle data
    idx_permute = jax.random.permutation(rng_permute, num_tokens) 
    inputs_permute = inputs[:, idx_permute, :]
    print(idx_permute)

    # compare output
    print("Check equivariance of induced point set transformer")
    func = jax.jit(model.apply)
    output1, output2 = func(params, inputs), func(params, inputs_permute)
    print(jnp.corrcoef(output1[:, idx_permute, :].ravel(), output2.ravel()))
    print(output1[:, idx_permute, :][0])
    print("\n")
    print(output2[0])
    print("\n")

    # init model
    params_agg = agg.init(rng_init, output1)

    # check output1 != output2
    print("Check output1 and 2 are different pre-permutation matching")
    print(output1[0])
    print("\n")
    print(output2[0])
    
    # compare output
    func_agg = jax.jit(agg.apply)
    agg1, agg2 = func_agg(params_agg, output1), func_agg(params_agg, output2)
    print(jnp.corrcoef(agg1.ravel(), agg2.ravel()))
    print(agg1.ravel())
    print("\n")
    print(agg2.ravel())
