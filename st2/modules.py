import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from typing import Any, Callable, Sequence, Optional, Union
from jax import Array
from jax.typing import ArrayLike

from .config import ModuleConfig
from .normalization import Norm

# Embedding
class HiddenPadding(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            ) -> Array:

        num_batches, num_samples, num_orig = x.shape
        num_hidden = self.config.num_hidden
        padding = jnp.zeros((num_batches, num_samples, num_hidden - num_orig))
        
        return jnp.concatenate((x , padding), axis=-1)
        
# SetTransformer modules
class MAB(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            y: ArrayLike,
            mask: Optional[ArrayLike] = None,
            ) -> Array:

        num_batches, num_tokens = x.shape[0], x.shape[1]
        num_hidden, num_attn_heads = self.config.num_hidden, self.config.num_attn_heads 
        attn_fn, act_fn, norm_type = self.config.attn_fn, self.config.act_fn, self.config.norm_type
        eps_norm, use_bias = self.config.eps_norm, self.config.use_bias
        ln_attn, ln_ffn = self.config.ln_attn, self.config.ln_ffn

        # attention 
        attn_head_dim = num_hidden // num_attn_heads
        q = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(x)
        k = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(y)
        v = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(y)

        attn = attn_fn(q, k ,v, mask=mask[:, None, None, :])
        h = x + attn.reshape(num_batches, num_tokens, num_hidden)
        h = Norm(norm_type, epsilon=eps_norm, use_bias=False, use_scale=False)(h) if ln_attn else h

        # feed-forward
        h = h + act_fn(nn.Dense(num_hidden, use_bias=use_bias)(h))
        h = Norm(norm_type, epsilon=eps_norm, use_bias=False, use_scale=False)(h) if ln_ffn else h

        return h

class MAB2(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            y: ArrayLike,
            mask: Optional[ArrayLike] = None,
            ) -> Array:
        
        num_batches, num_tokens = x.shape[0], x.shape[1]
        num_hidden, num_attn_heads = self.config.num_hidden, self.config.num_attn_heads 
        attn_fn, act_fn, norm_type= self.config.attn_fn, self.config.act_fn, self.config.norm_type
        eps_norm, use_bias = self.config.eps_norm, self.config.use_bias
        ln_query, ln_key, ln_ffn = self.config.ln_query, self.config.ln_key, self.config.ln_ffn

        # normalization
        q = Norm(norm_type, epsilon=eps_norm, use_bias=False, use_scale=False)(x, mask) if ln_query else x
        k = Norm(norm_type, epsilon=eps_norm, use_bias=False, use_scale=False)(y, mask) if ln_key else y

        # attention 
        attn_head_dim = num_hidden // num_attn_heads
        q = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(q) 
        k = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(k)
        v = nn.DenseGeneral(features = (num_attn_heads, attn_head_dim), use_bias=use_bias)(y)

        attn = attn_fn(q, k ,v, mask=mask[:, None, None, :])
        h = x + attn.reshape(num_batches, num_tokens, num_hidden)

        # feed-forward
        r = Norm(norm_type, epsilon=eps_norm, use_bias=False, use_scale=False)(h, mask) if ln_ffn else h
        h = h + act_fn(nn.Dense(num_hidden, use_bias=use_bias)(r))

        return h

class SAB(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            ) -> Array:

        return MAB(self.config)(x, x, mask)    

class SAB2(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            ) -> Array:

        return MAB2(self.config)(x, x, mask)    

class ISAB(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            scale: Optional[ArrayLike] = None,
            ) -> Array:

        num_hidden, num_induced = self.config.num_hidden, self.config.num_induced  
        num_batches = x.shape[0]

        i = self.param("induced_points", nn.initializers.xavier_uniform(), (1, num_induced, num_hidden))
        i = jnp.repeat(i, num_batches, axis=0).reshape(num_batches, num_induced, num_hidden)
        if scale is not None:
            i *= scale[:, None, None]

        h = MAB(self.config)(i, x, mask=mask)
        h = MAB(self.config)(x, h)
        return h

class ISAB2(nn.Module):
    config1: ModuleConfig
    config2: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            scale: Optional[ArrayLike] = None,
            ) -> Array:

        num_batches = x.shape[0]
        num_hidden, num_induced = self.config.num_hidden, self.config.num_induced  

        i = self.param("induced_points", nn.initializers.xavier_uniform(), (1, num_induced, num_hidden))
        i = jnp.repeat(i, num_batches, axis=0).reshape(num_batches, num_induced, num_hidden)
        if scale is not None:
            i *= scale[:, None, None]

        h = MAB2(self.config1)(i, x, mask=mask)
        h = MAB2(self.config2)(x, h)
        return h
        
class PMA(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            scale: Optional[ArrayLike] = None,
            ) -> Array:
        
        num_batches = x.shape[0]
        num_hidden, num_seeds = self.config.num_hidden, self.config.num_seeds
        use_scale = self.config.use_scale

        s = self.param("seeds", nn.initializers.xavier_uniform(), (1, num_seeds, num_hidden))
        s = jnp.repeat(s, num_batches, axis=0).reshape(num_batches, num_seeds, num_hidden)
        if scale is not None:
            s *= scale[:, None, None]

        x = MAB(self.config)(s, x, mask=mask)
        return x

class PMA2(nn.Module):
    config: ModuleConfig

    @nn.compact
    def __call__(
            self,
            x: ArrayLike,
            mask: Optional[ArrayLike] = None,
            scale: Optional[ArrayLike] = None,
            ) -> Array:
        
        num_batches = x.shape[0]
        num_hidden, num_seeds = self.config.num_hidden, self.config.num_seeds
        use_scale = self.config.use_scale

        s = self.param("seeds", nn.initializers.xavier_uniform(), (1, num_seeds, num_hidden))
        s = jnp.repeat(s, num_batches, axis=0).reshape(num_batches, num_seeds, num_hidden)
        if scale is not None:
            s *= scale[:, None, None]

        x = MAB2(self.config)(s, x, mask=mask)
        return x
       
# Test functions
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
