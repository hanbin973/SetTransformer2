import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from typing import Any, Callable, Sequence, Optional, Union
from jax import Array
from jax.typing import ArrayLike

@flax.struct.dataclass
class ModuleConfig:
    num_hidden: int
    num_attn_heads: int
    attn_fn: Optional[Callable] = nn.dot_product_attention
    act_fn: Optional[Callable] = nn.relu
    num_induced: Optional[int] = None
    num_seeds: Optional[int] = None
    eps_norm: Optional[float] = 1e-6
    norm_type: Optional[float] = "layer" 
    ln_attn: Optional[bool] = False
    ln_query: Optional[bool] = False
    ln_key: Optional[bool] = False
    ln_ffn: Optional[bool] = False
    use_bias: Optional[bool] = True
    use_scale: Optional[bool] = False

@flax.struct.dataclass
class ModelConfig:
    embedder_module: nn.Module
    module: nn.Module
    num_layers: int
    num_hidden: int
    num_attn_heads: int
    attn_fn: Optional[Callable] = nn.dot_product_attention
    num_induced: Optional[int] = None
    num_seeds: Optional[int] = None
    eps_norm: Optional[float] = 1e-6
    layer_norm: Optional[bool] = False
    layer_norm_ffn: Optional[bool] = False
    act_fn: Optional[Callable] = nn.relu
    use_bias: Optional[bool] = True
    use_scale: Optional[bool] = False

