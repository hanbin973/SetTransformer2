import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state

from typing import Any, Callable, Sequence, Optional, Union, Tuple
from jax import Array
from jax.typing import ArrayLike

import st2.modules as modules
from st2.config import ModuleConfig

# Training functions
def simulate_max_regression(
        num_batches: int,
        num_elements: int,
        low: float = 0.,
        high: float = 10.,
        bigneg: Optional[float] = -1e5
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_mask = .5 + .5 * np.random.rand(num_batches)
    mask_ = np.random.rand(num_batches, num_elements) < p_mask[:, None]
    set_ = low + (high - low) * np.random.rand(num_batches, num_elements)
    max_ = np.where(mask_, set_, bigneg).max(axis=1, keepdims=True)
    return set_[..., None], max_[..., None], mask_

@jax.jit
def train_step(state: train_state, set_: np.ndarray, max_: np.ndarray, mask_: np.ndarray):
    
    def loss_fn(params, set_, max_, mask_):
        out = state.apply_fn(params, set_, mask_)
        not_empty = mask_.sum(axis=-1, keepdims=True) > 0
        return jnp.mean(jnp.abs(max_ - out) * not_empty[:, None])

    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(state.params, set_, max_, mask_)
    state = state.apply_gradients(grads=grads)
    
    return state, loss

def train_model(
        state: train_state, 
        num_batches: int,
        num_elements: int,
        low: float = 0.,
        high: float = 10.,
        bigneg: Optional[float] = -1e5,
        num_epochs: Optional[int] = 10000, 
        patience: Optional[int] = 100
        ):

    best_loss = np.inf
    epochs_wo_improvement = 0

    for epoch in range(num_epochs):
        # simulate data

        set_, max_, mask_ = simulate_max_regression(num_batches, num_elements, low, high, bigneg)
        # train step
        state, loss = train_step(state, set_, max_, mask_)
        if loss < best_loss:
            best_loss = loss
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

        # early stopping
        if epochs_wo_improvement > patience:
            break

    return state, loss, epoch

# Model class
class Max(nn.Module):
    cfg: ModuleConfig
    cfg_agg: ModuleConfig

    @nn.compact
    def __call__(self, x, mask):
        x = modules.HiddenPadding(self.cfg)(x, mask)
        x = modules.SAB2(self.cfg)(x, mask)
        x = modules.SAB2(self.cfg)(x, mask)
        x = modules.PMA(self.cfg)(x, mask)
        x = nn.Dense(1)(x)
        return x
