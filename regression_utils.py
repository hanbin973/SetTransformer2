import numpy as np
import numba
from typing import Any, Callable, Sequence, Optional, Union

import jax 
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state

import optax

def generate_max_regression(
    num_simulations: int,
    num_samples: int,
    low: float = 0.,
    high: float = 10.,
):
    # Output shapes:
    # X: num_batch * num_tokens (set size) * 1
    # y: num_batch * 1 (max val) * 1
    p_mask = 0.5 + 0.5 * np.random.rand(num_simulations)
    mask = np.random.rand(num_simulations, num_samples) < p_mask[:, None]
    X = low + (high - low) * np.random.rand(num_simulations, num_samples)
    y = np.where(mask, X, -2 * np.abs(low)).max(axis=1)
    return X[..., None], y[..., None], mask

def generate_mean_regression(
    num_simulations: int,
    num_samples: int,
    low: float = 0.,
    high: float = 10.,
):
    # Output shapes:
    # X: num_batch * num_tokens (set size) * 1
    # y: num_batch * 1 (max val) * 1
    p_mask = 0.5 + 0.5 * np.random.rand(num_simulations)
    mask = np.random.rand(num_simulations, num_samples) < p_mask[:, None]
    X = low + (high - low) * np.random.rand(num_simulations, num_samples)
    y = (X * mask).sum(axis=1) / mask.sum(axis=1)
    return X[..., None], y[..., None], mask

@numba.jit
def generate_linear_regression(
    num_simulations: int,
    predictors: np.ndarray,
    tau: float = 1,
    sigma: float =1,
):
    num_samples, num_features = predictors.shape
    b_arr, y_arr = np.zeros((num_simulations, num_features)), np.zeros((num_simulations, num_samples))
    for i in range(num_simulations):
        b = np.random.normal(loc=0, scale=tau, size=num_features)
        e = np.random.normal(loc=0, scale=sigma, size=num_samples)
        y = predictors @ b + e
        b_arr[i], y_arr[i] = b, y
    return b_arr[..., None], y_arr[..., None]

@numba.jit
def generate_linear_regression_dynamic(
    num_simulations: int,
    num_samples: int,
    num_features: int,
    tau: float = 1,
    sigma: float = 1,
):
    b_arr, y_arr = np.zeros((num_simulations, num_features)), np.zeros((num_simulations, num_samples))
    predictors_arr = np.zeros((num_simulations, num_samples, num_features))
    mask_arr = np.empty((num_simulations, num_samples), dtype=np.bool_)
    for i in range(num_simulations):
        p_mask = 0.3 + 0.7 * np.random.rand(1)
        predictors = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))
        b = np.random.normal(loc=0, scale=tau, size=num_features)
        e = np.random.normal(loc=0, scale=sigma, size=num_samples)
        y = predictors @ b + e
        mask = np.random.rand(num_samples) < p_mask
        b_arr[i], y_arr[i] = b, y
        predictors_arr[i] = predictors
        mask_arr[i] = mask
    return b_arr[..., None], y_arr[..., None], predictors_arr, mask_arr

@jax.jit
def train_step_simple(state, X, y, mask):
    def loss_fn(params):
        max_pred = state.apply_fn(params, X, mask)
        return jnp.mean(jnp.abs(y[..., None] - max_pred))
    
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def train_step_regression(state, bs, ys, masks):
    def loss_fn(params):
        b_pred = state.apply_fn(params, ys, masks)
        return jnp.mean(jnp.power(bs - b_pred, 2))
    
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss