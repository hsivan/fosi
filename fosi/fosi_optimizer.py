from typing import Any, Optional, NamedTuple

from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src.base import OptState
from optax._src.combine import chain
from optax._src.base import Callable


def approx_learning_rates_and_eigenvectors(ese_fn, params):
    k_eigenvals, k_eigenvecs = ese_fn(params)
    k_learning_rates = jnp.abs(1.0 / k_eigenvals)
    # k_learning_rates = jnp.where(jnp.abs(k_eigenvals) < 1.0, jnp.sqrt(jnp.abs(k_eigenvals)), jnp.abs(k_eigenvals))
    return k_learning_rates, k_eigenvecs


def appprox_newton_direction(g1, state, num_iters_to_approx_eigs, ese_fn, params, warmup_w):
    flatten_grad, unravel = ravel_pytree(g1)

    k_learning_rates, k_eigenvecs = jax.lax.cond(
        (state.count + 1 >= warmup_w) & ((state.count + 1 - warmup_w) % num_iters_to_approx_eigs == 0),
        lambda x: approx_learning_rates_and_eigenvectors(ese_fn, x),
        lambda x: (state.k_learning_rates, state.k_eigenvecs), params)

    # Compute coeffs (size of the gradient projection on the k leading eigenvectors)
    coeffs = k_eigenvecs @ flatten_grad

    # Compute newton_direction (sum of gradient projections on leading eigenvectors scaled by eigenvalues)
    # and batch_gradients (residual of the gradient)
    newton_direction = jnp.dot(k_learning_rates * coeffs, k_eigenvecs)
    newton_direction = unravel(newton_direction)

    return newton_direction, k_learning_rates, k_eigenvecs


def orthogonalize_vector_wrt_eigenvectors(v, k_eigenvecs):
    v, unravel = ravel_pytree(v)
    v = v - jnp.dot(jnp.dot(k_eigenvecs, v), k_eigenvecs)
    v = unravel(v)
    return v


def get_g1_and_g2(g, k_eigenvecs):
    g, unravel = ravel_pytree(g)
    g1 = jnp.dot(jnp.dot(k_eigenvecs, g), k_eigenvecs)  # g1 is the sum of g's projections on k_eigenvecs
    g2 = g - g1  # g2 is orthogonal to g1 and is the sum of g's projection on the rest of the eigenvectors
    g1 = unravel(g1)
    g2 = unravel(g2)
    return g1, g2


class ScaleByFosiState(NamedTuple):
    base_opt_state: OptState
    velocity: base.Params
    count: jnp.ndarray
    k_learning_rates: jnp.ndarray
    k_eigenvecs: jnp.ndarray


def scale_by_fosi(
        base_optimizer: base.GradientTransformation,
        ese_fn: Callable,
        momentum_func: Callable,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_newton_k: int = 5,
        approx_newton_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
) -> base.GradientTransformation:
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    warmup_w = warmup_w if warmup_w is not None else num_iters_to_approx_eigs

    # Note: Adam should use learning_rate_clip = 1.0

    def init_fn(params):
        flat_params, _ = ravel_pytree(params)
        num_params = flat_params.shape[0]

        base_opt_state = base_optimizer.init(params)

        velocity = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params)
        count = jnp.zeros([], jnp.int32)
        k_learning_rates = jnp.array([0.0] * (approx_newton_k + approx_newton_l), dtype=jnp.float32)
        k_eigenvecs = jnp.zeros((approx_newton_k + approx_newton_l, num_params), dtype=jnp.float32)
        return ScaleByFosiState(base_opt_state=base_opt_state, velocity=velocity, count=count,
                                k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs)

    def update_fn(updates, state, params):
        g1, g2 = get_g1_and_g2(updates, state.k_eigenvecs)

        new_velocity = jax.tree_map(momentum_func, g1, state.velocity)

        new_velocity = utils.cast_tree(new_velocity, accumulator_dtype)

        newton_direction, k_learning_rates, k_eigenvecs = appprox_newton_direction(new_velocity, state,
                                                                                   num_iters_to_approx_eigs,
                                                                                   ese_fn, params,
                                                                                   warmup_w)

        base_opt_deltas, new_base_opt_state = base_optimizer.update(g2, state.base_opt_state, params)

        # Reduce from base_opt_deltas its projection on k_eigenvecs, which makes base_opt_deltas orthogonal to
        # k_eigenvecs as well as to newton_direction (which is a linear combination of k_eigenvecs).
        base_opt_deltas = orthogonalize_vector_wrt_eigenvectors(base_opt_deltas, k_eigenvecs)

        # Scale base_opt_deltas by a clipped factor k_learning_rates[approx_newton_l] / k_learning_rates[-1]
        scaling_factor = jax.lax.cond((state.count + 1 >= warmup_w) & ((state.count + 1 - warmup_w) >= 0),
                                      lambda x: jnp.clip(k_learning_rates[approx_newton_l] / k_learning_rates[-1], 1.0,
                                                         learning_rate_clip),
                                      lambda x: jnp.float32(1.0),
                                      None)
        base_opt_deltas = jax.tree_map(lambda x: x * scaling_factor, base_opt_deltas)

        # base_opt_deltas already negated, therefore in the update it appears without the minus sign
        updates = jax.tree_map(lambda x, y: x - alpha * y, base_opt_deltas, newton_direction)
        count_inc = numerics.safe_int32_increment(state.count)
        return updates, ScaleByFosiState(base_opt_state=new_base_opt_state, velocity=new_velocity, count=count_inc,
                                         k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs)

    return base.GradientTransformation(init_fn, update_fn)


def fosi(
        base_optimizer: base.GradientTransformation,
        ese_fn: Callable,
        momentum_func: Callable,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_newton_k: int = 5,
        approx_newton_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
) -> base.GradientTransformation:
    return chain(
        (scale_by_fosi(base_optimizer=base_optimizer, ese_fn=ese_fn, momentum_func=momentum_func,
                       accumulator_dtype=accumulator_dtype, num_iters_to_approx_eigs=num_iters_to_approx_eigs,
                       approx_newton_k=approx_newton_k, approx_newton_l=approx_newton_l, warmup_w=warmup_w, alpha=alpha,
                       learning_rate_clip=learning_rate_clip)),
    )


def fosi_adam(
        base_optimizer: base.GradientTransformation,  # Should be optax.adam
        ese_fn: Callable,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_newton_k: int = 5,
        approx_newton_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 1.0,
) -> base.GradientTransformation:
    return fosi(base_optimizer, ese_fn, lambda g, t: (1 - decay) * g + decay * t, accumulator_dtype,
                num_iters_to_approx_eigs, approx_newton_k, approx_newton_l, warmup_w, alpha, 1.0)


def fosi_momentum(
        base_optimizer: base.GradientTransformation,  # Should be optax.sgd with momentum
        ese_fn: Callable,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_newton_k: int = 5,
        approx_newton_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
) -> base.GradientTransformation:
    return fosi(base_optimizer, ese_fn, lambda g, t: g + decay * t, accumulator_dtype,
                num_iters_to_approx_eigs, approx_newton_k, approx_newton_l, warmup_w, alpha, learning_rate_clip)


def fosi_sgd(
        base_optimizer: base.GradientTransformation,  # Should be optax.sgd without momentum
        ese_fn: Callable,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_newton_k: int = 5,
        approx_newton_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
) -> base.GradientTransformation:
    return fosi(base_optimizer, ese_fn, lambda g, t: g, accumulator_dtype,
                num_iters_to_approx_eigs, approx_newton_k, approx_newton_l, warmup_w, alpha, learning_rate_clip)
