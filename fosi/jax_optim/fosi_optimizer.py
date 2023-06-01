from typing import Any, Optional, NamedTuple, Callable, Tuple, Union

import typing_extensions
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
from optax import safe_int32_increment, Params, Updates, OptState, TransformInitFn, GradientTransformation, Schedule

from fosi.jax_optim.extreme_spectrum_estimation import get_ese_fn

ScalarOrSchedule = Union[float, Schedule]


class UpdateEseFn(typing_extensions.Protocol):
    def __call__(
            self,
            state: OptState,
            params: Params = None,
            batch: Any = None
    ) -> OptState:
        """The `update` function of the extreme spectrum estimation (extreme eigenvalues and eigenvectors of the Hessian).
        Args:
            state: The state of the gradient transformation.
            params: The current value of the parameters.
        Returns:
            The updated state.
        """


class TransformUpdateFn(typing_extensions.Protocol):
    def __call__(
            self,
            updates: Updates,
            state: OptState,
            params: Params
    ) -> Tuple[Updates, OptState]:
        """The `update` function.
        Args:
            updates: A tree of candidate updates.
            state: The state of the gradient transformation.
            params: The current value of the parameters.
        Returns:
            The transformed updates, and the updated state.
        """


class GradientTransformationFosi(NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn
    update_ese: UpdateEseFn


class ScaleByFosiState(NamedTuple):
    base_opt_state: OptState
    velocity: jnp.ndarray
    count: jnp.ndarray
    k_learning_rates: jnp.ndarray
    k_eigenvecs: jnp.ndarray
    scaling_factor: jnp.float32


def scale_by_fosi(
        base_optimizer: GradientTransformation,
        momentum_func: Callable,
        loss_fn: Callable,
        batch: Any,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    accumulator_dtype = None if accumulator_dtype is None else jax.dtypes.canonicalize_dtype(accumulator_dtype)
    warmup_w = warmup_w if warmup_w is not None else num_iters_to_approx_eigs
    if b_parallel and b_call_ese_internally:
        raise Exception("b_parallel=True and b_call_ese_internally=True is not supported.")
    ese_fn = get_ese_fn(loss_fn, approx_k, None, approx_l, b_parallel=b_parallel)

    @jax.jit
    def _approx_learning_rates_and_eigenvectors(params, state, e_batch):
        k_eigenvals, k_eigenvecs = ese_fn(params, e_batch)
        k_learning_rates = jnp.abs(1.0 / k_eigenvals)
        # k_learning_rates = jnp.where(jnp.abs(k_eigenvals) < 1.0, jnp.sqrt(jnp.abs(k_eigenvals)), jnp.abs(k_eigenvals))
        # Scaling factor for base_opt_deltas, which is clipped k_learning_rates[approx_l] / k_learning_rates[-1]
        scaling_factor = jnp.clip(k_learning_rates[approx_l] / k_learning_rates[-1], jnp.float32(1.0), learning_rate_clip)
        state = ScaleByFosiState(base_opt_state=state.base_opt_state, velocity=state.velocity, count=state.count,
                                 k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)
        return state

    def approx_learning_rates_and_eigenvectors(params, state, e_batch=None):
        if b_call_ese_internally:
            raise Exception("External call to update_ese() method while FOSI optimizer was initializes with b_call_ese_internally=True.")

        if b_parallel:
            # Batch should be passed to this function only is using parallel training, since pmap is done on 'batch'
            return _approx_learning_rates_and_eigenvectors(params, state, e_batch)
        else:
            return _approx_learning_rates_and_eigenvectors(params, state, batch)

    def _appprox_newton_direction(g1, k_eigenvecs, k_learning_rates):
        # Compute newton_direction (sum of gradient projections on leading eigenvectors scaled by eigenvalues)
        # and batch_gradients (residual of the gradient)
        newton_direction = jnp.dot(k_learning_rates * jnp.dot(k_eigenvecs, g1), k_eigenvecs)
        return newton_direction

    def _orthogonalize_vector_wrt_eigenvectors(v, k_eigenvecs):
        v = v - jnp.dot(jnp.dot(k_eigenvecs, v), k_eigenvecs)
        return v

    def _get_g1_and_g2(g, k_eigenvecs):
        g1 = jnp.dot(jnp.dot(k_eigenvecs, g), k_eigenvecs)  # g1 is the sum of g's projections on k_eigenvecs
        g2 = g - g1  # g2 is orthogonal to g1 and is the sum of g's projection on the rest of the eigenvectors
        return g1, g2

    def init_fn(params):
        flatten_param = ravel_pytree(params)[0]
        num_params = flatten_param.shape[0]
        base_opt_state = base_optimizer.init(flatten_param)

        velocity = jnp.zeros(num_params, dtype=accumulator_dtype)
        count = jnp.zeros([], jnp.int32)
        k_learning_rates = jnp.array([0.0] * (approx_k + approx_l), dtype=jnp.float32)
        k_eigenvecs = jnp.zeros((approx_k + approx_l, num_params), dtype=jnp.float32)
        scaling_factor = jnp.float32(1.0)
        state = ScaleByFosiState(base_opt_state=base_opt_state, velocity=velocity, count=count,
                                 k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)
        # If b_call_ese_internally is false, calling the function in order to produce compilation during initialization,
        # rather than later. This is not required if b_call_ese_internally is true, as it will be automatically compiled
        # on the first call to update_fn as part of the jax.lax.cond compilation of both branches.
        if not b_call_ese_internally:
            # Just produce compilation, without using the results
            if b_parallel:
                # Usually, at this point batch is already replicated, but not params
                jax.pmap(_approx_learning_rates_and_eigenvectors, axis_name='batch')(jax.device_put_replicated(params, jax.local_devices()),
                                                                                     jax.device_put_replicated(state, jax.local_devices()), batch)
            else:
                _approx_learning_rates_and_eigenvectors(params, state, batch)
        return state

    def update_fn(updates, state, params):
        # To simplify usage, the ESE procedure can be called internally in the update function.
        # However, jax.lax.cond increases the latency, even when the condition is not satisfied, since the
        # 'false_func' must receive and return the same types as the 'true_func', which include the state that is large.
        # (This is also the case, behind the scenes, when using lax.while_loop, lax.fori_loop, and lax.switch.)
        #
        # Alternatively, the user could set b_call_ese_internally to False and call state.update_ese externally.
        # To obtain the best performance, make sure this is not called from within a jitted function, but rather
        # directly from the main training loop, and to not involve any device variables - only cpu variables - when
        # checking the condition. An example could be found in transfer_learning_cifar10.py.
        if b_call_ese_internally:
            state = jax.lax.cond(
                (state.count + 1 >= warmup_w) & (jnp.mod((state.count + 1 - warmup_w), num_iters_to_approx_eigs) == 0),
                lambda x, y: _approx_learning_rates_and_eigenvectors(x, y, batch),
                lambda x, y: y, params, state)

        g, unravel = ravel_pytree(updates)
        flatten_param = ravel_pytree(params)[0]

        # TODO: Weight decay should be added to the loss directly as an L2 regularization, rather than indirectly
        #  through the optimizer step, i.e., the optimizer weight_decay must be 0.
        #  The reason is that using weight_decay indirectly through the optimizer step, and not directly through the
        #  loss function, results in ESE inaccurate eigenvectors and eigenvalues estimation; the gradient of the loss
        #  is not the real gradient used for the update step.

        g1, g2 = _get_g1_and_g2(g, state.k_eigenvecs)

        new_velocity = momentum_func(g1, state.velocity)
        # Cast the tree to accumulator_dtype
        new_velocity = new_velocity if accumulator_dtype is None else new_velocity.astype(accumulator_dtype)

        newton_direction = _appprox_newton_direction(new_velocity, state.k_eigenvecs, state.k_learning_rates)

        base_opt_deltas, new_base_opt_state = base_optimizer.update(g2, state.base_opt_state, flatten_param)

        # Reduce from base_opt_deltas its projection on k_eigenvecs, which makes base_opt_deltas orthogonal to
        # k_eigenvecs as well as to newton_direction (which is a linear combination of k_eigenvecs).
        base_opt_deltas = _orthogonalize_vector_wrt_eigenvectors(base_opt_deltas, state.k_eigenvecs)

        alpha_val = alpha(state.count) if callable(alpha) else alpha

        # base_opt_deltas already negated, therefore in the update it appears without the minus sign
        updates = unravel(state.scaling_factor * base_opt_deltas - alpha_val * newton_direction)
        count_inc = safe_int32_increment(state.count)
        return updates, ScaleByFosiState(base_opt_state=new_base_opt_state, velocity=new_velocity, count=count_inc,
                                         k_learning_rates=state.k_learning_rates, k_eigenvecs=state.k_eigenvecs, scaling_factor=state.scaling_factor)

    return GradientTransformationFosi(init_fn, update_fn, approx_learning_rates_and_eigenvectors)


# Using fosi() directly is possible, but requires defining the momentum_func and learning_rate_clip parameters.
# For ease of use, we offer fosi_adam, fosi_momentum, and fosi_sgd which automatically set the momentum_func based on
# the underlying optimizer.
# Note: fosi_adam has learning_rate_clip set to 1 as Adam's learning rate shouldn't be scaled.
def fosi(
        base_optimizer: GradientTransformation,
        momentum_func: Callable,
        loss_fn: Callable,
        batch: Any,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    return scale_by_fosi(base_optimizer=base_optimizer, momentum_func=momentum_func, loss_fn=loss_fn, batch=batch,
                         accumulator_dtype=accumulator_dtype, num_iters_to_approx_eigs=num_iters_to_approx_eigs,
                         approx_k=approx_k, approx_l=approx_l, warmup_w=warmup_w, alpha=alpha,
                         learning_rate_clip=learning_rate_clip, b_call_ese_internally=b_call_ese_internally,
                         b_parallel=b_parallel)


def fosi_adam(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by optax.adam()
        loss_fn: Callable,
        batch: Any,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    # Note: Adam should use learning_rate_clip = 1.0
    return fosi(base_optimizer, lambda g, t: (1 - decay) * g + decay * t, loss_fn, batch, accumulator_dtype,
                num_iters_to_approx_eigs, approx_k, approx_l, warmup_w, alpha, 1.0, b_call_ese_internally, b_parallel)


def fosi_momentum(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by optax.sgd() with momentum
        loss_fn: Callable,
        batch: Any,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    return fosi(base_optimizer, lambda g, t: g + decay * t, loss_fn, batch, accumulator_dtype, num_iters_to_approx_eigs,
                approx_k, approx_l, warmup_w, alpha, learning_rate_clip, b_call_ese_internally, b_parallel)


def fosi_nesterov(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by optax.sgd() with momentum and nesterov
        loss_fn: Callable,
        batch: Any,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    return fosi(base_optimizer, lambda g, t: (1 + decay) * g + decay**2 * t, loss_fn, batch, accumulator_dtype, num_iters_to_approx_eigs,
                approx_k, approx_l, warmup_w, alpha, learning_rate_clip, b_call_ese_internally, b_parallel)


def fosi_sgd(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by optax.sgd() without momentum
        loss_fn: Callable,
        batch: Any,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: ScalarOrSchedule = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        b_call_ese_internally: bool = True,
        b_parallel: bool = False,
) -> GradientTransformationFosi:
    return fosi(base_optimizer, lambda g, t: g, loss_fn, batch, accumulator_dtype, num_iters_to_approx_eigs,
                approx_k, approx_l, warmup_w, alpha, learning_rate_clip, b_call_ese_internally, b_parallel)
