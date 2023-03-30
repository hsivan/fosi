from typing import Any, Optional, NamedTuple, Callable

import torch
# from torchopt.typing import Params, OptState  # Replace velocity and base_opt_state current 'Any' with Params and OptState for Version >= 0.6.0
from torchopt._src.base import GradientTransformation  # Replace with 'from torchopt.base import GradientTransformation' for version >= 0.6.0

from fosi.torch_optim.lanczos_algorithm import ravel, unravel
from fosi.torch_optim.extreme_spectrum_estimation import get_ese_fn


def tree_map(f: Callable[..., Any], device, tree: Any, *rest: Any) -> Any:
    leaves = ravel(tree)
    all_leaves = [leaves] + [ravel(r) for r in rest]
    return unravel(torch.stack([f(*xs) for xs in zip(*all_leaves)]), tree, device)


class ScaleByFosiState(NamedTuple):
    base_opt_state: Any  # OptState
    velocity: Any  # Params
    count: torch.tensor
    k_learning_rates: torch.tensor
    k_eigenvecs: torch.tensor
    scaling_factor: torch.tensor


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
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
        device: torch.device = torch.device("cpu"),
) -> GradientTransformation:
    accumulator_dtype = None if accumulator_dtype is None else torch.float32
    warmup_w = warmup_w if warmup_w is not None else num_iters_to_approx_eigs
    ese_fn = get_ese_fn(loss_fn, approx_k, batch, approx_l, device=device)

    def _approx_learning_rates_and_eigenvectors(params, state):
        k_eigenvals, k_eigenvecs = ese_fn(params)
        k_learning_rates = torch.abs(1.0 / k_eigenvals)
        # Scaling factor for base_opt_deltas, which is clipped k_learning_rates[approx_l] / k_learning_rates[-1]
        scaling_factor = torch.clip(k_learning_rates[approx_l] / k_learning_rates[-1], 1.0, learning_rate_clip)
        state = ScaleByFosiState(base_opt_state=state.base_opt_state, velocity=state.velocity, count=state.count,
                                 k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)
        return state

    def _appprox_newton_direction(g1, k_eigenvecs, k_learning_rates):
        flatten_grad = ravel(g1)

        # Compute coeffs (size of the gradient projection on the k leading eigenvectors)
        coeffs = k_eigenvecs @ flatten_grad

        # Compute newton_direction (sum of gradient projections on leading eigenvectors scaled by eigenvalues)
        # and batch_gradients (residual of the gradient)
        newton_direction = torch.matmul(k_learning_rates * coeffs, k_eigenvecs)
        newton_direction = unravel(newton_direction, g1, device)

        return newton_direction

    def _orthogonalize_vector_wrt_eigenvectors(v, k_eigenvecs):
        v_ = torch.unsqueeze(ravel(v), -1)
        v_ = v_ - k_eigenvecs.t().matmul(k_eigenvecs.matmul(v_))
        v = unravel(v_, v, device)
        return v

    def _get_g1_and_g2(g, k_eigenvecs):
        g_ = torch.unsqueeze(ravel(g), -1)
        g1 = k_eigenvecs.t().matmul(k_eigenvecs.matmul(g_))  # g1 is the sum of g's projections on k_eigenvecs
        g2 = g_ - g1  # g2 is orthogonal to g1 and is the sum of g's projection on the rest of the eigenvectors
        g1 = unravel(g1, g, device)
        g2 = unravel(g2, g, device)
        return g1, g2

    def init_fn(params):
        num_params = ravel(params).shape[0]
        base_opt_state = base_optimizer.init(params)

        velocity = tree_map(lambda t: torch.zeros_like(t, dtype=accumulator_dtype), device, params)
        count = torch.zeros((1,), dtype=torch.int32, device=device)
        k_learning_rates = torch.zeros((approx_k + approx_l,), dtype=torch.float32, device=device)
        k_eigenvecs = torch.zeros((approx_k + approx_l, num_params), dtype=torch.float32, device=device)
        scaling_factor = torch.ones((1,), dtype=torch.float32, device=device)
        return ScaleByFosiState(base_opt_state=base_opt_state, velocity=velocity, count=count,
                                k_learning_rates=k_learning_rates, k_eigenvecs=k_eigenvecs, scaling_factor=scaling_factor)

    def update_fn(updates, state, params):
        if (state.count + 1 >= warmup_w) & ((state.count + 1 - warmup_w) % num_iters_to_approx_eigs == 0):
            state = _approx_learning_rates_and_eigenvectors(params, state)

        g1, g2 = _get_g1_and_g2(updates, state.k_eigenvecs)

        new_velocity = tree_map(momentum_func, device, g1, state.velocity)
        # Cast the tree to accumulator_dtype
        new_velocity = new_velocity if accumulator_dtype is None else tree_map(lambda t: t.astype(accumulator_dtype), device, new_velocity)

        newton_direction = _appprox_newton_direction(new_velocity, state.k_eigenvecs, state.k_learning_rates)

        base_opt_deltas, new_base_opt_state = base_optimizer.update(g2, state.base_opt_state)

        # Reduce from base_opt_deltas its projection on k_eigenvecs, which makes base_opt_deltas orthogonal to
        # k_eigenvecs as well as to newton_direction (which is a linear combination of k_eigenvecs).
        base_opt_deltas = _orthogonalize_vector_wrt_eigenvectors(base_opt_deltas, state.k_eigenvecs)

        # Scale base_opt_deltas by a clipped factor k_learning_rates[approx_l] / k_learning_rates[-1]
        base_opt_deltas = tree_map(lambda x: x * state.scaling_factor, device, base_opt_deltas)

        # base_opt_deltas already negated, therefore in the update it appears without the minus sign
        updates = tree_map(lambda x, y: x - alpha * y, device, base_opt_deltas, newton_direction)
        count_inc = state.count + 1
        return updates, ScaleByFosiState(base_opt_state=new_base_opt_state, velocity=new_velocity, count=count_inc,
                                         k_learning_rates=state.k_learning_rates, k_eigenvecs=state.k_eigenvecs, scaling_factor=state.scaling_factor)

    return GradientTransformation(init_fn, update_fn)


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
        alpha: float = 1.0,
        learning_rate_clip: Optional[float] = 3.0,
        device: torch.device = torch.device("cpu"),
) -> GradientTransformation:
    return scale_by_fosi(base_optimizer=base_optimizer, momentum_func=momentum_func, loss_fn=loss_fn, batch=batch,
                         accumulator_dtype=accumulator_dtype, num_iters_to_approx_eigs=num_iters_to_approx_eigs,
                         approx_k=approx_k, approx_l=approx_l, warmup_w=warmup_w,
                         alpha=alpha, learning_rate_clip=learning_rate_clip, device=device)


def fosi_adam(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by torchopt.adam()
        loss_fn: Callable,
        batch: Any,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 0.1,
        device: torch.device = torch.device("cpu"),
) -> GradientTransformation:
    # Note: Adam should use learning_rate_clip = 1.0
    return fosi(base_optimizer, lambda g, t: (1 - decay) * g + decay * t, loss_fn, batch, accumulator_dtype,
                num_iters_to_approx_eigs, approx_k, approx_l, warmup_w, alpha, 1.0, device)


def fosi_momentum(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by torchopt.sgd() with momentum
        loss_fn: Callable,
        batch: Any,
        decay: float = 0.9,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        device: torch.device = torch.device("cpu"),
) -> GradientTransformation:
    return fosi(base_optimizer, lambda g, t: g + decay * t, loss_fn, batch, accumulator_dtype,
                num_iters_to_approx_eigs, approx_k, approx_l, warmup_w, alpha, learning_rate_clip, device)


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
        alpha: float = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
) -> GradientTransformation:
    return fosi(base_optimizer, lambda g, t: (1 + decay) * g + decay**2 * t, loss_fn, batch, accumulator_dtype,
                num_iters_to_approx_eigs, approx_k, approx_l, warmup_w, alpha, learning_rate_clip)


def fosi_sgd(
        base_optimizer: GradientTransformation,  # Should be a GradientTransformation instance returned by torchopt.sgd() without momentum
        loss_fn: Callable,
        batch: Any,
        accumulator_dtype: Optional[Any] = None,
        num_iters_to_approx_eigs: int = 100,
        approx_k: int = 5,
        approx_l: int = 0,
        warmup_w: Optional[int] = None,
        alpha: float = 0.1,
        learning_rate_clip: Optional[float] = 3.0,
        device: torch.device = torch.device("cpu"),
) -> GradientTransformation:
    return fosi(base_optimizer, lambda g, t: g, loss_fn, batch, accumulator_dtype,
                num_iters_to_approx_eigs, approx_k, approx_l, warmup_w, alpha, learning_rate_clip, device)
