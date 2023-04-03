import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb

from fosi.jax_optim.lanczos_algorithm import lanczos_alg


def print_cb(args, transforms, device):
    if device == jax.devices()[0]:
        print(f"lambda_max: {jnp.max(args[0])} lrs: {1.0 / args[0]} eigenvals: {args[0]}")


def _ese(lanczos_alg_gitted, batch, params):
    k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs = lanczos_alg_gitted(params, batch)
    k_eigenvals = jnp.append(l_smallest_eigenvals, k_largest_eigenvals)
    k_eigenvecs = jnp.append(l_smallest_eigenvecs, k_largest_eigenvecs, axis=0)
    hcb.id_tap(print_cb, (k_eigenvals,), tap_with_device=True)
    return (k_eigenvals, k_eigenvecs)


def get_ese_fn(loss_fn, k_largest, batch=None, l_smallest=0, return_precision='32', b_parallel=False):
    key = jax.random.PRNGKey(0)
    # TODO: the following should be max(4 * (k_largest + l_smallest), 2 * int(log(num_params))), however,
    # num_params is not available at the time of the construction of the optimizer. Note that log(1e+9) ~= 40,
    # therefore, for num_params < 1e+9 and k>=10 we have 4 * (k_largest + l_smallest) > 2 * log(num_params).
    # Hence, 4 * (k_largest + l_smallest) is the maximum in our experiments.
    lanczos_order = 4 * (k_largest + l_smallest)
    lanczos_alg_gitted = lanczos_alg(lanczos_order, loss_fn, key, k_largest, l_smallest, return_precision=return_precision, b_parallel=b_parallel)

    # The returned ese_fn can be jitted
    if batch is not None:
        # Use static batch mode: all evaluation of the Hessian are done at the same batch
        ese_fn = lambda params: _ese(lanczos_alg_gitted, batch, params)
    else:
        # Use dynamic batch mode: the batch is sent to Lanczos from within the optimizer with args = (params, batch).
        ese_fn = lambda params, batch: _ese(lanczos_alg_gitted, batch, params)

    print("Returned ESE function. Lanczos order (m) is", lanczos_order, ".")
    return ese_fn
