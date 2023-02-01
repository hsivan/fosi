import jax
import jax.numpy as jnp

from fosi.lanczos_algorithm import lanczos_alg
import jax.experimental.host_callback as hcb


def _ese(lanczos_alg_gitted, batch, params):
    # TODO: in the future we can support average of Lanczos outputs on multiple batches (while keeping eigenvectors orthogonal).

    #approx_start = timer()

    k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs = lanczos_alg_gitted(params, batch)
    k_eigenvals = jnp.append(l_smallest_eigenvals, k_largest_eigenvals)
    k_eigenvecs = jnp.append(l_smallest_eigenvecs, k_largest_eigenvecs, axis=0)

    # TODO: timer does not work under jit
    #elapsed_time = timer() - approx_start
    #hcb.call(lambda args: print(f"lambda_max: {jnp.max(args[0])} lrs: {1.0 / args[0]} eigenvals: {args[0]} latency: {args[1]}"), (k_eigenvals, elapsed_time))
    hcb.call(lambda args: print(f"lambda_max: {jnp.max(args[0])} lrs: {1.0 / args[0]} eigenvals: {args[0]}"), (k_eigenvals,))

    # TODO: return only positive eigenvalues and their corresponding eigenvectors
    return (k_eigenvals, k_eigenvecs)


def get_ese_fn(loss_fn, k_largest, batch=None, return_precision='32', k_smallest=0):
    key = jax.random.PRNGKey(0)
    # TODO: the following should be int(jnp.max(4 * (k_largest + k_smallest), 2 * jnp.log(num_params))), however,
    # num_params is not available at the time of the optimizer construction. Note that for jnp.log(1e9) ~= 40,
    # therefore, for num_params < 1e9 and k>=10 we have 4 * (k_largest + k_smallest) > 2 * jnp.log(num_params).
    lanczos_order = 4 * (k_largest + k_smallest)
    lanczos_alg_gitted = lanczos_alg(lanczos_order, loss_fn, key, k_largest, k_smallest, return_precision=return_precision)

    # ese_fn will be jitted inside the optimizer
    if batch is not None:
        # Use static batch mode: all evaluation of the Hessian are done at the same batch
        ese_fn = lambda params: _ese(lanczos_alg_gitted, batch, params)
    else:
        # Use dynamic batch mode: the batches are sent to Lanczos from within the optimizer.
        # args = (params, batches_for_lanczos)
        ese_fn = lambda args: _ese(lanczos_alg_gitted, args[1], args[0])

    print("Returned ESE function. Lanczos order (m) is", lanczos_order, ".")
    return ese_fn
