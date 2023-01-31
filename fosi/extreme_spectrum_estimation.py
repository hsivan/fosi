import jax
import jax.numpy as jnp

from fosi.lanczos_algorithm import lanczos_alg
import jax.experimental.host_callback as hcb


def _ese(lanczos_alg_gitted, batches, params):
    # TODO: in the future we can support average of Lanczos outputs on multiple batches (while keeping eigenvectors orthogonal).
    assert len(batches) == 1

    #approx_start = timer()

    for i, batch in enumerate(batches):
        k_largest_eigenvals, k_largest_eigenvecs, k_smallest_eigenvals, k_smallest_eigenvecs = lanczos_alg_gitted(params, batch)
        k_eigenvals = jnp.append(k_smallest_eigenvals, k_largest_eigenvals)
        k_eigenvecs = jnp.append(k_smallest_eigenvecs, k_largest_eigenvecs, axis=0)

    # TODO: timer does not work under jit
    #elapsed_time = timer() - approx_start
    #hcb.call(lambda args: print(f"lambda_max: {jnp.max(args[0])} lrs: {1.0 / args[0]} eigenvals: {args[0]} latency: {args[1]}"), (k_eigenvals, elapsed_time))
    hcb.call(lambda args: print(f"lambda_max: {jnp.max(args[0])} lrs: {1.0 / args[0]} eigenvals: {args[0]}"), (k_eigenvals,))

    # TODO: return only positive eigenvalues and their corresponding eigenvectors
    return (k_eigenvals, k_eigenvecs)


def get_ese_fn(loss_fn, num_params, k_largest, batches_for_lanczos=None, return_precision='32', k_smallest=0):
    key = jax.random.PRNGKey(0)
    lanczos_order = int(jnp.max(4 * (k_largest + k_smallest), 2 * jnp.log(num_params)))
    lanczos_alg_gitted = lanczos_alg(num_params, lanczos_order, loss_fn, key, k_largest, k_smallest, return_precision=return_precision)

    # ese_fn will be jitted inside the optimizer
    if batches_for_lanczos is not None:
        # Use static batch mode: all evaluation of the Hessian are done at the same batch
        ese_fn = lambda params: _ese(lanczos_alg_gitted, batches_for_lanczos, params)
    else:
        # Use dynamic batch mode: the batches are sent to Lanczos from within the optimizer.
        # args = (params, batches_for_lanczos)
        ese_fn = lambda args: _ese(lanczos_alg_gitted, args[1], args[0])

    print("Returned ESE function for", num_params, "num_params. Lanczos order (m) is", lanczos_order, ".")
    return ese_fn
