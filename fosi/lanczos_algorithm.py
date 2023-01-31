# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for running the Lanczos algorithm."""

import jax.numpy as np
import jax.random as random
from jax import jit
import jax
from jax.flatten_util import ravel_pytree
from jax import jvp, grad
from jax.config import config


def lanczos_alg(dim, order, loss, rng_key, k_largest, k_smallest=0, return_precision='32'):
    # The algorithm must run in high precision; however, after extracting the eigenvalues and eigenvectors we can
    # cast it back to 32 bit.
    config.update("jax_enable_x64", True)

    def orthogonalization(vecs, w, tridiag, i):
        # Full reorthogonalization.
        # Note that orthogonalization here includes all vectors in vecs, and not just vectors j s.t. j <= i.
        # Since all vectors j s.t. j > i are zeros (vecs is initialized to zeros), there is no impact on w if iteration
        # continues for j > i.
        # However, using the iteration on all the vectors enables us to use jit over this function.
        # Otherwise, we will have to iterate/slice by i, which is not supported by jit.

        # The operation np.dot(np.dot(vecs, w), vecs) is equivalent to multiply (scale) each vector in its own coeff,
        # where coeffs = np.dot(vecs, w) is array of coeffs (scalars) with shape (order,), and then sum all the
        # scaled vectors.
        w = w - np.dot(np.dot(vecs, w), vecs)  # single vector with the shape of w
        # repeat the full orthogonalization for stability
        w = w - np.dot(np.dot(vecs, w), vecs)  # single vector with the shape of w

        beta = np.linalg.norm(w)

        tridiag = tridiag.at[i, i + 1].set(beta)
        tridiag = tridiag.at[i + 1, i].set(beta)
        vecs = vecs.at[i + 1].set(w / beta)

        return (tridiag, vecs)

    def lanczos_iteration(i, args, params, unravel, batch):
        vecs, tridiag = args

        # Get last two vectors
        v = vecs[i]

        # Assign to w the Hessian vector product Hv. Uses forward-over-reverse mode for computing Hv.
        # We assume here that the default precision is 32 bit.
        v_flat = unravel(v) if return_precision == '64' else unravel(v.astype(np.float32))  # convert v to the param tree structure
        loss_fn = lambda x: loss(x, batch)
        hessian_vp = jvp(grad(loss_fn), [params], [v_flat])[1]
        w, _ = ravel_pytree(hessian_vp)
        w = w.astype(np.float64)

        # Evaluates alpha and update tridiag diagonal with alpha
        alpha = np.dot(w, v)
        tridiag = tridiag.at[i, i].set(alpha)

        # For every iteration except the last one, perform full orthogonalization on w and normalized it (beta is w's
        # norm). Update tridiag secondary diagonals with beta and update vecs with the normalized orthogonal w.
        tridiag, vecs = jax.lax.cond(i + 1 < order,
                                     lambda: orthogonalization(vecs, w, tridiag, i),
                                     lambda: (tridiag, vecs))

        return (vecs, tridiag)

    @jit
    def lanczos_alg_jitted(params, batch):
        """
            Lanczos algorithm for tridiagonalizing a real symmetric matrix.
            This function applies Lanczos algorithm of a given order.  This function
            does full reorthogonalization.
            The first time the function is called it is compiled, which can take ~30 second for 10,000,000 parameters and order 100.
            Args:
                matrix_vector_product: Maps v -> Hv for a real symmetric matrix H. Input/Output must be of shape [dim].
                dim: Matrix H is [dim, dim].
                order: An integer corresponding to the number of Lanczos steps to take.
                rng_key: The jax PRNG key.
            Returns:
                tridiag: A tridiagonal matrix of size (order, order).
                vecs: A numpy array of size (order, dim) corresponding to the Lanczos vectors.
            """

        # Initialization
        _, unravel = ravel_pytree(params)
        tridiag = np.zeros((order, order), dtype=np.float64)
        vecs = np.zeros((order, dim), dtype=np.float64)
        init_vec = random.normal(rng_key, shape=(dim,))
        init_vec = init_vec / np.linalg.norm(init_vec)
        vecs = vecs.at[0].set(init_vec)

        lanczos_iter = lambda i, args: lanczos_iteration(i, args, params, unravel, batch)
        # Lanczos iterations.
        # Use fori_loop which makes jax compile lanczos_iter() function only once and accelerates the compilation.
        vecs, tridiag = jax.lax.fori_loop(0, order, lanczos_iter, (vecs, tridiag))

        eigs_tridiag, eigvecs_triag = np.linalg.eigh(tridiag)  # eigs_tridiag are also eigenvalues of  the Hessian

        precision = np.float64 if return_precision == '64' else np.float32
        k_largest_eigenvals = eigs_tridiag[order-k_largest:].astype(precision)
        k_largest_eigenvecs = (eigvecs_triag.T[order-k_largest:] @ vecs).astype(precision)
        k_smallest_eigenvals = eigs_tridiag[:k_smallest].astype(precision)
        k_smallest_eigenvecs = (eigvecs_triag.T[:k_smallest] @ vecs).astype(precision)

        return k_largest_eigenvals, k_largest_eigenvecs, k_smallest_eigenvals, k_smallest_eigenvecs

    return lanczos_alg_jitted
