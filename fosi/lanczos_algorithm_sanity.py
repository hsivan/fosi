import jax.numpy as np
import jax.random as random
from jax.example_libraries.stax import Dense, LogSoftmax, Relu, Tanh
from jax.example_libraries import stax
from jax.flatten_util import ravel_pytree
from jax import jacfwd, jacrev
from jax.config import config
import jax

import functools
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from lanczos_algorithm import lanczos_alg


def lanczos_algorithm_test():

    def get_batch(input_size, output_size, batch_size, key):
        key, split = random.split(key)
        xs = random.normal(split, shape=(batch_size, input_size))
        key, split = random.split(key)
        ys = random.randint(split, minval=0, maxval=output_size, shape=(batch_size,))
        ys = np.eye(output_size)[ys]
        return (xs, ys), key

    def prepare_single_layer_model(input_size, output_size, width, key):
        init_random_params, predict = stax.serial(Dense(width), Tanh, Dense(output_size), LogSoftmax)
        key, split = random.split(key)
        _, params = init_random_params(split, (-1, input_size))
        return predict, params, key

    def loss(y, y_hat):
        return -np.sum(y * y_hat)

    def full_hessian(loss, params):
        flat_params, unravel = ravel_pytree(params)

        def loss_flat(flat_params):
            params = unravel(flat_params)
            return loss(params)

        def hessian(f):
            return jacfwd(jacrev(f))

        hessian_matrix = hessian(loss_flat)(flat_params)
        return hessian_matrix

    key = random.PRNGKey(0)
    input_size = 10
    output_size = 10
    width = 100
    batch_size = 5
    atol_e = 1e-4

    predict, params, key = prepare_single_layer_model(input_size, output_size, width, key)
    num_params = ravel_pytree(params)[0].shape[0]
    b, key = get_batch(input_size, output_size, batch_size, key)

    def loss_fn(params, batch):
        return loss(predict(params, batch[0]), batch[1])

    largest_k = 10
    smallest_k = 3
    lanczos_order = 100
    hvp_cl = lanczos_alg(lanczos_order, loss_fn, key, lanczos_order, return_precision='32')  # Return all lanczos_order eigen products

    # compute the full hessian
    loss_cl = functools.partial(loss_fn, batch=b)
    hessian = full_hessian(loss_cl, params)
    eigs_true, eigvecs_true = np.linalg.eigh(hessian)

    for i in range(10):
        start_iteration = timer()
        print("About to run lanczos for", num_params, "parameters")
        eigs_lanczos, eigvecs_lanczos, _, _ = hvp_cl(params, b)

        if i == 0:
            assert np.allclose(eigs_true[-largest_k:], eigs_lanczos[-largest_k:], atol=atol_e), print("i:", i, "eigs_true:", eigs_true[-largest_k:], "eigs_lanczos:", eigs_lanczos[-largest_k:])
            assert np.allclose(eigs_true[:smallest_k], eigs_lanczos[:smallest_k], atol=atol_e), print("i:", i, "eigs_true:", eigs_true[:smallest_k], "eigs_lanczos:", eigs_lanczos[:smallest_k])
            perfect_vectors_similarity = np.eye(largest_k)
            top_vectors_similarity = eigvecs_lanczos[-largest_k:] @ eigvecs_true[:, -largest_k:]
            assert np.allclose(perfect_vectors_similarity, np.abs(top_vectors_similarity), atol=atol_e)
            perfect_vectors_similarity = np.eye(smallest_k)
            bottom_vector_similarity = eigvecs_lanczos[:smallest_k] @ eigvecs_true[:, :smallest_k]
            assert np.allclose(perfect_vectors_similarity, np.abs(bottom_vector_similarity), atol=atol_e)

        lambda_min, lambda_max = eigs_lanczos[0], eigs_lanczos[-1]
        end_iteration = timer()
        print("iterations", i, ": lambda min:", lambda_min, "lambda max:", lambda_max, "time(s):", end_iteration - start_iteration)

        if i == 0:
            fig, ax = plt.subplots(1, 1)
            eigs_true_edges = np.append(eigs_true[:lanczos_order//2], eigs_true[-lanczos_order//2:])
            ax.plot(range(eigs_lanczos.shape[0]), np.abs(eigs_true_edges - eigs_lanczos) / np.abs(eigs_true_edges))
            ax.set_title("Accuracy of lanczos eigenvalues")
            ax.set_xlabel("eigenvalue index")
            ax.set_ylabel("| eig_true - eig_lanczos | / eig_true")
            plt.show()

        params = jax.tree_map(lambda p: p * 0.99, params)

    print("True lambda min:", np.min(eigs_true), "true lambda max:", np.max(eigs_true))


def lanczos_eigen_approx_test():
    n_dim = 1500
    atol_e = 1e-4
    key = random.PRNGKey(0)
    lanczos_order = 100
    largest_k = 8
    smallest_k = 8

    def fill_diagonal(a, val):
        assert a.ndim >= 2
        i, j = np.diag_indices(min(a.shape[-2:]))
        return a.at[..., i, j].set(val)

    eigenvectors = np.eye(n_dim)

    eigenvectors = eigenvectors.at[0, 0].set(0.5)
    eigenvectors = eigenvectors.at[0, 1].set(0.5)
    eigenvectors = eigenvectors.at[1, 0].set(-0.5)
    eigenvectors = eigenvectors.at[1, 1].set(0.5)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1)

    eigenvalues = random.normal(key, shape=(n_dim,))
    eigenvalues_matrix = np.zeros_like(eigenvectors)
    eigenvalues_matrix = fill_diagonal(eigenvalues_matrix, eigenvalues)
    hessian = eigenvectors.T @ eigenvalues_matrix @ eigenvectors

    eigs_true, eigvecs_true = np.linalg.eigh(hessian)

    def objective(x, batch=None):
        return 0.5 * x @ hessian @ x.T

    x_initial = np.ones(n_dim) * 0.5
    x_initial = x_initial.at[1].set(1.0)
    x_initial = x_initial @ eigenvectors

    hvp_cl = lanczos_alg(lanczos_order, objective, key, lanczos_order, return_precision='32')  # Return all lanczos_order eigen products
    eigs_lanczos, eigvecs_lanczos, _, _ = hvp_cl(x_initial, None)

    assert np.allclose(eigs_true[-largest_k:], eigs_lanczos[-largest_k:], atol=atol_e), print("eigs_true:", eigs_true[-largest_k:], "eigs_lanczos:", eigs_lanczos[-largest_k:])
    assert np.allclose(eigs_true[:smallest_k], eigs_lanczos[:smallest_k], atol=atol_e), print("eigs_true:", eigs_true[:smallest_k], "eigs_lanczos:", eigs_lanczos[:smallest_k])

    perfect_vectors_similarity = np.eye(largest_k)
    top_vectors_similarity = eigvecs_lanczos[-largest_k:] @ eigvecs_true[:, -largest_k:]
    assert np.allclose(perfect_vectors_similarity, np.abs(top_vectors_similarity), atol=atol_e)

    perfect_vectors_similarity = np.eye(smallest_k)
    bottom_vectors_similarity = eigvecs_lanczos[:smallest_k] @ eigvecs_true[:, :smallest_k]
    assert np.allclose(perfect_vectors_similarity, np.abs(bottom_vectors_similarity), atol=atol_e)

    fig, ax = plt.subplots(1, 1)
    eigs_true_edges = np.append(eigs_true[:lanczos_order // 2], eigs_true[-lanczos_order // 2:])
    ax.plot(range(eigs_lanczos.shape[0]), np.abs(eigs_true_edges - eigs_lanczos) / np.abs(eigs_true_edges))
    ax.set_title("Accuracy of lanczos eigenvalues")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("| eig_true - eig_lanczos | / eig_true")
    plt.show()


if __name__ == '__main__':
    config.update("jax_enable_x64", False)
    lanczos_algorithm_test()
    config.update("jax_enable_x64", False)
    lanczos_eigen_approx_test()
