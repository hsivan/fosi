import pickle
import jax
import jax.numpy as np
from jax import random
from jax.config import config

from fosi import get_ese_fn
from experiments.quadratic.plot_quadratic import plot_quadratic_random_orthogonal_basis_gd
from quadratic_jax_kappa_zeta import gd_update, fosi_gd_update, optimize, fill_diagonal, get_x_initial, objective

config.update("jax_enable_x64", True)


def prepare_hessian(n_dim):
    A = random.uniform(random.PRNGKey(0), shape=(n_dim, n_dim))
    B = np.dot(A, A.transpose())
    _, eigenvectors = np.linalg.eigh(B)

    num_large_eigenvals = 10
    eigenvalues = np.ones(n_dim)

    d = np.linspace(9, 10, num_large_eigenvals)[::-1]
    for i in range(0, num_large_eigenvals):
        eigenvalues = eigenvalues.at[i].set(d[i])

    d = np.linspace(0.01, 0.1, n_dim - num_large_eigenvals)[::-1]
    for i in range(num_large_eigenvals, n_dim):
        eigenvalues = eigenvalues.at[i].set(d[i - num_large_eigenvals])

    eigenvalues_matrix = fill_diagonal(np.zeros_like(eigenvectors), eigenvalues)
    hessian = eigenvectors @ eigenvalues_matrix @ eigenvectors.T
    print("Condition number:", eigenvalues[0] / eigenvalues[-1])

    return hessian, eigenvalues, eigenvectors


def optimize_quadratic_func_with_gd_and_fosi_gd():
    n_dim = 100
    optimizers_scores = {}

    hessian, eigenvalues, eigenvectors = prepare_hessian(n_dim)

    approx_k = 9
    assert approx_k <= n_dim

    objective_fn = lambda x, batch=None: objective(x, hessian, batch)

    x_initial = get_x_initial(objective_fn, eigenvectors, n_dim)
    n_iter = 250
    alpha = 0.001
    beta1 = 0.9  # factor for average gradient (first moment)
    beta2 = 0.999  # factor for average squared gradient (second moment)

    ese_fn = get_ese_fn(objective_fn, approx_k, [None], return_precision='64')
    k_eigenvals, k_eigenvecs = ese_fn(x_initial)
    print("k_eigenvals:", k_eigenvals)

    fosi_sgd_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_gd_update(x, g, m, v, t, eta, beta1, beta2,
                                                                                     mn, k_eigenvals, k_eigenvecs)

    optimizers = {"GD": gd_update,
                  "FOSI-GD": fosi_sgd_update_fn}

    for optimizer_name, optimizer_update in optimizers.items():
        if 'FOSI' in optimizer_name:
            effective_condition_number = 1 / (alpha * eigenvalues[-1])
            print("effective_condition_number:", effective_condition_number)
        scores, solutions = optimize(objective_fn, x_initial, n_iter, alpha, beta1, beta2, optimizer_update)
        print('%s: f = %.10f' % (optimizer_name, scores[-1]))
        optimizers_scores[(n_dim, jax.device_get(eigenvalues[0]).item(), optimizer_name)] = [x.item() for x in jax.device_get(scores)]

    # Plot learning curves
    pickle.dump(optimizers_scores, open("test_results/optimizers_scores_quadratic_gd.pkl", 'wb'))


if __name__ == "__main__":
    optimize_quadratic_func_with_gd_and_fosi_gd()
    plot_quadratic_random_orthogonal_basis_gd()
