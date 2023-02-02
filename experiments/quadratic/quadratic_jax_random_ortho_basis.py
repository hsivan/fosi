import pickle
import jax
import jax.numpy as np
from jax import random
from jax.config import config

from experiments.quadratic.quadratic_jax_kappa_zeta import get_x_initial, objective, optimize, fosi_adam_update, \
    fosi_momentum_update, fosi_gd_update, adam_update, momentum_update, gd_update, fill_diagonal
from fosi import get_ese_fn
from experiments.quadratic.plot_quadratic import plot_quadratic_random_orthogonal_basis

config.update("jax_enable_x64", True)


def prepare_hessian(n_dim, max_eigval):
    A = random.uniform(random.PRNGKey(0), shape=(n_dim, n_dim))
    B = np.dot(A, A.transpose())
    _, eigenvectors = np.linalg.eigh(B)

    eigenvalues = np.ones(n_dim)
    eigenvalues = eigenvalues.at[0].set(max_eigval)
    eigenvalues = eigenvalues.at[1].set(1.0)
    if n_dim > 2:
        for i in range(2, n_dim):
            eigenvalues = eigenvalues.at[i].set(eigenvalues[i - 1] / 1.5)
    eigenvalues_matrix = fill_diagonal(np.zeros_like(eigenvectors), eigenvalues)
    hessian = eigenvectors @ eigenvalues_matrix @ eigenvectors.T
    print("Condition number:", eigenvalues[0] / eigenvalues[-1])

    return hessian, eigenvalues, eigenvectors


def optimize_quadratic_funcs_with_varying_dim_and_cond_number():
    n_dims = [100, 1500]
    max_eigvals = [5, 10, 20, 50, 200]
    optimizers_scores = {}

    for n_dim in n_dims:

        approx_k = 10
        assert approx_k <= n_dim

        for max_eigval in max_eigvals:

            hessian, eigenvalues, eigenvectors = prepare_hessian(n_dim, max_eigval)

            objective_fn = lambda x, batch=None: objective(x, hessian, batch)

            x_initial = get_x_initial(objective_fn, eigenvectors, n_dim)
            n_iter = 200
            eta = 0.1
            beta1 = 0.9  # factor for average gradient (first moment)
            beta2 = 0.999  # factor for average squared gradient (second moment)

            ese_fn = get_ese_fn(objective_fn, approx_k, [None], return_precision='64')
            k_eigenvals, k_eigenvecs = ese_fn(x_initial)
            print("k_eigenvals:", k_eigenvals)

            fosi_adam_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_adam_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)
            fosi_momentum_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_momentum_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)
            fosi_sgd_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_gd_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)

            optimizers = {"Adam": adam_update,
                          "FOSI-Adam": fosi_adam_update_fn,
                          "HB": momentum_update,
                          "FOSI-HB": fosi_momentum_update_fn,
                          "GD": gd_update,
                          "FOSI-GD": fosi_sgd_update_fn}

            for optimizer_name, optimizer_update in optimizers.items():
                if 'Adam' in optimizer_name:
                    eta = 0.05
                elif optimizer_name == 'HB':
                    eta = 2 / (np.sqrt(eigenvalues[0]) + np.sqrt(eigenvalues[-1])) ** 2
                elif optimizer_name == 'FOSI-HB':
                    # Use FOSI's learning rate scaling technique with c=inf (no clipping):
                    # start with eta of Heavy-Ball and scale it.
                    eta = 2 / (np.sqrt(eigenvalues[0]) + np.sqrt(eigenvalues[-1])) ** 2
                    eta = eta * k_eigenvals[-1] / k_eigenvals[0]
                elif optimizer_name == 'GD':
                    eta = 2 / (eigenvalues[0] + eigenvalues[-1])
                elif optimizer_name == 'FOSI-GD':
                    # Use FOSI's learning rate scaling technique with c=inf (no clipping):
                    # start with eta of GD and scale it.
                    eta = 2 / (eigenvalues[0] + eigenvalues[-1])
                    eta = eta * k_eigenvals[-1] / k_eigenvals[0]
                scores, solutions = optimize(objective_fn, x_initial, n_iter, eta, beta1, beta2, optimizer_update)
                print('%s: f = %.10f' % (optimizer_name, scores[-1]))
                optimizers_scores[(n_dim, max_eigval, optimizer_name)] = [x.item() for x in jax.device_get(scores)]

    # Plot learning curves
    pickle.dump(optimizers_scores, open("test_results/optimizers_scores_quadratic.pkl", 'wb'))


if __name__ == "__main__":
    optimize_quadratic_funcs_with_varying_dim_and_cond_number()
    plot_quadratic_random_orthogonal_basis()
