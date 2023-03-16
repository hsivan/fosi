import pickle
import jax
import jax.numpy as np
import os
from jax.config import config
import numpy
from scipy.stats import special_ortho_group

from experiments.quadratic.quadratic_test_utils import fill_diagonal, get_x_initial, objective, optimize, adam_update, \
    fosi_adam_update, momentum_update, fosi_momentum_update, gd_update, fosi_gd_update
from fosi import get_ese_fn

config.update("jax_enable_x64", True)
numpy.random.seed(1234)


if not os.path.isdir('./test_results'):
    os.makedirs('./test_results')


def prepare_hessian(kappa, dim_non_diag, n_dim=100):
    hessian = np.eye(n_dim)

    diag = [1e-3 * (kappa ** i) for i in range(n_dim)]
    diag = diag[:dim_non_diag // 2] + diag[-dim_non_diag // 2:] + diag[dim_non_diag // 2:-dim_non_diag // 2]
    hessian = fill_diagonal(hessian, diag)

    # Build a PD block
    V = special_ortho_group.rvs(dim_non_diag)
    D = numpy.diag(diag[:dim_non_diag])
    A = V @ D @ V.T

    print("Zeta:", numpy.sum(A.diagonal() < numpy.sum(np.abs(A), axis=0) - A.diagonal()))

    hessian = hessian.at[:dim_non_diag, :dim_non_diag].set(A)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)

    print(eigenvalues)
    print("Condition number:", np.max(eigenvalues) / np.min(eigenvalues))

    return hessian, eigenvalues, eigenvectors


def run_grid_kappa_zeta():
    pkl_file_name = "test_results/quadratic_jax_kappa_zeta.pkl"
    optimizers_scores = {}
    # if os.path.isfile(pkl_file_name):
    #     optimizers_scores = pickle.load(open(pkl_file_name, 'rb'))

    dim_non_diag_arr = numpy.concatenate((np.arange(2, 20, 2), numpy.arange(20, 80, 10), numpy.arange(80, 102, 2)))
    kappa_arr = numpy.concatenate((numpy.arange(1.10, 1.14, 0.01), numpy.arange(1.14, 1.17, 0.002)))

    for dim_non_diag in dim_non_diag_arr:

        for kappa in kappa_arr:
            numpy.random.seed(1234)
            approx_k = 10

            hessian, eigenvalues, eigenvectors = prepare_hessian(kappa, dim_non_diag)

            objective_fn = lambda x, batch=None: objective(x, hessian, batch)

            x_initial = get_x_initial(objective_fn, eigenvectors)
            n_iter = 200
            eta = 0.1
            beta1 = 0.9  # factor for average gradient (first moment)
            beta2 = 0.999  # factor for average squared gradient (second moment)

            ese_fn = get_ese_fn(objective_fn, approx_k, [None], l_smallest=0, return_precision='64')
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
                    eta = 2 / (np.sqrt(eigenvalues[0]) + np.sqrt(eigenvalues[-1]))**2
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

                if optimizer_name not in optimizers_scores.keys():
                    optimizers_scores[optimizer_name] = []
                optimizers_scores[optimizer_name].append((dim_non_diag, kappa, jax.device_get(scores)[-1], jax.device_get(scores)))

    # Plot learning curves
    pickle.dump(optimizers_scores, open(pkl_file_name, 'wb'))


def turn_lr_for_adam(kappa=1.14, dim_non_diag=50):
    hessian, eigenvalues, eigenvectors = prepare_hessian(kappa, dim_non_diag)

    objective_fn = lambda x, batch=None: objective(x, hessian, batch)

    x_initial = get_x_initial(objective_fn, eigenvectors)
    n_iter = 200
    beta1 = 0.9  # factor for average gradient (first moment)
    beta2 = 0.999  # factor for average squared gradient (second moment)

    best_score = np.inf
    best_lr = 0.001

    for eta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        scores, solutions = optimize(objective_fn, x_initial, n_iter, eta, beta1, beta2, adam_update)
        print("LR:", eta, "score:", jax.device_get(scores)[-1])
        if jax.device_get(scores)[-1] < best_score:
            best_score = jax.device_get(scores)[-1]
            best_lr = eta

    print("Best LR:", best_lr)


def run_optimizers_with_different_learning_rates(kappa=1.14, dim_non_diag=50):
    hessian, eigenvalues, eigenvectors = prepare_hessian(kappa, dim_non_diag)

    objective_fn = lambda x, batch=None: objective(x, hessian, batch)

    x_initial = get_x_initial(objective_fn, eigenvectors)
    n_iter = 200
    beta1 = 0.9  # factor for average gradient (first moment)
    beta2 = 0.999  # factor for average squared gradient (second moment)
    approx_k = 10

    ese_fn = get_ese_fn(objective_fn, approx_k, [None], l_smallest=0, return_precision='64')
    k_eigenvals, k_eigenvecs = ese_fn(x_initial)
    fosi_adam_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_adam_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)
    fosi_momentum_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_momentum_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)
    fosi_sgd_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_gd_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)

    optimizers = {"Adam": adam_update,
                  "FOSI-Adam": fosi_adam_update_fn,
                  "HB": momentum_update,
                  "FOSI-HB (c=1)": fosi_momentum_update_fn,
                  "FOSI-HB (c=inf)": fosi_momentum_update_fn,
                  "GD": gd_update,
                  "FOSI-GD (c=1)": fosi_sgd_update_fn,
                  "FOSI-GD (c=inf)": fosi_sgd_update_fn}

    optimizers_scores = {}

    etas = numpy.concatenate((np.logspace(-5, -1, 50), np.linspace(0.10001, 1.0, 50), np.linspace(1.1, 10.0, 20)))

    for optimizer_name, optimizer_update in optimizers.items():
        for eta in etas:
            lr = eta
            if 'c=inf' in optimizer_name:
                lr = eta * k_eigenvals[-1] / k_eigenvals[0]
            scores, solutions = optimize(objective_fn, x_initial, n_iter, lr, beta1, beta2, optimizer_update)
            print(optimizer_name, "LR:", lr, "score:", jax.device_get(scores)[-1])
            if optimizer_name not in optimizers_scores.keys():
                optimizers_scores[optimizer_name] = []
            optimizers_scores[optimizer_name].append((eta, jax.device_get(scores)[-1]))

    lr_pkl_file_name = "test_results/quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    pickle.dump(optimizers_scores, open(lr_pkl_file_name, 'wb'))


def run_optimizers_with_different_momentums(kappa=1.14, dim_non_diag=50):
    hessian, eigenvalues, eigenvectors = prepare_hessian(kappa, dim_non_diag)

    objective_fn = lambda x, batch=None: objective(x, hessian, batch)

    x_initial = get_x_initial(objective_fn, eigenvectors)
    n_iter = 200
    approx_k = 10

    ese_fn = get_ese_fn(objective_fn, approx_k, [None], l_smallest=0, return_precision='64')
    k_eigenvals, k_eigenvecs = ese_fn(x_initial)
    fosi_adam_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_adam_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)

    optimizers = {"Adam": adam_update,
                  "FOSI-Adam": fosi_adam_update_fn}

    optimizers_scores = {}

    momentums = numpy.concatenate((np.linspace(0.2, 0.8, 25), np.linspace(0.8, 0.99, 25)))

    for beta2 in [0.99, 0.999, 0.9999]:
        for optimizer_name, optimizer_update in optimizers.items():
            for beta1 in momentums:
                lr = 0.05
                scores, solutions = optimize(objective_fn, x_initial, n_iter, lr, beta1, beta2, optimizer_update)
                print(optimizer_name, "momentum:", beta1, "score:", jax.device_get(scores)[-1])
                if optimizer_name not in optimizers_scores.keys():
                    optimizers_scores[optimizer_name] = []
                optimizers_scores[optimizer_name].append((beta2, beta1, jax.device_get(scores)[-1]))

    lr_pkl_file_name = "test_results/quadratic_jax_kappa_zeta_momentum_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    pickle.dump(optimizers_scores, open(lr_pkl_file_name, 'wb'))


def grid_search_lr_momentum_adam(kappa, dim_non_diag):
    hessian, eigenvalues, eigenvectors = prepare_hessian(kappa, dim_non_diag)

    objective_fn = lambda x, batch=None: objective(x, hessian, batch)

    x_initial = get_x_initial(objective_fn, eigenvectors)
    n_iter = 200
    approx_k = 10
    beta2 = 0.999

    ese_fn = get_ese_fn(objective_fn, approx_k, [None], l_smallest=0, return_precision='64')
    k_eigenvals, k_eigenvecs = ese_fn(x_initial)
    fosi_adam_update_fn = lambda x, g, m, v, t, eta, beta1, beta2, mn: fosi_adam_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs)

    optimizers = {"Adam": adam_update,
                  "FOSI-Adam": fosi_adam_update_fn}

    optimizers_scores = {}

    momentums = numpy.linspace(0.7, 1.0, 40, endpoint=False)
    lrs = numpy.logspace(-4, 1, 50)

    for lr in lrs:

        for optimizer_name, optimizer_update in optimizers.items():
            best_score = np.inf
            best_beta1 = None
            for beta1 in momentums:
                scores, solutions = optimize(objective_fn, x_initial, n_iter, lr, beta1, beta2,
                                             optimizers[optimizer_name])
                if jax.device_get(scores)[-1] < best_score:
                    best_score = jax.device_get(scores)[-1]
                    best_beta1 = beta1
            print(optimizer_name, "lr:", lr, "best_beta1:", best_beta1, "score:", best_score)
            if optimizer_name not in optimizers_scores.keys():
                optimizers_scores[optimizer_name] = []
            optimizers_scores[optimizer_name].append((lr, best_beta1, best_score))

    lr_pkl_file_name = "test_results/quadratic_jax_kappa_zeta_lr_momentum_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    pickle.dump(optimizers_scores, open(lr_pkl_file_name, 'wb'))


if __name__ == "__main__":
    turn_lr_for_adam()
    run_grid_kappa_zeta()

    kappa_dim_non_diag_tuples = zip([90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16])
    for dim_non_diag, kappa in kappa_dim_non_diag_tuples:
        run_optimizers_with_different_learning_rates(kappa, dim_non_diag)

    run_optimizers_with_different_momentums(1.12, 90)
    grid_search_lr_momentum_adam(1.12, 90)
