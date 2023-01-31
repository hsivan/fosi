import pickle

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random, grad
from jax.config import config

from fosi.extreme_spectrum_estimation import get_ese_fn

from plot_quadratic import plot_quadratic_random_orthogonal_basis_gd

config.update("jax_enable_x64", True)


n_dims = [100]
rng_key = random.PRNGKey(0)

optimizers_scores = {}


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = np.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


for n_dim_idx, n_dim in enumerate(n_dims):

    '''eigenvectors = np.eye(n_dim)
    #for i in range(0, n_dim, 2):
    #    eigenvectors[i][i], eigenvectors[i][i+1], eigenvectors[i+1][i], eigenvectors[i+1][i+1] = 0.5, 0.5, -0.5, 0.5
    eigenvectors = eigenvectors.at[0, 0].set(0.5)
    eigenvectors = eigenvectors.at[0, 1].set(0.5)
    eigenvectors = eigenvectors.at[1, 0].set(-0.5)
    eigenvectors = eigenvectors.at[1, 1].set(0.5)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1)'''

    A = random.uniform(rng_key, shape=(n_dim, n_dim))
    B = np.dot(A, A.transpose())
    _, eigenvectors = np.linalg.eigh(B)

    #eigenvectors = np.eye(n_dim)

    num_large_eigenvals = 10

    approx_newton_k = 9
    assert approx_newton_k <= n_dim

    eigenvalues = np.ones(n_dim)

    tt = np.linspace(9, 10, num_large_eigenvals)[::-1]
    for i in range(0, num_large_eigenvals):
        eigenvalues = eigenvalues.at[i].set(tt[i])

    tt = np.linspace(0.01, 0.1, n_dim - num_large_eigenvals)[::-1]
    for i in range(num_large_eigenvals, n_dim):
        eigenvalues = eigenvalues.at[i].set(tt[i - num_large_eigenvals])


    eigenvalues_matrix = fill_diagonal(np.zeros_like(eigenvectors), eigenvalues)
    hessian = eigenvectors @ eigenvalues_matrix @ eigenvectors.T
    #print("hessian:\n", hessian)
    print("Condition number:", eigenvalues[0] / eigenvalues[-1])


    def plot_trajectory(solutions, title, ax):
        if n_dim != 2:
            return
        r_min, r_max = -1.25, 1.25
        xaxis = np.arange(r_min, r_max, 0.1)
        yaxis = np.arange(r_min, r_max, 0.1)
        x, y = np.meshgrid(xaxis, yaxis)
        shape = x.shape[0]
        x_arr = np.concatenate((np.reshape(x, (shape**2, 1)), np.reshape(y, (shape**2, 1))), axis=1)
        results = objective(x_arr)
        results = np.reshape(results, (shape, shape))
        levels = np.linspace(0, 60, num=50)
        solutions = np.asarray(solutions)

        #fig, ax = plt.subplots()
        ax.contourf(x, y, results, levels=levels, cmap='jet')
        ax.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
        ax.set_title(title)


    # gradient descent optimization with adam for a two-dimensional test function.
    # Taken from: https://machinelearningmastery.com/adam-optimization-from-scratch

    def objective(x, batch=None):
        if len(x.shape) == 1:
            return 0.5 * x @ hessian @ x.T  # + np.sum(np.sin(4 * x)) / 4**2 + n_dim / 4**2


    def derivative(x):
        return grad(objective)(x)


    def adam_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        eps = 1e-8
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        mhat = m / (1.0 - beta1 ** (t + 1))
        vhat = v / (1.0 - beta2 ** (t + 1))
        x = x - alpha * mhat / (np.sqrt(vhat) + eps)

        #if t == 50:
        #    adam_effective_condition_number(alpha / (np.sqrt(vhat) + eps))

        return x, m, v, mn


    def adam_effective_condition_number(adam_inverse_preconditioner_vector):
        adam_inverse_preconditioner = np.zeros_like(hessian)
        adam_inverse_preconditioner = fill_diagonal(adam_inverse_preconditioner, adam_inverse_preconditioner_vector)
        effective_hessian = adam_inverse_preconditioner @ hessian
        eigs, _ = np.linalg.eigh(effective_hessian)
        adam_condition_number = np.max(eigs) / np.min(eigs)
        print("adam_condition_number:", adam_condition_number, "max eigenvalue:", np.max(eigs))
        num_bins = 20
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(jax.device_get(eigs), bins=num_bins)
        ax.set_xlabel('eigenvalue')
        ax.set_title('Adam')
        plt.show()


    def fosi_adam_effective_condition_number(adam_inverse_preconditioner_vector, u):
        u = u[::-1]  # Revert u since ese_fn returns k_eigenvals sorted from smallest to largest and we need it to match the eigenvectors, which are sorted from largest to smallest
        u_diagonal_matrix = fill_diagonal(np.zeros((approx_newton_k, approx_newton_k)), u)
        fosis_part_preconditioner_matrix = eigenvectors[:, :approx_newton_k] @ u_diagonal_matrix @ eigenvectors[:, :approx_newton_k].T

        adams_part_diagonal_matrix = fill_diagonal(np.zeros_like(hessian), adam_inverse_preconditioner_vector)
        adams_part_preconditioner_matrix = adams_part_diagonal_matrix @ (eigenvectors[:, approx_newton_k:] @ eigenvectors[:, approx_newton_k:].T)

        effective_hessian = (fosis_part_preconditioner_matrix + adams_part_preconditioner_matrix) @ hessian
        eigs, _ = np.linalg.eigh(effective_hessian)
        condition_number = np.max(eigs) / np.min(eigs)
        print("fosi_adam_condition_number:", condition_number, "max eigenvalue:", np.max(eigs))
        num_bins = 20
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(jax.device_get(eigs), bins=num_bins)
        ax.set_xlabel('eigenvalue')
        ax.set_title('FOSI w/ Adam')
        plt.show()

        '''fig, ax = plt.subplots(1, 1)
        tt = np.ones(n_dim)
        tt = tt.at[approx_newton_k:].set((eigenvalues * adam_inverse_preconditioner_vector)[approx_newton_k:])
        n, bins, patches = ax.hist(jax.device_get(tt), bins=num_bins)
        ax.set_xlabel('eigenvalue')
        ax.set_title('FOSI w/ Adam (theory)')
        plt.show()'''



    def adam_with_approx_newton_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        eps = 1e-8
        #k_eigenvals, k_eigenvecs = ese_fn(x)
        learning_rates = 1.0 / k_eigenvals

        g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
        approx_newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

        mn = beta1 * mn + (1.0 - beta1) * approx_newton_direction

        # For Adam direction use g_residual instead of g
        m = beta1 * m + (1.0 - beta1) * g_residual
        v = beta2 * v + (1.0 - beta2) * g_residual**2
        mhat = m / (1.0 - beta1 ** (t + 1))
        vhat = v / (1.0 - beta2 ** (t + 1))

        update_adam = mhat / (np.sqrt(vhat) + eps)
        '''projection_coeff = (update_adam @ mn) / (np.linalg.norm(update_adam)**2)
        projection_coeff = np.min(np.array([projection_coeff, 1.0 - 1e-4]))
        projection_coeff = np.max(np.array([projection_coeff, 0.0]))
        projection_of_update_newton_on_update_adam = projection_coeff * update_adam
        mn = mn - projection_of_update_newton_on_update_adam'''

        x = x - alpha * update_adam - mn

        #if t == 50:
        #    fosi_adam_effective_condition_number(alpha / (np.sqrt(vhat) + eps), learning_rates)

        return x, m, v, mn


    def rmsprop_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        eps = 1e-8
        sg = g ** 2.0
        m = beta1 * m + (1.0 - beta1) * sg
        x = x - alpha * g / (np.sqrt(m) + eps)
        return x, m, v, mn


    def momentum_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        m = beta1 * m + g
        x = x - alpha * m
        return x, m, v, mn


    def momentum_with_approx_newton_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        #k_eigenvals, k_eigenvecs = ese_fn(x)
        learning_rates = np.abs(1.0 / k_eigenvals)

        g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
        approx_newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

        mn = beta1 * mn + (1.0 - beta1) * approx_newton_direction

        # For momentum direction use g_residual instead of g
        m = beta1 * m + g_residual

        x = x - alpha * m - mn

        return x, m, v, mn


    def gd_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        x = x - alpha * g
        return x, m, v, mn


    def gd_with_approx_newton_update(x, g, m, v, t, alpha, beta1, beta2, mn):
        #k_eigenvals, k_eigenvecs = ese_fn(x)
        learning_rates = np.abs(1.0 / k_eigenvals)

        g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
        approx_newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

        x = x - alpha * g_residual - approx_newton_direction

        return x, m, v, mn


    def optimize(objective, derivative, x_initial, n_iter, alpha, beta1, beta2, update_func):
        solutions = []
        scores = []
        x = x_initial.copy()
        solutions.append(x.copy())
        score = objective(x)
        scores.append(score)
        # initialize first and second moments
        mn = np.zeros_like(x)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        for t in range(n_iter):
            g = derivative(x)
            x, m, v, mn = update_func(x, g, m, v, t, alpha, beta1, beta2, mn)
            score = objective(x)
            scores.append(score)
            solutions.append(x.copy())
            #print('>%d f(%s) = %.10f' % (t, x, score))
            #print('>%d = %.10f' % (t, score))
        return scores, solutions


    #config.update("jax_enable_x64", False)

    x_initial = np.ones(n_dim) * 0.5
    x_initial = x_initial.at[1].set(1.0)
    x_initial = x_initial @ eigenvectors.T
    #print("x_initial:", x_initial)
    n_iter = 250
    alpha = 0.001  # optimal steps size 1/lambda_max
    beta1 = 0.9  # factor for average gradient (first moment)
    beta2 = 0.999  # factor for average squared gradient (second moment)

    ese_fn = get_ese_fn(objective, n_dim, approx_newton_k, [None], return_precision='64')
    k_eigenvals, k_eigenvecs = ese_fn(x_initial)
    print("k_eigenvals:", k_eigenvals)

    '''optimizers = {"GD": gd_update,
                  "FOSI w/ GD": gd_with_approx_newton_update,
                  "Heavy-Ball": momentum_update,
                  "FOSI w/ Heavy-Ball": momentum_with_approx_newton_update}'''
    optimizers = {"GD": gd_update,
                  "FOSI w/ GD": gd_with_approx_newton_update}

    for i, (optimizer_name, optimizer_update) in enumerate(optimizers.items()):
        if 'GD' in optimizer_name:
            alpha = 0.001
        else:
            alpha = 0.0001
        if 'FOSI' in optimizer_name:
            effective_condition_number = 1 / (alpha * eigenvalues[-1])
            print("effective_condition_number:", effective_condition_number)
        scores, solutions = optimize(objective, derivative, x_initial, n_iter, alpha, beta1, beta2, optimizer_update)
        print('%s: f = %.10f' % (optimizer_name, scores[-1]))
        optimizers_scores[(n_dim, jax.device_get(eigenvalues[0]).item(), optimizer_name)] = [x.item() for x in jax.device_get(scores)]


# Plot learning curves
pickle.dump(optimizers_scores, open("./optimizers_scores_quadratic_gd.pkl", 'wb'))
plot_quadratic_random_orthogonal_basis_gd()
