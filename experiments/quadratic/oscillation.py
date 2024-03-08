import pickle
import jax
import jax.numpy as np
from jax.config import config
from matplotlib import pyplot as plt

from experiments.quadratic.quadratic_test_utils import fill_diagonal, get_x_initial, objective, optimize, adam_update, momentum_update
from experiments.visualization.visualization_utils import get_figsize

config.update("jax_enable_x64", True)


def prepare_hessian(max_eigval):
    n_dim = 2
    eigenvectors = np.array(((1/np.sqrt(2), -1/np.sqrt(2)), (1/np.sqrt(2), 1/np.sqrt(2))))  # 45 degree rotation

    eigenvalues = np.ones(n_dim)
    eigenvalues = eigenvalues.at[0].set(max_eigval)
    eigenvalues = eigenvalues.at[1].set(0.00001)
    if n_dim > 2:
        for i in range(2, n_dim):
            eigenvalues = eigenvalues.at[i].set(eigenvalues[i - 1] / 1.5)
    eigenvalues_matrix = fill_diagonal(np.zeros_like(eigenvectors), eigenvalues)
    hessian = eigenvectors @ eigenvalues_matrix @ eigenvectors.T
    print("Condition number:", eigenvalues[0] / eigenvalues[-1])

    return hessian, eigenvalues, eigenvectors


def get_x_initial(objective, eigenvectors):
    n_dim = 2
    #x_initial = np.array((-0.5, 1.5))
    x_initial = np.ones(n_dim) * 0.5
    x_initial = x_initial.at[1].set(1.0)
    x_initial = x_initial @ eigenvectors.T
    print("f(x0)=", objective(x_initial))
    return x_initial

def get_x_initial_(objective, eigenvectors):
    n_dim = 2
    #x_initial = np.array((-0.5, 1.5))
    x_initial = np.ones(n_dim) * 0.01
    x_initial = x_initial.at[1].set(1.0)
    x_initial = x_initial @ eigenvectors.T
    print("f(x0)=", objective(x_initial))
    return x_initial


def prep_domain_grid():
    X_domain = np.arange(-2, 2.05, 0.05)
    Y_domain = np.arange(-2, 2.05, 0.05)
    X, Y = np.meshgrid(X_domain, Y_domain)
    domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    return X, Y, domain_grid_as_vector


def plot_func_contour(objective_fn, solutions):
    X, Y, domain_grid_as_vector = prep_domain_grid()
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(columnwidth=487.8225, wf=0.3, hf=0.4), sharex=True, sharey=True)
    cmap = 'binary'

    Z = objective_fn(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    contour_set = axs[0].contour(X, Y, Z, cmap=cmap, alpha=0.8, linewidths=0.9)
    #_ = axs[0].pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', alpha=0.5)
    axs[0].plot([x for (x, y) in solutions], [y for (x, y) in solutions])

    plt.show()


def plot_loss(scores):
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(columnwidth=487.8225, wf=0.3, hf=0.4), sharex=True, sharey=True)
    axs[0].plot(range(len(scores)), scores)
    axs[0].set_yscale('log')

    plt.show()


def optimize_quadratic_funcs_oscillation():
    max_eigvals = [50, 50]
    optimizers_scores = {}
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(columnwidth=487.8225, wf=0.6), sharex=True, sharey=True)

    for idx, max_eigval in enumerate(max_eigvals):

        hessian, eigenvalues, eigenvectors = prepare_hessian(max_eigval)

        objective_fn = lambda x, batch=None: objective(x, hessian, batch)

        x_initial = get_x_initial(objective_fn, eigenvectors)
        if idx == 1:
            x_initial = get_x_initial_(objective_fn, eigenvectors)
        n_iter = 200
        eta = 0.1
        beta1 = 0.9  # factor for average gradient (first moment)
        beta2 = 0.999  # factor for average squared gradient (second moment)

        optimizers = {"Adam": adam_update,
                      "HB": momentum_update}
        optimizers = {"HB": momentum_update}

        for optimizer_name, optimizer_update in optimizers.items():
            if 'Adam' in optimizer_name:
                eta = 0.05
            elif optimizer_name == 'HB':
                eta = 0.1 / (np.sqrt(eigenvalues[0]) + np.sqrt(eigenvalues[-1])) ** 2
            scores, solutions = optimize(objective_fn, x_initial, n_iter, eta, beta1, beta2, optimizer_update)
            print('%s: f = %.10f' % (optimizer_name, scores[-1]))
            optimizers_scores[(max_eigval, optimizer_name)] = [x.item() for x in jax.device_get(scores)]

        axs[0].plot(range(len(scores)), scores, label=str(max_eigval), linewidth=0.9)

    axs[0].set_yscale('log')
    axs[0].legend()
    plt.show()


    # Plot learning curves
    #pickle.dump(optimizers_scores, open("test_results/optimizers_scores_quadratic.pkl", 'wb'))


if __name__ == "__main__":
    optimize_quadratic_funcs_oscillation()
