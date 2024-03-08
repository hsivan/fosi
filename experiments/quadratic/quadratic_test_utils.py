from jax import numpy as np, grad


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = np.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def get_x_initial(objective, eigenvectors, n_dim=100):
    x_initial = np.ones(n_dim) * 0.5
    x_initial = x_initial.at[1].set(1.0)
    x_initial = x_initial @ eigenvectors.T
    print("f(x0)=", objective(x_initial))
    return x_initial


def objective(x, hessian, batch=None):
    if len(x.shape) == 1:
        return 0.5 * x @ hessian @ x.T
    # For plotting
    return 0.5 * np.sum(x * (hessian @ x.T).T, axis=1)


def optimize(objective, x_initial, n_iter, eta, beta1, beta2, update_func):
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
        g = grad(objective)(x)
        x, m, v, mn = update_func(x, g, m, v, t, eta, beta1, beta2, mn)
        score = objective(x)
        scores.append(score)
        solutions.append(x.copy())
        # print('>%d f(%s) = %.10f' % (t, x, score))
        # print('>%d = %.10f' % (t, score))
    return scores, solutions


def adam_update(x, g, m, v, t, eta, beta1, beta2, mn):
    eps = 1e-8
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g ** 2
    mhat = m / (1.0 - beta1 ** (t + 1))
    vhat = v / (1.0 - beta2 ** (t + 1))
    x = x - eta * mhat / (np.sqrt(vhat) + eps)

    return x, m, v, mn


def fosi_adam_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs):
    eps = 1e-8
    learning_rates = 1.0 / k_eigenvals

    g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
    newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

    mn = beta1 * mn + (1.0 - beta1) * newton_direction

    # For Adam direction use g_residual instead of g
    m = beta1 * m + (1.0 - beta1) * g_residual
    v = beta2 * v + (1.0 - beta2) * g_residual ** 2
    mhat = m / (1.0 - beta1 ** (t + 1))
    vhat = v / (1.0 - beta2 ** (t + 1))

    adam_direction = mhat / (np.sqrt(vhat) + eps)

    # Reduce from adam_direction its projection on k_eigenvecs. This makes adam_direction and mn orthogonal.
    adam_direction = adam_direction - (k_eigenvecs @ adam_direction) @ k_eigenvecs

    x = x - eta * adam_direction - mn

    return x, m, v, mn


def momentum_update(x, g, m, v, t, eta, beta1, beta2, mn):
    m = beta1 * m + g
    x = x - eta * m
    return x, m, v, mn


def fosi_momentum_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs):
    learning_rates = 1.0 / k_eigenvals

    g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
    newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

    # Note: this momentum term is mathematically equivalent to mn = beta1 * mn + newton_direction
    # and then using FOSI's alpha 0.1: x = x - eta * m - 0.1 * mn.
    # We get similar momentum term for FOSI as for Heavy-Ball without using alpha.
    mn = beta1 * mn + (1.0 - beta1) * newton_direction

    # For momentum direction use g_residual instead of g
    m = beta1 * m + g_residual

    x = x - eta * m - mn

    return x, m, v, mn


def gd_update(x, g, m, v, t, eta, beta1, beta2, mn):
    x = x - eta * g
    return x, m, v, mn


def fosi_gd_update(x, g, m, v, t, eta, beta1, beta2, mn, k_eigenvals, k_eigenvecs):
    learning_rates = 1.0 / k_eigenvals

    g_residual = g - k_eigenvecs.T @ (k_eigenvecs @ g)
    newton_direction = k_eigenvecs.T @ ((k_eigenvecs @ g) * learning_rates)

    x = x - eta * g_residual - newton_direction

    return x, m, v, mn
