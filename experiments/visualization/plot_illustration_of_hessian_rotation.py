import os
from matplotlib import rcParams, rcParamsDefault
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from experiments.visualization.visualization_utils import get_figsize


def prep_domain_grid():
    X_domain = np.arange(-1, 1.05, 0.05)
    Y_domain = np.arange(-1, 1.05, 0.05)
    X, Y = np.meshgrid(X_domain, Y_domain)
    domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    return X, Y, domain_grid_as_vector


def f(X):
    H = V @ Q @ V.T
    res = 0.5 * np.sum((X @ H) * X, axis=1)
    return res


def f1(X):
    H = V @ (Q * np.array([[1, 0], [0, 0]])) @ V.T
    res = 0.5 * np.sum((X @ H) * X, axis=1)
    return res


def f2(X):
    H = V @ (Q * np.array([[0, 0], [0, 1]])) @ V.T
    res = 0.5 * np.sum((X @ H) * X, axis=1)
    return res


def grad(X):
    H = V @ Q @ V.T
    res = X @ H
    return res


def plot_3f_function(suffix):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4*3})
    rcParams.update({'font.size': 5.8*3})

    fig1 = plt.figure(figsize=get_figsize(wf=0.33*3, hf=1.0))
    ax1 = fig1.add_subplot(projection='3d')

    axs = [ax1]

    for ax in axs:
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.azim = -110  # default -60
        ax.elev = 20  # default 30
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_linewidth(0.3*3)
            axis.pane.set_facecolor('0.9')
            axis.pane.set_edgecolor('0.7')
            axis.pane.set_lw(0.3*3)
            axis.pane.set_alpha(1.0)

        ax.patch.set_facecolor('aliceblue')
        ax.patch.set_alpha(0.0)

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    Z = f(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    ax1.plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.9, cmap=cm.coolwarm)

    fig1.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.98, wspace=0.20)
    fig1.savefig('figures/' + 'fosi_illustration_f_3d_for_presentation_' + suffix +'.png')
    #plt.show()
    plt.close(fig1)
    rcParams.update(rcParamsDefault)


def plot_illustration(suffix):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4*3})
    rcParams.update({'font.size': 5.8*3})

    other_f_color = 'grey'
    head_width = 0.05*1.2
    head_length = 0.05*1.5
    theta_color = 'blue'
    updated_theta_color = 'red'
    cmap = 'binary'
    vmin = 0
    vmax = 2.3125
    linewidth = 0.9*3
    markersize = 2*3
    fontsize = 5.8*3

    fig1 = plt.figure(figsize=get_figsize(wf=0.33*4, hf=1.0))
    ax1 = fig1.add_subplot()
    ax1.set_aspect('equal', adjustable='box')

    axs = [ax1]

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    Z = f(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    contour_set = axs[0].contour(X, Y, Z, cmap=cmap, alpha=0.8, levels=np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1]) + 0.01, linewidths=linewidth)
    _ = axs[0].pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', alpha=0.5)

    for ax in axs:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
        #ax.tick_params(axis='x', colors='w')
        #ax.tick_params(axis='y', colors='w')
        #ax.tick_params(pad=1)
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')

        ax.spines['right'].set_linewidth(0.5*3)
        ax.spines['top'].set_linewidth(0.5*3)
        ax.spines['bottom'].set_linewidth(0.5*3)
        ax.spines['left'].set_linewidth(0.5*3)
        ax.tick_params(width=0.5*3)

    plt.tight_layout()
    #fig1.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.98, wspace=0.20)
    fig1.savefig('figures/' + 'fosi_illustration_of_hessina_rotation_' + suffix + '.png')
    #plt.show()
    plt.close(fig1)

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    if not os.path.isdir('./figures'):
        os.makedirs('./figures')

    Q = np.array([[4.0, 0.0], [0.0, 1.0]])

    # No rotation
    V = np.array([[1, 0], [0, 1]])

    plot_3f_function('no_rot')
    plot_illustration('no_rot')

    # Rotation
    V = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])

    plot_3f_function('rot')
    plot_illustration('rot')
