from matplotlib import rcParams, rcParamsDefault
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from experiments.utils.visualization_utils import get_figsize

V = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])
Q = np.array([[4.0, 0.0], [0.0, 1.0]])


def prep_domain_grid():
    X_domain = np.arange(-0.5, 1.05, 0.05)
    Y_domain = np.arange(-1, 0.5, 0.05)
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


def draw_fosi_illustration_2d():
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    other_f_color = 'grey'
    head_width = 0.05
    head_length = 0.05
    theta_color = 'blue'
    updated_theta_color = 'red'
    cmap = 'binary'
    vmin = 0
    vmax = 2.3125
    linewidth = 0.9

    fig1 = plt.figure(figsize=get_figsize(wf=0.33, hf=1.0))
    ax1 = fig1.add_axes([0.17, 0.16, 0.81, 0.83])
    ax12 = fig1.add_axes([0.01, 0.48, 0.43, 0.75], projection='3d')

    fig2 = plt.figure(figsize=get_figsize(wf=0.33, hf=1.0))
    ax2 = fig2.add_axes([0.17, 0.16, 0.81, 0.83])
    ax22 = fig2.add_axes([0.00+0.6, 0.46, 0.43, 0.75], projection='3d')

    fig3 = plt.figure(figsize=get_figsize(wf=0.33, hf=1.0))
    ax3 = fig3.add_axes([0.17, 0.16, 0.81, 0.83])
    ax32 = fig3.add_axes([0.00+0.6, 0.46, 0.43, 0.75], projection='3d')

    axs = [ax1, ax2, ax3]
    axs_2 = [ax12, ax22, ax32]

    for ax in axs_2:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel(r'$\theta_1$', labelpad=-18)
        ax.set_ylabel(r'$\theta_2$', labelpad=-18)
        ax.azim = -110  # default -60
        ax.elev = 20  # default 30
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_linewidth(0.3)
            axis.pane.set_facecolor('0.9')
            axis.pane.set_edgecolor('0.7')
            axis.pane.set_lw(0.3)
            axis.pane.set_alpha(1.0)

        ax.patch.set_facecolor('aliceblue')
        ax.patch.set_alpha(0.0)

    init_val = np.array([0.3, -0.9])
    eta = 0.3
    final_val_f1 = init_val - (1 / Q[0, 0]) * V[:, 0] * (V[:, 0] @ grad(init_val))  # step of size 1/lambda1 in the oposite direction of grad projection on v1
    grad_residual = grad(init_val) - V[:, 0] * (V[:, 0] @ grad(init_val))
    final_val_f2 = init_val - eta * grad_residual  # step of size eta in the direction of grad_residual (v2)
    final_val_f = init_val + (final_val_f1 - init_val) + (final_val_f2 - init_val)

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    Z = f(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    contour_set = axs[0].contour(X, Y, Z, cmap=cmap, alpha=0.8, levels=np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1]) + 0.01, vmin=-0.1, vmax=2.1, linewidths=linewidth)
    _ = axs[0].pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', alpha=0.5,  vmin=vmin, vmax=vmax)
    axs[0].plot([init_val[0]], [init_val[1]], marker='o', markersize=2, label='$p_0$', color=theta_color)
    axs[0].annotate(r'$\theta_0$', xy=(0.68, 0.01), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color=theta_color)
    axs[0].annotate(r'$\theta_1 = \theta_0 + d_1 + d_2$', xy=(0.56, 0.33), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color=updated_theta_color)
    axs[0].plot([final_val_f[0]], [final_val_f[1]], marker='o', markersize=2, label='$p_0$', color=updated_theta_color)
    axs[0].arrow(init_val[0], init_val[1], final_val_f[0] - init_val[0], final_val_f[1] - init_val[1], color='black', head_width=head_width, head_length=head_length, length_includes_head=True, zorder=2)

    ax12.plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.9, cmap=cm.coolwarm)

    Z = f1(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    axs[1].contour(X, Y, Z, contour_set.levels, cmap=cmap, alpha=0.5, vmin=-0.1, vmax=2.1, linewidths=linewidth)
    axs[1].pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', alpha=0.5, vmin=vmin, vmax=vmax)
    axs[1].plot([init_val[0]], [init_val[1]], marker='o', markersize=2, label='$p_0$', color=theta_color)
    axs[1].plot([final_val_f[0]], [final_val_f[1]], marker='o', markersize=2, label='$p_0$', color=updated_theta_color)
    axs[1].arrow(init_val[0], init_val[1], final_val_f1[0] - init_val[0], final_val_f1[1] - init_val[1], color='black', head_width=head_width, head_length=head_length, length_includes_head=True, zorder=2)
    axs[1].annotate(r'$d_1$', xy=(0.76, 0.07), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color='black')

    axs[1].arrow(final_val_f1[0], final_val_f1[1], final_val_f2[0] - init_val[0], final_val_f2[1] - init_val[1], color=other_f_color, head_width=0, head_length=0, length_includes_head=True, linestyle=':')
    axs[1].arrow(final_val_f1[0] + 0.9 * (final_val_f2[0] - init_val[0]), final_val_f1[1] + 0.9 * (final_val_f2[1] - init_val[1]), 0.1 * (final_val_f2[0] - init_val[0]), 0.1 * (final_val_f2[1] - init_val[1]), color=other_f_color,
                 head_width=head_width, head_length=head_length, length_includes_head=True)
    axs[1].annotate(r'$d_2$', xy=(0.8, 0.33), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color=other_f_color)

    ax22.plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.9, cmap=cm.coolwarm)

    Z = f2(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    axs[2].contour(X, Y, Z, contour_set.levels, cmap=cmap, alpha=0.5, vmin=-0.1, vmax=2.1, linewidths=linewidth)
    axs[2].pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', alpha=0.5, vmin=vmin, vmax=vmax)
    axs[2].plot([init_val[0]], [init_val[1]], marker='o', markersize=2, label='$p_0$', color=theta_color)
    axs[2].plot([final_val_f[0]], [final_val_f[1]], marker='o', markersize=2, label='$p_0$', color=updated_theta_color)
    axs[2].arrow(init_val[0], init_val[1], final_val_f2[0] - init_val[0], final_val_f2[1] - init_val[1], color='black', head_width=head_width, head_length=head_width, length_includes_head=True, zorder=2)
    axs[2].annotate(r'$d_2$', xy=(0.58, 0.13), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color='black')

    axs[2].arrow(final_val_f2[0], final_val_f2[1], final_val_f1[0] - init_val[0], final_val_f1[1] - init_val[1], color=other_f_color, head_width=0, head_length=0, length_includes_head=True, linestyle=':')
    axs[2].arrow(final_val_f2[0] + 0.9 * (final_val_f1[0] - init_val[0]), final_val_f2[1] + 0.9 * (final_val_f1[1] - init_val[1]), 0.1 * (final_val_f1[0] - init_val[0]), 0.1 * (final_val_f1[1] - init_val[1]), color=other_f_color,
                 head_width=head_width, head_length=head_length, length_includes_head=True)
    axs[2].annotate(r'$d_1$', xy=(0.49, 0.29), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.8, color=other_f_color)

    ax32.plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.9, cmap=cm.coolwarm)

    for i in range(6):
        init_val = final_val_f
        final_val_f1 = init_val - (1 / Q[0, 0]) * V[:, 0] * (V[:, 0] @ grad(init_val))  # step of size 1/lambda1 in the oposite direction of grad projection on v1
        grad_residual = grad(init_val) - V[:, 0] * (V[:, 0] @ grad(init_val))
        final_val_f2 = init_val - eta * grad_residual  # step of size eta in the direction of grad_residual (v2)
        final_val_f = init_val + (final_val_f1 - init_val) + (final_val_f2 - init_val)

        axs[0].arrow(init_val[0], init_val[1], final_val_f[0] - init_val[0], final_val_f[1] - init_val[1], color='black', head_width=head_width, head_length=head_length, length_includes_head=True, zorder=2)

        axs[1].arrow(final_val_f1[0], final_val_f1[1], final_val_f2[0] - init_val[0], final_val_f2[1] - init_val[1],
                     color=other_f_color, head_width=0, head_length=0, length_includes_head=True, linestyle=':')
        axs[1].arrow(final_val_f1[0] + 0.9 * (final_val_f2[0] - init_val[0]), final_val_f1[1] + 0.9 * (final_val_f2[1] - init_val[1]), 0.1 * (final_val_f2[0] - init_val[0]), 0.1 * (final_val_f2[1] - init_val[1]),
                     color=other_f_color, head_width=head_width, head_length=head_length, length_includes_head=True)

        axs[2].arrow(init_val[0], init_val[1], final_val_f2[0] - init_val[0], final_val_f2[1] - init_val[1], color='black', head_width=head_width, head_length=head_length, length_includes_head=True, zorder=2)

    for ax in axs:
        ax.set_xticks([0, 1])
        ax.set_yticks([-1, 0])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ax.tick_params(pad=1)
        ax.set_xlabel(r'$\theta_1$', labelpad=-5)
        ax.set_ylabel(r'$\theta_2$', labelpad=-11)

        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    fig1.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.98, wspace=0.20)

    fig2.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.98, wspace=0.20)
    fig3.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.98, wspace=0.20)
    fig1.savefig('figures/' + 'fosi_illustration_f.pdf')
    fig2.savefig('figures/' + 'fosi_illustration_f1.pdf')
    fig3.savefig('figures/' + 'fosi_illustration_f2.pdf')
    #plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    draw_fosi_illustration_2d()
