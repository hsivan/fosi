import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from matplotlib import ticker

from experiments.visualization.visualization_utils import get_figsize, set_rc_params


def plot_quadratic_random_orthogonal_basis(fig_file_name="quadratic_random_orthogonal_basis.pdf", pkl_file="optimizers_scores_quadratic.pkl"):
    set_rc_params()

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()

    # Plot learning curves
    fig_learning_curve, ax_learning_curve = plt.subplots(len(n_dims), len(max_eigvals), figsize=get_figsize(columnwidth=487.8225, hf=0.4), sharex=True)

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve[n_dim_idx][max_eigval_idx]
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name, color=optimizer_color, linewidth=0.8, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI-' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name, color=optimizer_color, linewidth=0.8, linestyle="-")

            ax.set_yscale('log')

            ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                        xy=(1, 0), xycoords='axes fraction',
                        xytext=(-10, 57), textcoords='offset pixels',
                        horizontalalignment='right',
                        verticalalignment='bottom')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            if n_dim_idx == 1:
                ax.set_xlabel('iteration')
            if max_eigval_idx % len(max_eigvals) == 0:
                ax.set_ylabel(r'$f(\theta)$')

    handles, labels = ax_learning_curve[0][0].get_legend_handles_labels()
    fig_learning_curve.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.93, bottom=0.12, left=0.07, right=0.99, wspace=0.3, hspace=0.1)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_random_orthogonal_basis_4_funcs(fig_file_name="quadratic_random_orthogonal_basis_4_funcs.pdf", pkl_file="optimizers_scores_quadratic.pkl"):
    scaling_factor = 1.0 if 'pdf' in fig_file_name else 1.5
    y_shift = 0 if 'pdf' in fig_file_name else 30
    set_rc_params(scaling_factor)

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()

    # Plot only extreme dimensions and extreme eigvals
    n_dims = [n_dims[0], n_dims[-1]]
    max_eigvals = [max_eigvals[0], max_eigvals[-1]]

    # Plot learning curves
    fig_learning_curve, ax_learning_curve = plt.subplots(len(n_dims), len(max_eigvals), figsize=get_figsize(wf=scaling_factor, hf=0.55), sharex=True, sharey=True)

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve[n_dim_idx][max_eigval_idx]
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name, color=optimizer_color, linewidth=0.8*scaling_factor, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI-' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name, color=optimizer_color, linewidth=0.8*scaling_factor, linestyle="-")

            ax.set_yscale('log')

            if max_eigval_idx == 0:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 35+y_shift), textcoords='offset pixels',
                            horizontalalignment='right',
                            verticalalignment='bottom')
            else:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 30+y_shift), textcoords='offset pixels',
                            horizontalalignment='right',
                            verticalalignment='bottom')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            if n_dim_idx == 1:
                ax.set_xlabel('iteration')
            if max_eigval_idx % len(max_eigvals) == 0:
                ax.set_ylabel(r'$f(\theta)$')

    handles, labels = ax_learning_curve[0][0].get_legend_handles_labels()
    fig_learning_curve.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.15, right=0.99, wspace=0.13, hspace=0.13)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_random_orthogonal_basis_4_funcs_flat(fig_file_name="quadratic_random_orthogonal_basis_4_funcs_flat.pdf", pkl_file="optimizers_scores_quadratic.pkl"):
    scaling_factor = 1.0 if 'pdf' in fig_file_name else 1.5
    y_shift = 0 if 'pdf' in fig_file_name else 30
    set_rc_params(scaling_factor)

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()

    # Plot only extreme dimensions and extreme eigvals
    n_dims = [n_dims[0], n_dims[-1]]
    max_eigvals = [max_eigvals[0], max_eigvals[-1]]

    # Plot learning curves
    fig_learning_curve, ax_learning_curve = plt.subplots(1, len(n_dims) + len(max_eigvals), figsize=get_figsize(columnwidth=397.48499, wf=scaling_factor, hf=0.2), sharex=True, sharey=True)

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve[n_dim_idx*len(max_eigvals) + max_eigval_idx]
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name, color=optimizer_color, linewidth=0.8*scaling_factor, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI-' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name, color=optimizer_color, linewidth=0.8*scaling_factor, linestyle="-")

            ax.set_yscale('log')

            if max_eigval_idx == 0:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 35+y_shift), textcoords='offset pixels',
                            horizontalalignment='right',
                            verticalalignment='bottom')
            else:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 30+y_shift), textcoords='offset pixels',
                            horizontalalignment='right',
                            verticalalignment='bottom')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            ax.set_xlabel('iteration')
            if max_eigval_idx + n_dim_idx == 0:
                ax.set_ylabel(r'$f(\theta)$')

    handles, labels = ax_learning_curve[0].get_legend_handles_labels()
    fig_learning_curve.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.87, bottom=0.29, left=0.08, right=0.99, wspace=0.13, hspace=0.13)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_random_orthogonal_basis_gd(fig_file_name="quadratic_random_orthogonal_basis_gd.pdf"):
    set_rc_params()

    optimizers_scores = pickle.load(open(root_result_folder + "optimizers_scores_quadratic_gd.pkl", 'rb'))
    optimizers_colors = {'GD': 'tab:blue'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()  # Only one
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()  # Only one

    # Plot learning curves
    fig_learning_curve, ax_learning_curve = plt.subplots(1, 1, figsize=get_figsize(wf=0.6, hf=0.45))

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name, color=optimizer_color, linewidth=0.8, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI-' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name, color=optimizer_color, linewidth=0.8, linestyle="-")
            ax.set_yticks([0, 5, 10, 15])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            ax.set_xlabel('iteration')
            ax.set_ylabel(r'$f(\theta)$')

    fig_learning_curve.legend(framealpha=0, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.05), ncol=1)
    plt.subplots_adjust(top=0.99, bottom=0.38, left=0.18, right=0.99, wspace=0.3, hspace=0.1)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def get_min_and_max(optimizers_scores, b_ignore_gd=True):
    min = np.inf
    max = 0.0
    for idx, optimizer in enumerate(optimizers_scores.keys()):
        # Ignore GD optimizer
        if b_ignore_gd and 'GD' in optimizer:
            continue
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: (val[0], val[1]))
        res = list(zip(*res))
        score_arr = np.array(res[2])
        min = np.min(score_arr) if np.min(score_arr) < min else min
        max = np.max(score_arr) if np.max(score_arr) > max else max
    return min, max


def filter_results_to_range(optimizers_scores, min_kappa=1.1, min_zeta=30):
    new_optimizers_scores = {}

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]
        new_res = [(w, x, y, z) for (w, x, y, z) in res if w >= min_zeta and x >= min_kappa]
        new_optimizers_scores[optimizer] = new_res

    return new_optimizers_scores


zeta_vals_to_show_learning_curves, beta_vals_to_show_learning_curves = [90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16]
zeta_val_to_show_horizontal_cut = 80
beta_val_to_show_vertical_cut = 1.13
beta_val_adam_tuning, zeta_val_adam_tuning = 1.14, 50


def plot_quadratic_jax_kappa_zeta(fig_file_name="quadratic_jax_kappa_zeta.pdf", pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_scores = filter_results_to_range(optimizers_scores)
    vmin, vmax = get_min_and_max(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    num_rows = num_optimizers // num_columns
    fig_learning_curve, axs = plt.subplots(num_rows, num_columns, figsize=get_figsize(hf=0.9), sharex=True, sharey=True)
    contourf_set = None

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        ax = axs[idx // num_columns][idx % num_columns]

        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: (val[0], val[1]))
        res = list(zip(*res))

        dim_non_diag_arr = np.array(res[0])
        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])
        shape = (np.unique(dim_non_diag_arr).shape[0], np.unique(kappa_arr).shape[0])

        cs = ax.contourf(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape), score_arr.reshape(shape),
                         locator=ticker.LogLocator(base=2), vmin=vmin, vmax=vmax,
                         levels=[2**i for i in range(-10, 3)], extend='both', cmap='binary')  # cmap=cm.PuBu_r
        cs.cmap.set_over('purple')
        cs.cmap.set_under('yellow')
        cs.changed()

        if optimizer == 'HB':
            contourf_set = cs

        ax.set_title(optimizer, pad=3)
        ax.set_ylim(30, 100)

        if idx // num_columns == (num_rows-1):
            ax.set_xlabel(r'$b$')
        if idx % num_columns == 0:
            ax.set_ylabel(r'$\zeta$')

        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

        print("Optimizer:", optimizer, "max score:", max(score_arr), "min score:", min(score_arr))

    cbar_ax = fig_learning_curve.add_axes([0.86, 0.13, 0.04, 0.82])
    fig_learning_curve.colorbar(contourf_set, cax=cbar_ax)

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.13, right=0.83, wspace=0.1, hspace=0.3)

    # Mark the sub figures with indicators to other insightful figures
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB': 'tab:orange', 'FOSI-GD': 'tab:red'}

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        ax = axs[idx // num_columns][idx % num_columns]
        linestyle = '-' if 'FOSI' in optimizer else '--'
        # Draw horizontal line in zeta_val_to_show_horizontal_cut
        ax.hlines(y=zeta_val_to_show_horizontal_cut, xmin=1.1, xmax=1.17, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
        # Draw vertical line in beta_val_to_show_vertical_cut
        ax.axvline(x=beta_val_to_show_vertical_cut, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
        # Draw 4 markers (x-markers) in zeta_vals_to_show_learning_curves, beta_vals_to_show_learning_curves
        ax.scatter(beta_vals_to_show_learning_curves, zeta_vals_to_show_learning_curves, marker='x', color=optimizers_colors[optimizer], s=10, linewidth=1)

        # For Adam plot a red dot in beta_val_adam_tuning, zeta_val_adam_tuning
        if optimizer == 'Adam':
            ax.plot(beta_val_adam_tuning, zeta_val_adam_tuning, marker='o', color=optimizers_colors[optimizer], markersize=3)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_per_optimizer(fig_file_name="quadratic_jax_kappa_zeta_.pdf", pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_scores = filter_results_to_range(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    contourf_set = None

    levels_per_optimizer = {'Adam': [2**i for i in range(-9, 3)],
                            'HB': [2**i for i in range(-12, 0)],
                            'GD': [2**i for i in range(-8, 4)]}

    for optimizer_name in ['Adam', 'HB', 'GD']:
        fig, axs = plt.subplots(1, num_columns, figsize=get_figsize(columnwidth=487.8225, wf=0.3, hf=0.4), sharex=True, sharey=True)

        optimizers_scores_specific = {}
        for idx, optimizer in enumerate([optimizer_name, 'FOSI-' + optimizer_name]):
            optimizers_scores_specific[optimizer] = optimizers_scores[optimizer]
        vmin, vmax = get_min_and_max(optimizers_scores_specific, b_ignore_gd=False)

        for idx, optimizer in enumerate([optimizer_name, 'FOSI-' + optimizer_name]):

            ax = axs[idx]

            res = optimizers_scores_specific[optimizer]
            res.sort(key=lambda val: (val[0], val[1]))
            res = list(zip(*res))

            dim_non_diag_arr = np.array(res[0])
            kappa_arr = np.array(res[1])
            score_arr = np.array(res[2])
            shape = (np.unique(dim_non_diag_arr).shape[0], np.unique(kappa_arr).shape[0])

            cs = ax.contourf(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape), score_arr.reshape(shape),
                             locator=ticker.LogLocator(base=2), vmin=vmin, vmax=vmax, extend='both',
                             levels=levels_per_optimizer[optimizer_name])
            contourf_set = cs

            ax.set_ylim(30, 100)
            ax.set_xlabel(r'$b$', labelpad=1.5)
            if idx == 0:
                ax.set_ylabel(r'$\zeta$', labelpad=0)

            ax.spines['right'].set_linewidth(0.5)
            ax.spines['top'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            print("Optimizer:", optimizer, "max score:", max(score_arr), "min score:", min(score_arr))

        cbar_ax = fig.add_axes([0.8, 0.13, 0.04, 0.82])
        fig.colorbar(contourf_set, cax=cbar_ax)

        fig.subplots_adjust(top=0.96, bottom=0.33, left=0.175, right=0.77, wspace=0.13, hspace=0.3)

        for idx, optimizer in enumerate([optimizer_name, 'FOSI-' + optimizer_name]):
            ax = axs[idx]
            # Draw 4 markers (x-markers) in zeta_vals_to_show_learning_curves, beta_vals_to_show_learning_curves
            ax.scatter(beta_vals_to_show_learning_curves, zeta_vals_to_show_learning_curves, marker='x', color='black', s=10, linewidth=1)

        plt.savefig(output_folder + fig_file_name.replace('.pdf', optimizer_name + '.pdf'))
        #plt.show()
        plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_curves(fig_file_name="quadratic_jax_kappa_zeta_learning_curves.pdf", pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB': 'tab:orange', 'FOSI-GD': 'tab:red'}

    # Plot learning curves
    num_columns = 2
    num_rows = len(zeta_vals_to_show_learning_curves) // num_columns
    fig_learning_curve, axs = plt.subplots(num_rows, num_columns, figsize=get_figsize(hf=0.55), sharex=True, sharey='row')

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: (val[0], val[1]))
        res = list(zip(*res))

        for j, (beta, zeta) in enumerate(zip(beta_vals_to_show_learning_curves, zeta_vals_to_show_learning_curves)):

            dim_non_diag_arr = np.array(res[0])
            kappa_arr = np.array(res[1])
            scores_arr = np.array(res[3])

            arg = np.where(np.logical_and(np.isclose(kappa_arr, beta), np.isclose(dim_non_diag_arr, zeta)))
            scores_arr = scores_arr[arg][0]

            ax = axs[j // num_columns][j % num_columns]
            linestyle = '-' if 'FOSI' in optimizer else '--'
            ax.plot(range(len(scores_arr)), scores_arr, label=optimizer, color=optimizers_colors[optimizer],
                    linewidth=0.8, linestyle=linestyle)

            ax.set_yscale('log')
            ax.set_xlim(0)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            if j // num_columns == (num_rows - 1):
                ax.set_xlabel('iteration')
            ax.set_ylabel(r'$f_{' + str(beta) + ',' + str(zeta) + r'}(\theta)$')

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig_learning_curve.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.14, right=0.99, wspace=0.23, hspace=0.13)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_zeta(pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB': 'tab:orange', 'FOSI-GD': 'tab:red'}

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_scores = filter_results_to_range(optimizers_scores)

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(wf=0.5, hf=0.9))

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val_to_show_horizontal_cut
        scores_for_zeta_val = [val for val in res if val[0] == zeta_val_to_show_horizontal_cut]
        scores_for_zeta_val.sort(key=lambda val: val[1])
        res = list(zip(*scores_for_zeta_val))

        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(kappa_arr, score_arr, label=optimizer, color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

    ax.set_yscale('log')
    ax.set_xlabel('$b$')
    ax.set_ylabel(r'$f_{\zeta=' + str(zeta_val_to_show_horizontal_cut) + r'}(\theta_{200})$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.85, bottom=0.23, left=0.15, right=0.99)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_constant_zeta.pdf")
    #plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_beta(pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB': 'tab:orange', 'FOSI-GD': 'tab:red'}

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_scores = filter_results_to_range(optimizers_scores)

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(hf=0.5))

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_beta_val = [val for val in res if np.isclose(val[1], beta_val_to_show_vertical_cut)]
        scores_for_beta_val.sort(key=lambda val: val[0])
        res = list(zip(*scores_for_beta_val))

        dim_non_diag_arr = np.array(res[0])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(dim_non_diag_arr, score_arr, label=optimizer, color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

    ax.set_yscale('log')
    ax.set_xlabel('$\zeta$')
    ax.set_ylabel(r'$f_{b=' + str(beta_val_to_show_vertical_cut) + r'}(\theta_{200})$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.85, bottom=0.23, left=0.15, right=0.99)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_constant_beta.pdf")
    #plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_zeta_beta(pkl_file="quadratic_jax_kappa_zeta.pkl"):
    set_rc_params()

    zeta_val = 80
    beta_val = 1.13
    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB': 'tab:orange', 'FOSI-GD': 'tab:red'}

    optimizers_scores = pickle.load(open(root_result_folder + pkl_file, 'rb'))
    optimizers_scores = filter_results_to_range(optimizers_scores)
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(hf=0.5))

    ax = axs[0]
    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_zeta_val = [val for val in res if val[0] == zeta_val]
        scores_for_zeta_val.sort(key=lambda val: val[1])
        res = list(zip(*scores_for_zeta_val))

        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(kappa_arr, score_arr, label=optimizer, color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

    ax.set_yscale('log')
    ax.set_xlabel('$b$')
    ax.set_ylabel(r'$f_{\zeta=' + str(zeta_val) + r'}(\theta_{200})$', labelpad=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    ax = axs[1]
    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_beta_val = [val for val in res if np.isclose(val[1], beta_val)]
        scores_for_beta_val.sort(key=lambda val: val[0])
        res = list(zip(*scores_for_beta_val))

        dim_non_diag_arr = np.array(res[0])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(dim_non_diag_arr, score_arr, label=optimizer, color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

    ax.set_yscale('log')
    ax.set_xlabel('$\zeta$')
    ax.set_ylabel(r'$f_{b=' + str(beta_val) + r'}(\theta_{200})$', labelpad=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.12, right=0.99, wspace=0.33)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_constant_zeta_beta.pdf")
    #plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_diagonals():
    set_rc_params()

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(wf=0.7, hf=0.5))
    for idx, b in enumerate([1.1, 1.15, 1.16, 1.17]):
        eigenvalues = [1e-3 * (b ** i) for i in range(100)]
        eigenvalues = eigenvalues[::-1]
        plt.scatter(range(1, len(eigenvalues)+1), eigenvalues, label='$b=$' + str(b), marker=str(idx+1), s=4, linewidths=0.4)

    ax.set_xlabel(r'eigenvalue index')
    ax.set_ylabel(r'eigenvalue value')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.set_xticks([1, 25, 50, 75, 100])

    ax.legend(frameon=False)

    plt.subplots_adjust(top=0.99, bottom=0.27, left=0.2, right=0.99)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_diagonals.pdf")
    #plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate(kappa=1.14, dim_non_diag=50):
    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB (c=1)': 'tab:pink', 'FOSI-GD (c=1)': 'brown',
                         'FOSI-HB (c=inf)': 'tab:orange', 'FOSI-GD (c=inf)': 'tab:red'}

    min_score = np.inf
    fig, ax = plt.subplots(1, 1)
    for idx, optimizer in enumerate(optimizers_scores.keys()):
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: val[0])
        res = list(zip(*res))

        eta_arr = np.array(res[0])
        scores_arr = np.array(res[1])
        if np.min(scores_arr) < min_score:
            min_score = np.min(scores_arr)

        scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(min_score/10, 400)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$f_{b=' + str(kappa) + r',\zeta=' + str(dim_non_diag) + r'}(\theta_{200})$')
    ax.set_title(r'$b=' + str(kappa) + r', \zeta=' + str(dim_non_diag) + '$')
    fig.legend()
    fig.savefig(output_folder + 'quadratic_jax_kappa_zeta_lr_' + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + '.pdf')
    #plt.show()
    plt.close(fig)


def plot_quadratic_jax_kappa_zeta_learning_rate_uni():
    set_rc_params()

    optimizers_colors = {'Adam': 'tab:blue', 'HB': 'tab:orange', 'GD': 'tab:red',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB (c=1)': 'tab:pink', 'FOSI-GD (c=1)': 'brown',
                         'FOSI-HB (c=inf)': 'tab:orange', 'FOSI-GD (c=inf)': 'tab:red'}

    fig, axs = plt.subplots(2, 2, figsize=get_figsize(hf=0.8), sharex=True, sharey='row')
    min_score = np.inf

    kappa_dim_non_diag_tuples = zip([90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16])
    for j, (dim_non_diag, kappa) in enumerate(kappa_dim_non_diag_tuples):
        lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
        optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

        ax = axs[j // 2][j % 2]

        for idx, optimizer in enumerate(optimizers_scores.keys()):
            res = optimizers_scores[optimizer]
            res.sort(key=lambda val: val[0])
            res = list(zip(*res))

            eta_arr = np.array(res[0])
            scores_arr = np.array(res[1])
            if np.min(scores_arr) < min_score:
                min_score = np.min(scores_arr)

            scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

            linestyle = '-' if 'FOSI' in optimizer else '--'
            linewidth = 0.7 if 'FOSI' in optimizer else 1.2
            ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 450)
            ax.set_ylabel(r'$f_{b=' + str(kappa) + r',\zeta=' + str(dim_non_diag) + r'}(\theta_{200})$')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

    axs[1][0].set_xlabel(r'$\eta$')
    axs[1][1].set_xlabel(r'$\eta$')

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.87, bottom=0.13, left=0.14, right=0.99, wspace=0.2)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_lr.pdf")
    #plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate_single_func():
    kappa_dim_non_diag_tuples = zip([90], [1.12])

    set_rc_params()

    optimizers_colors = {'Adam': 'black', 'HB': 'black', 'GD': 'black',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB (c=1)': 'tab:orange',
                         'FOSI-GD (c=1)': 'tab:red',
                         'FOSI-HB (c=inf)': 'tab:orange', 'FOSI-GD (c=inf)': 'tab:red'}

    fig, axs = plt.subplots(1, 3, figsize=get_figsize(hf=0.4), sharex=True, sharey='row')

    dim_non_diag, kappa = next(kappa_dim_non_diag_tuples)

    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

    for idx, optimizer_name in enumerate(['Adam', 'HB', 'GD']):
        ax = axs[idx]

        optimizers_related = [optimizer_name, 'FOSI-' + optimizer_name]
        if optimizer_name == 'HB':
            optimizers_related = [optimizer_name, 'FOSI-HB (c=1)', 'FOSI-HB (c=inf)']
        if optimizer_name == 'GD':
            optimizers_related = [optimizer_name, 'FOSI-GD (c=1)', 'FOSI-GD (c=inf)']

        for optimizer in optimizers_related:
            res = optimizers_scores[optimizer]
            res.sort(key=lambda val: val[0])
            res = list(zip(*res))

            eta_arr = np.array(res[0])
            scores_arr = np.array(res[1])

            scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

            linestyle = '-' if 'FOSI' in optimizer else '--'
            linestyle = '-.' if '(c=1)' in optimizer else linestyle
            linewidth = 0.7 if 'FOSI' in optimizer else 1.2
            ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 450)
        if idx == 0:
            ax.set_ylabel(r'$f_{' + str(kappa) + r',' + str(dim_non_diag) + r'}(\theta_{200})$')
        ax.set_xlabel(r'$\eta$', labelpad=0)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), frameon=False)

    plt.subplots_adjust(top=0.72, bottom=0.2, left=0.14, right=0.99, wspace=0.2)
    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_lr_single_func.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate_momentum_single_func():
    kappa_dim_non_diag_tuples = zip([90], [1.12])

    set_rc_params()

    optimizers_colors = {'Adam': 'black', 'HB': 'black',
                         'FOSI-Adam': 'tab:blue', 'FOSI-HB (c=1)': 'tab:orange',
                         'FOSI-HB (c=inf)': 'tab:orange'}

    fig, axs = plt.subplots(1, 2, figsize=get_figsize(wf=0.7, hf=0.4/0.7), sharex=True, sharey='row')

    dim_non_diag, kappa = next(kappa_dim_non_diag_tuples)

    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_momentum_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

    for idx, optimizer_name in enumerate(['Adam', 'HB']):
        ax = axs[idx]

        optimizers_related = [optimizer_name, 'FOSI-' + optimizer_name]
        if optimizer_name == 'HB':
            optimizers_related = [optimizer_name, 'FOSI-HB (c=1)', 'FOSI-HB (c=inf)']

        for optimizer in optimizers_related:
            res = optimizers_scores[optimizer]
            res.sort(key=lambda val: val[0])
            res = list(zip(*res))

            eta_arr = np.array(res[0])
            scores_arr = np.array(res[2])

            scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

            linestyle = '-' if 'FOSI' in optimizer else '--'
            linestyle = '-.' if '(c=1)' in optimizer else linestyle
            linewidth = 0.7 if 'FOSI' in optimizer else 1.2
            ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 450)
        if idx == 0:
            ax.set_ylabel(r'$f_{' + str(kappa) + r',' + str(dim_non_diag) + r'}(\theta_{200})$')
        ax.set_xlabel(r'$\eta$', labelpad=0)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), frameon=False)

    plt.subplots_adjust(top=0.72, bottom=0.2, left=0.2, right=0.99, wspace=0.2)
    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_lr_momentum_single_func.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate_momentum_single_func_adam(fig_type='pdf'):
    scaling_factor = 1.0 if 'pdf' in fig_type else 1.5
    kappa_dim_non_diag_tuples = zip([90], [1.12])

    set_rc_params(scaling_factor)

    optimizers_colors = {'Adam': 'black', 'FOSI-Adam': 'tab:blue'}

    fig, axs = plt.subplots(1, 3, figsize=get_figsize(wf=scaling_factor, hf=0.4), sharex=True, sharey='row')

    dim_non_diag, kappa = next(kappa_dim_non_diag_tuples)

    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))
    ax = axs[0]

    for optimizer in ['Adam', 'FOSI-Adam']:
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: val[0])
        res = list(zip(*res))

        eta_arr = np.array(res[0])
        scores_arr = np.array(res[1])

        scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

        linestyle = '-' if 'FOSI' in optimizer else '--'
        linewidth = 0.7*scaling_factor if 'FOSI' in optimizer else 1.2*scaling_factor
        ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)
        print(optimizer, "best LR:", eta_arr[np.argmin(scores_arr)])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 450)
    ax.set_ylabel(r'$f_{' + str(kappa) + r',' + str(dim_non_diag) + r'}(\theta_{200})$')
    ax.set_xlabel(r'$\eta$', labelpad=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    # Optimal beta1 for different lerning rates
    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_lr_momentum_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))
    ax = axs[1]

    for optimizer in ['Adam', 'FOSI-Adam']:
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: val[0])
        res = list(zip(*res))

        eta_arr = np.array(res[0])
        scores_arr = np.array(res[2])

        scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

        linestyle = '-' if 'FOSI' in optimizer else '--'
        linewidth = 0.7*scaling_factor if 'FOSI' in optimizer else 1.2*scaling_factor
        ax.plot(eta_arr, scores_arr, label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)
        print(optimizer, "best LR:", eta_arr[np.argmin(scores_arr)])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 450)
    ax.set_xlim(1e-4)
    ax.set_xlabel(r'$\eta$', labelpad=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    axs[0].annotate(r'$\beta_1=0.9$' + '\n' + r'$\beta_2=0.999$', xy=(0.37, 0.7), xycoords='axes fraction', xytext=(0, 0),
                      textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom',
                      fontsize=5.8 * scaling_factor)
    axs[1].annotate(r'$\beta_1=$' + r'tuned per $\eta$' + '\n' + r'$\beta_2=0.999$', xy=(0.37, 0.7), xycoords='axes fraction', xytext=(0, 0),
                    textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom',
                    fontsize=5.8 * scaling_factor)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.4, 1.0), ncol=2, columnspacing=1.5, handletextpad=0.29)
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.14, right=0.99, wspace=0.2)

    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_lr_momentum_single_func_adam." + fig_type)
    plt.close(fig)


    # different beta1 for beta2=0.99, 0.999, 0.9999
    fig, axs = plt.subplots(1, 3, figsize=get_figsize(wf=scaling_factor, hf=0.4), sharex=True, sharey='row')
    lr_pkl_file_name = root_result_folder + "quadratic_jax_kappa_zeta_momentum_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))
    beta2_vals = [0.99, 0.999, 0.9999]

    for optimizer in ['Adam', 'FOSI-Adam']:
        res = optimizers_scores[optimizer]
        res.sort(key=lambda val: val[0])
        res = list(zip(*res))

        beta2_arr = np.array(res[0])
        beta1_arr = np.array(res[1])
        scores_arr = np.array(res[2])
        scores_arr = np.nan_to_num(scores_arr, nan=np.inf)

        len = beta2_arr.shape[0] // 3

        linestyle = '-' if 'FOSI' in optimizer else '--'
        linewidth = 0.7*scaling_factor if 'FOSI' in optimizer else 1.2*scaling_factor

        for idx, beta2 in enumerate(beta2_vals):
            axs[idx].plot(beta1_arr[idx*len:(idx+1)*len], scores_arr[idx*len:(idx+1)*len], label=optimizer, color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)
            print(optimizer, "beta2:", beta2_arr[idx*len])

    axs[0].set_ylabel(r'$f_{' + str(kappa) + r',' + str(dim_non_diag) + r'}(\theta_{200})$')
    for idx, beta2 in enumerate(beta2_vals):
        axs[idx].annotate(r'$\beta_2=$' + str(beta2), xy=(0.8, 0.85), xycoords='axes fraction', xytext=(0, 0), textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontsize=5.8*scaling_factor)

    for ax in axs:
        ax.set_yscale('log')
        #ax.set_ylim(1e-3, 10)
        ax.set_xlabel(r'$\beta_1$', labelpad=0)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.5, handletextpad=0.29)

    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.14, right=0.99, wspace=0.2)
    plt.savefig(output_folder + "quadratic_jax_kappa_zeta_momentum_single_func_adam." + fig_type)
    plt.close(fig)

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_result_folder = sys.argv[1] + '/test_results/'
    else:
        root_result_folder = '../quadratic/test_results/'
    output_folder = './figures/'

    # Random orthogonal basis, different dimension and lambda_1
    #plot_quadratic_random_orthogonal_basis()
    plot_quadratic_random_orthogonal_basis_4_funcs("quadratic_random_orthogonal_basis_4_funcs.pdf")
    plot_quadratic_random_orthogonal_basis_4_funcs("quadratic_random_orthogonal_basis_4_funcs.png")
    plot_quadratic_random_orthogonal_basis_4_funcs_flat("quadratic_random_orthogonal_basis_4_funcs_flat.pdf")

    # GD and FOSI-GD, for effective condition number larger than the original one
    plot_quadratic_random_orthogonal_basis_gd()

    # Function with different kappa (condition number) and zeta (non diagonally dominant fraction)
    plot_quadratic_jax_kappa_zeta()
    plot_quadratic_jax_kappa_zeta_per_optimizer()
    plot_quadratic_jax_kappa_zeta_learning_curves()
    plot_quadratic_jax_kappa_zeta_constant_zeta()
    plot_quadratic_jax_kappa_zeta_constant_beta()
    plot_quadratic_jax_kappa_zeta_constant_zeta_beta()
    plot_diagonals()

    # Learning rate impact
    kappa_dim_non_diag_tuples = zip([90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16])
    for dim_non_diag, kappa in kappa_dim_non_diag_tuples:
        plot_quadratic_jax_kappa_zeta_learning_rate(kappa, dim_non_diag)
    plot_quadratic_jax_kappa_zeta_learning_rate_uni()
    plot_quadratic_jax_kappa_zeta_learning_rate_single_func()

    plot_quadratic_jax_kappa_zeta_learning_rate_momentum_single_func_adam('pdf')
    plot_quadratic_jax_kappa_zeta_learning_rate_momentum_single_func_adam('png')
    plot_quadratic_jax_kappa_zeta_learning_rate_momentum_single_func()
