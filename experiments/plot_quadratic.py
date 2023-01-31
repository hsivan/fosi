import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from matplotlib import ticker, cm


# ICML: textwidth=487.8225, columnwidth=234.8775
# ACM: textwidth=506.295, columnwidth=241.14749

#def get_figsize(columnwidth=241.14749, wf=1.0, hf=(5. ** 0.5 - 1.0) / 2.0, b_fixed_height=False):
def get_figsize(columnwidth=234.8775, wf=1.0, hf=(5. ** 0.5 - 1.0) / 2.0, b_fixed_height=False):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex (pt). Get this from LaTeX
                             using \showthe\columnwidth (or \the\columnwidth)
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    if b_fixed_height:
        fig_height = hf
    #print("fig_width", fig_width, "fig_height", fig_height)
    return [fig_width, fig_height]


def plot_quadratic_random_orthogonal_basis(fig_file_name="quadratic_random_orthogonal_basis.pdf", pkl_file="./optimizers_scores_quadratic.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()

    # Plot learning curves
    # textwidth is 506.295, columnwidth is 241.14749
    fig_learning_curve, ax_learning_curve = plt.subplots(len(n_dims), len(max_eigvals), figsize=get_figsize(columnwidth=487.8225, hf=0.4), sharex=True)

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve[n_dim_idx][max_eigval_idx]
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name.replace('Heavy-Ball', 'HB'), color=optimizer_color, linewidth=0.8, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI w/ ' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name.replace('Heavy-Ball', 'HB'), color=optimizer_color, linewidth=0.8, linestyle="-")

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


    plt.savefig('figures/' + fig_file_name)

    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_random_orthogonal_basis_4_funcs(fig_file_name="quadratic_random_orthogonal_basis_4_funcs.pdf", pkl_file="./optimizers_scores_quadratic.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red'}

    n_dims = list(set([x for x, _, _ in optimizers_scores.keys()]))
    n_dims.sort()
    max_eigvals = list(set([x for _, x, _ in optimizers_scores.keys()]))
    max_eigvals.sort()

    # Plot only extreme dimensions and extreme eigvals
    n_dims = [n_dims[0], n_dims[-1]]
    max_eigvals = [max_eigvals[0], max_eigvals[-1]]

    # Plot learning curves
    # textwidth is 506.295, columnwidth is 241.14749
    fig_learning_curve, ax_learning_curve = plt.subplots(len(n_dims), len(max_eigvals), figsize=get_figsize(hf=0.55), sharex=True, sharey=True)

    for n_dim_idx, n_dim in enumerate(n_dims):
        for max_eigval_idx, max_eigval in enumerate(max_eigvals):
            ax = ax_learning_curve[n_dim_idx][max_eigval_idx]
            for optimizer_name, optimizer_color in optimizers_colors.items():
                # base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label=optimizer_name.replace('Heavy-Ball', 'HB'), color=optimizer_color, linewidth=0.8, linestyle="--")
                # FOSI w/ base optimizer
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI w/ ' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name.replace('Heavy-Ball', 'HB'), color=optimizer_color, linewidth=0.8, linestyle="-")

            ax.set_yscale('log')

            if max_eigval_idx == 0:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 35), textcoords='offset pixels',
                            horizontalalignment='right',
                            verticalalignment='bottom')
            else:
                ax.annotate(r'$n$=' + str(n_dim) + r', $\lambda_1$=' + str(max_eigval),
                            xy=(1, 0), xycoords='axes fraction',
                            xytext=(-5, 30), textcoords='offset pixels',
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


    plt.savefig('figures/' + fig_file_name)

    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_random_orthogonal_basis_gd(fig_file_name="quadratic_random_orthogonal_basis_gd.pdf"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    optimizers_scores = pickle.load(open("./optimizers_scores_quadratic_gd.pkl", 'rb'))
    # GD color is blue and Heavy-Ball is orange, FOSI regular line and base dashed line
    #optimizers_colors = {'GD': 'tab:blue', 'Heavy-Ball': 'tab:orange'}
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
                optimizer_scores = optimizers_scores[(n_dim, max_eigval, 'FOSI w/ ' + optimizer_name)]
                ax.plot(range(len(optimizer_scores)), optimizer_scores, label='FOSI-' + optimizer_name, color=optimizer_color, linewidth=0.8, linestyle="-")
            ax.set_yticks([0, 5, 10, 15])
            #ax.set_ylim(0, 15)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

            ax.set_xlabel('iteration')
            ax.set_ylabel(r'$f(\theta)$')

    fig_learning_curve.legend(framealpha=0, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.05), ncol=1)
    plt.subplots_adjust(top=0.99, bottom=0.38, left=0.18, right=0.99, wspace=0.3, hspace=0.1)

    plt.savefig('figures/' + fig_file_name)

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

def plot_quadratic_jax_kappa_zeta(fig_file_name="quadratic_jax_kappa_zeta.pdf", pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    #optimizers_scores.pop('GD')
    #optimizers_scores.pop('FOSI w/ GD')
    optimizers_scores = filter_results_to_range(optimizers_scores)
    vmin, vmax = get_min_and_max(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    num_rows = num_optimizers // num_columns
    fig_learning_curve, axs = plt.subplots(num_rows, num_columns, figsize=get_figsize(hf=0.9), sharex=True, sharey=True)  # hf=0.9 with GD
    #contour_set = None
    #colormesh_set = None
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


        #cs = ax.contour(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape),
        #                score_arr.reshape(shape), vmin=vmin, vmax=vmax, levels=30)

        #cms = ax.pcolormesh(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape),
        #                    score_arr.reshape(shape), shading='gouraud', alpha=1.0, vmin=vmin, vmax=vmax, norm='log')

        cs = ax.contourf(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape), score_arr.reshape(shape),
                         locator=ticker.LogLocator(base=2), vmin=vmin, vmax=vmax,
                         levels=[2**i for i in range(-10, 3)], extend='both', cmap='binary')  # cmap=cm.PuBu_r
        cs.cmap.set_over('purple')
        cs.cmap.set_under('yellow')
        cs.changed()

        if optimizer == 'Heavy-Ball':
            contourf_set = cs

        #if contour_set is None or max(cs.levels) > max(contour_set.levels):
        #    contour_set = cs
        #    colormesh_set = cms

        ax.set_title(optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), pad=3)
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
    #plt.subplots_adjust(top=0.95, bottom=0.13, left=0.12, right=0.83, wspace=0.1, hspace=0.25)  # without GD


    # Mark the sub figures with indicators to other insightful figures
    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball': 'tab:orange', 'FOSI w/ GD': 'tab:red'}

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        ax = axs[idx // num_columns][idx % num_columns]
        linestyle = '-' if 'FOSI' in optimizer else '--'
        # Draw horizontal line in zeta_val_to_show_horizontal_cut
        ax.hlines(y=zeta_val_to_show_horizontal_cut, xmin=1.1, xmax=1.17, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
        # Draw vertical line in beta_val_to_show_vertical_cut
        ax.axvline(x=beta_val_to_show_vertical_cut, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
        # Draw 4 markers (x-markers) in zeta_vals_to_show_learning_curves, beta_vals_to_show_learning_curves
        #for zeta_val in zeta_vals_to_show_learning_curves:
        ax.scatter(beta_vals_to_show_learning_curves, zeta_vals_to_show_learning_curves, marker='x', color=optimizers_colors[optimizer], s=10, linewidth=1)

        # For Adam plot a red dot in beta_val_adam_tuning, zeta_val_adam_tuning
        if optimizer == 'Adam':
            ax.plot(beta_val_adam_tuning, zeta_val_adam_tuning, marker='o', color=optimizers_colors[optimizer], markersize=3)

    plt.savefig('figures/' + fig_file_name)

    #plt.show()
    plt.close(fig_learning_curve)

    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_per_optimizer(fig_file_name="quadratic_jax_kappa_zeta_.pdf", pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    #optimizers_scores.pop('GD')
    #optimizers_scores.pop('FOSI w/ GD')
    optimizers_scores = filter_results_to_range(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    num_rows = num_optimizers // num_columns
    #contour_set = None
    #colormesh_set = None
    contourf_set = None

    levels_per_optimizer = {'Adam': [2**i for i in range(-9, 3)],
                            'Heavy-Ball': [2**i for i in range(-12, 0)],
                            'GD': [2**i for i in range(-8, 4)]}

    for optimizer_name in ['Adam', 'Heavy-Ball', 'GD']:
        fig, axs = plt.subplots(1, num_columns, figsize=get_figsize(columnwidth=487.8225, wf=0.3, hf=0.4), sharex=True, sharey=True)

        optimizers_scores_specific = {}
        for idx, optimizer in enumerate([optimizer_name, 'FOSI w/ ' + optimizer_name]):
            optimizers_scores_specific[optimizer] = optimizers_scores[optimizer]
        vmin, vmax = get_min_and_max(optimizers_scores_specific, b_ignore_gd=False)

        for idx, optimizer in enumerate([optimizer_name, 'FOSI w/ ' + optimizer_name]):

            ax = axs[idx]

            res = optimizers_scores_specific[optimizer]
            res.sort(key=lambda val: (val[0], val[1]))
            res = list(zip(*res))

            dim_non_diag_arr = np.array(res[0])
            kappa_arr = np.array(res[1])
            score_arr = np.array(res[2])
            shape = (np.unique(dim_non_diag_arr).shape[0], np.unique(kappa_arr).shape[0])


            #cs = ax.contour(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape),
            #                score_arr.reshape(shape), vmin=vmin, vmax=vmax, levels=30)

            #cms = ax.pcolormesh(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape),
            #                    score_arr.reshape(shape), shading='gouraud', alpha=1.0, vmin=vmin, vmax=vmax, norm='log')

            '''cs = ax.contourf(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape), score_arr.reshape(shape),
                             locator=ticker.LogLocator(base=2), vmin=vmin, vmax=vmax,
                             levels=[2**i for i in range(-10, 3)], extend='both', cmap='binary')  # cmap=cm.PuBu_r'''
            cs = ax.contourf(kappa_arr.reshape(shape), dim_non_diag_arr.reshape(shape), score_arr.reshape(shape),
                             locator=ticker.LogLocator(base=2), vmin=vmin, vmax=vmax, extend='both',
                             levels=levels_per_optimizer[optimizer_name])
            #cs.cmap.set_over('black')
            #cs.cmap.set_under('white')
            #cs.changed()

            contourf_set = cs

            #if contour_set is None or max(cs.levels) > max(contour_set.levels):
            #    contour_set = cs
            #    colormesh_set = cms

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

        # Mark the sub figures with indicators to other insightful figures
        optimizers_colors = {'Adam': 'black', 'Heavy-Ball': 'black', 'GD': 'black',
                             'FOSI w/ Adam': 'black', 'FOSI w/ Heavy-Ball': 'black', 'FOSI w/ GD': 'black'}

        for idx, optimizer in enumerate([optimizer_name, 'FOSI w/ ' + optimizer_name]):
            ax = axs[idx]
            linestyle = '-' if 'FOSI' in optimizer else '--'
            # Draw horizontal line in zeta_val_to_show_horizontal_cut
            #ax.hlines(y=zeta_val_to_show_horizontal_cut, xmin=1.1, xmax=1.17, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
            # Draw vertical line in beta_val_to_show_vertical_cut
            #ax.axvline(x=beta_val_to_show_vertical_cut, linewidth=1, color=optimizers_colors[optimizer], linestyle=linestyle)
            # Draw 4 markers (x-markers) in zeta_vals_to_show_learning_curves, beta_vals_to_show_learning_curves
            ax.scatter(beta_vals_to_show_learning_curves, zeta_vals_to_show_learning_curves, marker='x', color=optimizers_colors[optimizer], s=10, linewidth=1)

            # For Adam plot a red dot in beta_val_adam_tuning, zeta_val_adam_tuning
            #if optimizer == 'Adam':
            #    ax.plot(beta_val_adam_tuning, zeta_val_adam_tuning, marker='o', color=optimizers_colors[optimizer], markersize=3)

        plt.savefig('figures/' + fig_file_name.replace('.pdf', optimizer_name + '.pdf'))

        #plt.show()
        plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_curves(fig_file_name="quadratic_jax_kappa_zeta_learning_curves.pdf", pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    #optimizers_scores.pop('GD')
    #optimizers_scores.pop('FOSI w/ GD')
    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball': 'tab:orange', 'FOSI w/ GD': 'tab:red'}

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
            ax.plot(range(len(scores_arr)), scores_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer],
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


    plt.savefig('figures/' + fig_file_name)

    #plt.show()
    plt.close(fig_learning_curve)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_zeta(pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball': 'tab:orange', 'FOSI w/ GD': 'tab:red'}


    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    # optimizers_scores.pop('GD')
    # optimizers_scores.pop('FOSI w/ GD')
    optimizers_scores = filter_results_to_range(optimizers_scores)
    vmin, vmax = get_min_and_max(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    num_rows = num_optimizers // num_columns
    #fig_learning_curve, axs = plt.subplots(num_rows, num_columns, figsize=get_figsize(hf=0.9), sharex=True, sharey=True)

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(wf=0.5, hf=0.9))

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        #if 'GD' in optimizer:
        #    continue
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val_to_show_horizontal_cut
        scores_for_zeta_val = [val for val in res if val[0] == zeta_val_to_show_horizontal_cut]
        scores_for_zeta_val.sort(key=lambda val: val[1])
        res = list(zip(*scores_for_zeta_val))

        dim_non_diag_arr = np.array(res[0])
        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])
        shape = (np.unique(dim_non_diag_arr).shape[0], np.unique(kappa_arr).shape[0])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(kappa_arr, score_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

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
    #plt.show()
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_constant_zeta.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_beta(pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball': 'tab:orange', 'FOSI w/ GD': 'tab:red'}


    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    # optimizers_scores.pop('GD')
    # optimizers_scores.pop('FOSI w/ GD')
    optimizers_scores = filter_results_to_range(optimizers_scores)
    vmin, vmax = get_min_and_max(optimizers_scores)

    num_optimizers = len(optimizers_scores.keys())
    num_columns = 2
    num_rows = num_optimizers // num_columns
    #fig_learning_curve, axs = plt.subplots(num_rows, num_columns, figsize=get_figsize(hf=0.9), sharex=True, sharey=True)

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(hf=0.5))

    for idx, optimizer in enumerate(optimizers_scores.keys()):
        #if 'GD' in optimizer:
        #    continue
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_beta_val = [val for val in res if np.isclose(val[1], beta_val_to_show_vertical_cut)]
        scores_for_beta_val.sort(key=lambda val: val[0])
        res = list(zip(*scores_for_beta_val))

        dim_non_diag_arr = np.array(res[0])
        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])
        shape = (np.unique(dim_non_diag_arr).shape[0], np.unique(kappa_arr).shape[0])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(dim_non_diag_arr, score_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

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
    #plt.show()
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_constant_beta.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_constant_zeta_beta(pkl_file="./quadratic_jax_kappa_zeta.pkl"):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    zeta_val = 80
    beta_val = 1.13
    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball': 'tab:orange', 'FOSI w/ GD': 'tab:red'}


    optimizers_scores = pickle.load(open(pkl_file, 'rb'))
    # Remove GD
    # optimizers_scores.pop('GD')
    # optimizers_scores.pop('FOSI w/ GD')
    optimizers_scores = filter_results_to_range(optimizers_scores)
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(hf=0.5))

    ax = axs[0]
    for idx, optimizer in enumerate(optimizers_scores.keys()):
        #if 'GD' in optimizer:
        #    continue
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_zeta_val = [val for val in res if val[0] == zeta_val]
        scores_for_zeta_val.sort(key=lambda val: val[1])
        res = list(zip(*scores_for_zeta_val))

        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(kappa_arr, score_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

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
        #if 'GD' in optimizer:
        #    continue
        res = optimizers_scores[optimizer]

        # Collect scores for zeta with zeta_val
        scores_for_beta_val = [val for val in res if np.isclose(val[1], beta_val)]
        scores_for_beta_val.sort(key=lambda val: val[0])
        res = list(zip(*scores_for_beta_val))

        dim_non_diag_arr = np.array(res[0])
        kappa_arr = np.array(res[1])
        score_arr = np.array(res[2])

        linestyle = '-' if 'FOSI' in optimizer else '--'
        ax.plot(dim_non_diag_arr, score_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linewidth=0.8, linestyle=linestyle)

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
    #plt.show()
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_constant_zeta_beta.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_diagonals():
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

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
    #ax.set_xlim(0, 101)
    ax.tick_params(width=0.5)
    ax.set_xticks([1, 25, 50, 75, 100])

    ax.legend(frameon=False)

    plt.subplots_adjust(top=0.99, bottom=0.27, left=0.2, right=0.99)
    #plt.show()
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_diagonals.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate(kappa=1.14, dim_non_diag=50):
    lr_pkl_file_name = "./quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball (c=1)': 'tab:pink', 'FOSI w/ GD (c=1)': 'brown',
                         'FOSI w/ Heavy-Ball (c=inf)': 'tab:orange', 'FOSI w/ GD (c=inf)': 'tab:red'}

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
        ax.plot(eta_arr, scores_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linestyle=linestyle)

    #ax.set_ylim(-5, 250)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(min_score/10, 400)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$f_{b=' + str(kappa) + r',\zeta=' + str(dim_non_diag) + r'}(\theta_{200})$')
    ax.set_title(r'$b=' + str(kappa) + r', \zeta=' + str(dim_non_diag) + '$')
    fig.legend()
    #plt.show()
    fig.savefig('figures/' + 'quadratic_jax_kappa_zeta_lr_' + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + '.pdf')


def plot_quadratic_jax_kappa_zeta_learning_rate_uni():
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_colors = {'Adam': 'tab:blue', 'Heavy-Ball': 'tab:orange', 'GD': 'tab:red',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball (c=1)': 'tab:pink', 'FOSI w/ GD (c=1)': 'brown',
                         'FOSI w/ Heavy-Ball (c=inf)': 'tab:orange', 'FOSI w/ GD (c=inf)': 'tab:red'}

    fig, axs = plt.subplots(2, 2, figsize=get_figsize(hf=0.8), sharex=True, sharey='row')
    min_score = np.inf

    kappa_dim_non_diag_tuples = zip([90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16])
    #kappa_dim_non_diag_tuples = zip([80, 80, 50, 50], [1.12, 1.14, 1.12, 1.14])
    for j, (dim_non_diag, kappa) in enumerate(kappa_dim_non_diag_tuples):
        lr_pkl_file_name = "./quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
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
            ax.plot(eta_arr, scores_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)

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

    #plt.show()
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_lr.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_quadratic_jax_kappa_zeta_learning_rate_single_func():
    kappa_dim_non_diag_tuples = zip([90], [1.12])  # kappa_dim_non_diag_tuples = zip([90], [1.16])

    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8

    optimizers_colors = {'Adam': 'black', 'Heavy-Ball': 'black', 'GD': 'black',
                         'FOSI w/ Adam': 'tab:blue', 'FOSI w/ Heavy-Ball (c=1)': 'tab:orange',
                         'FOSI w/ GD (c=1)': 'tab:red',
                         'FOSI w/ Heavy-Ball (c=inf)': 'tab:orange', 'FOSI w/ GD (c=inf)': 'tab:red'}

    fig, axs = plt.subplots(1, 3, figsize=get_figsize(hf=0.4), sharex=True, sharey='row')

    dim_non_diag, kappa = next(kappa_dim_non_diag_tuples)

    lr_pkl_file_name = "./quadratic_jax_kappa_zeta_lr_" + str(kappa).replace('.', '-') + '_' + str(dim_non_diag) + ".pkl"
    optimizers_scores = pickle.load(open(lr_pkl_file_name, 'rb'))

    for idx, optimizer_name in enumerate(['Adam', 'Heavy-Ball', 'GD']):
        ax = axs[idx]

        optimizers_related = [optimizer_name, 'FOSI w/ ' + optimizer_name]
        if optimizer_name == 'Heavy-Ball':
            optimizers_related = [optimizer_name, 'FOSI w/ Heavy-Ball (c=1)', 'FOSI w/ Heavy-Ball (c=inf)']
        if optimizer_name == 'GD':
            optimizers_related = [optimizer_name, 'FOSI w/ GD (c=1)', 'FOSI w/ GD (c=inf)']

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
            ax.plot(eta_arr, scores_arr, label=optimizer.replace('Heavy-Ball', 'HB').replace(' w/ ', '-'), color=optimizers_colors[optimizer], linestyle=linestyle, linewidth=linewidth)

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
    plt.savefig('figures/' + "quadratic_jax_kappa_zeta_lr_single_func.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    plot_quadratic_random_orthogonal_basis()
    plot_quadratic_random_orthogonal_basis_4_funcs()
    plot_quadratic_random_orthogonal_basis_gd()

    plot_quadratic_jax_kappa_zeta()
    plot_quadratic_jax_kappa_zeta_per_optimizer()
    plot_quadratic_jax_kappa_zeta_learning_curves()
    plot_quadratic_jax_kappa_zeta_constant_zeta()
    plot_quadratic_jax_kappa_zeta_constant_beta()
    plot_quadratic_jax_kappa_zeta_constant_zeta_beta()
    plot_diagonals()

    kappa_dim_non_diag_tuples = zip([90, 90, 50, 50], [1.12, 1.16, 1.12, 1.16])
    for dim_non_diag, kappa in kappa_dim_non_diag_tuples:
        plot_quadratic_jax_kappa_zeta_learning_rate(kappa, dim_non_diag)
    plot_quadratic_jax_kappa_zeta_learning_rate_uni()
    plot_quadratic_jax_kappa_zeta_learning_rate_single_func()
