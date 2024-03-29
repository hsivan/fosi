import os
import sys

import numpy as np
import pandas as pd
from matplotlib import rcParams, pyplot as plt, rcParamsDefault

from experiments.visualization.visualization_utils import get_figsize, set_rc_params

mapping = {'adam': ('tab:blue', '--', 'Adam'),
           'fosi_adam': ('tab:blue', '-', 'FOSI-Adam'),
           'momentum': ('tab:orange', '--', 'HB'),
           'fosi_momentum': ('tab:orange', '-', 'FOSI-HB'),
           'kfac': ('tab:red', '-', 'K-FAC'),
           'lbfgs': ('tab:green', '-', 'L-BFGS')
           }


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def get_best_config_test_folder(optimizer_name, test_result_root_folder, learning_rates, momentums):
    # Find the configuration that obtained the smallest loss value and return the corresponding result folder
    min_loss = np.inf
    best_result_folder = None
    best_learning_rate, best_momentum = None, None

    for learning_rate in learning_rates:
        for momentum in momentums:
            # Find the last result folder with learning_rate and momentum
            test_folders = [test_result_root_folder + f for f in os.listdir(test_result_root_folder) if
                            f.startswith('results_'+optimizer_name+'_')]
            test_folders.sort()
            for test_folder in test_folders[::-1]:
                conf = read_config_file(test_folder)
                if conf["learning_rate"] == learning_rate and conf["momentum"] == momentum:
                    # Read train_stats.csv file and get the best loss (smallest)
                    df = pd.read_csv(test_folder + '/train_stats.csv')
                    if np.min(df['val_loss'][df['val_loss'] != 0]) < min_loss:
                        min_loss = np.min(df['val_loss'][df['val_loss'] != 0])
                        best_result_folder = test_folder
                        best_learning_rate, best_momentum = learning_rate, momentum
                    break

    print("best_result_folder:", best_result_folder, "min_loss:", min_loss, "best_learning_rate:", best_learning_rate, "best_momentum:", best_momentum)
    return best_result_folder


def get_test_folders(test_result_root_folder, b_second_order_algos=False):
    # Return 4 test folders - last run for each optimizer, Adam (adam), HB (momentum), FOSI-Adam (fosi_adam),
    # and FOSI-HB (fosi_momentum).
    # If b_second_order_algos=True then return FOSI-HB, K-FAC, and L-BFGS.

    optimizers = ['adam', 'momentum', 'fosi_adam', 'fosi_momentum'] if not b_second_order_algos else ['fosi_momentum']

    test_folders = []

    for optimizer in optimizers:
        optimizer_test_folders = [test_result_root_folder + f for f in os.listdir(test_result_root_folder) if f.startswith('results_' + optimizer + '_')]
        optimizer_test_folders.sort()
        last_folder = optimizer_test_folders[-1]
        test_folders.append(last_folder)

    if b_second_order_algos:
        # Add K-FAC
        last_folder = get_best_config_test_folder('kfac', test_result_root_folder, learning_rates=[None, 1e-3, 1e-2, 1e-1], momentums=[None, 0.1, 0.5, 0.9])
        if last_folder is not None:
            test_folders.append(last_folder)

        # Add L-BFGS
        last_folder = get_best_config_test_folder('lbfgs', test_result_root_folder, learning_rates=[0], momentums=[10, 20, 40, 80, 100])  # momentum is actually history_size
        if last_folder is not None:
            test_folders.append(last_folder)

    return test_folders


def get_optimizer(test_folder):
    # Return the optimizer name (adam, momentum, fosi_adam, or fosi_momentum)
    conf = read_config_file(test_folder)
    return conf["optimizer"]


def plot_train_loss_over_iterations_and_wall_time(test_result_root_folder, fig_file_name, y_top_lim, max_data_point=None, x_label='epoch'):
    set_rc_params()

    test_folders = get_test_folders(test_result_root_folder)

    for learning_curve_type in ['train', 'val']:

        fig, axes = plt.subplots(1, 2, sharey=True, figsize=get_figsize(hf=0.5))

        min_val = np.inf
        tenth_min_val = np.inf

        for test_folder in test_folders:
            ax = axes[0]
            optimizer = get_optimizer(test_folder)
            df = pd.read_csv(test_folder + '/train_stats.csv')
            max_data_point = len(df) if max_data_point is None else max_data_point
            ax.plot(df['epoch'][df[learning_curve_type + '_loss'] != 0][:max_data_point], df[learning_curve_type + '_loss'][df[learning_curve_type + '_loss'] != 0][:max_data_point],
                    label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.8)

            ax = axes[1]
            ax.plot(df['wall_time'][df[learning_curve_type + '_loss'] != 0][:max_data_point], df[learning_curve_type + '_loss'][df[learning_curve_type + '_loss'] != 0][:max_data_point],
                    label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.8)
            sorted_vals = np.sort(df[learning_curve_type + '_loss'][df[learning_curve_type + '_loss'] != 0][:max_data_point])
            if sorted_vals[0] < min_val:
                min_val = sorted_vals[0]
                tenth_min_val = sorted_vals[10]

        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(learning_curve_type.replace('val', 'validation') + ' loss')
        axes[1].set_ylim(min_val - (tenth_min_val - min_val), y_top_lim)
        axes[1].set_xlabel('wall time (seconds)')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(test_folders), columnspacing=1.0, handletextpad=0.29, handlelength=1.0)
        plt.subplots_adjust(top=0.9, bottom=0.20, left=0.12, right=0.99, wspace=0.1)

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

        if learning_curve_type == 'train':
            plt.savefig(output_folder + fig_file_name)
        else:
            plt.savefig(output_folder + fig_file_name.replace('.pdf', '_validation.pdf'))
        #plt.show()
        plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_loss_and_accuracy_for_mobilenet(test_result_root_folder, fig_file_name, y_top_lim, y_bottom_lim=None, max_data_point=None, x_label='epoch', skip_val=1):
    scaling_factor = 1 if '.pdf' in fig_file_name else 1.5
    set_rc_params(scaling_factor)

    test_folders = get_test_folders(test_result_root_folder)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(wf=scaling_factor, hf=0.52), sharex='col')

    min_val = np.inf
    tenth_min_val = np.inf

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)
        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        sorted_vals = np.sort(df['train_loss'][df['train_loss'] != 0][:max_data_point])
        if sorted_vals[0] < min_val:
            min_val = sorted_vals[0]
            tenth_min_val = sorted_vals[10]

        ax = axes[0][0]
        ax.plot(df['epoch'][:max_data_point][df['train_loss'] != 0],
                df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0],
                linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[0][1]
        ax.plot(df['wall_time'][:max_data_point][df['train_loss'] != 0],
                df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0],
                linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[1][0]
        ax.plot(df['epoch'][:max_data_point][df['val_loss'] != 0][::skip_val],
                df['val_loss'][:max_data_point][df['val_loss'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0],
                linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[1][1]
        ax.plot(df['wall_time'][:max_data_point][df['val_acc'] != 0][::skip_val],
                df['val_acc'][:max_data_point][df['val_acc'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0],
                linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

    axes[0][0].set_ylabel('train loss', labelpad=6)
    axes[0][0].set_ylim(y_bottom_lim, y_top_lim)
    axes[0][0].set_yticks([0.02, 0.03])

    y_bottom_lim = min_val - (tenth_min_val - min_val) if y_bottom_lim is None else y_bottom_lim
    axes[0][1].set_ylim(y_bottom_lim, y_top_lim)
    axes[0][1].set_ylabel('train loss', labelpad=3)

    axes[1][0].set_xlabel(x_label)
    axes[1][0].set_ylabel('validation loss', labelpad=3)
    axes[1][0].set_ylim(0.023, y_top_lim)
    axes[1][0].set_yticks([0.025, 0.03])

    axes[1][1].set_xlabel('wall time (seconds)')
    axes[1][1].set_ylabel('validation accuracy', labelpad=6)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(test_folders),
               columnspacing=1.0, handletextpad=0.29, handlelength=1.0)
    plt.subplots_adjust(top=0.9, bottom=0.19, left=0.14, right=0.99, wspace=0.5, hspace=0.45)

    for ax in [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    plt.savefig(output_folder + fig_file_name)
    # plt.show()
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_train_loss_over_iterations_and_wall_time_and_validation_loss(test_result_root_folder, fig_file_name, y_top_lim, y_bottom_lim=None, max_data_point=None, x_label='epoch', skip_val=1, b_second_order_algos=False):
    scaling_factor = 1 if '.pdf' in fig_file_name else 1.5
    set_rc_params(scaling_factor)

    test_folders = get_test_folders(test_result_root_folder, b_second_order_algos)

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=get_figsize(wf=scaling_factor, hf=0.4))

    min_val = np.inf
    tenth_min_val = np.inf
    max_y_wall_time = -np.inf
    kfac_wall_times = None

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)

        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        sorted_vals = np.sort(df['train_loss'][df['train_loss'] != 0][:max_data_point])
        if sorted_vals[0] < min_val:
            min_val = sorted_vals[0]
            tenth_min_val = sorted_vals[10]

        ax = axes[0]
        ax.plot(df['epoch'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[1]
        ax.plot(df['wall_time'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)
        #if optimizer != 'lbfgs' and df['wall_time'][:max_data_point][df['train_loss'] != 0].iloc[-1] > max_y_wall_time:
        #    max_y_wall_time = df['wall_time'][:max_data_point][df['train_loss'] != 0].iloc[-1]
        #if optimizer == 'kfac':
        #    kfac_wall_times = df['wall_time'][:max_data_point][df['train_loss'] != 0]

        ax = axes[2]
        ax.plot(df['epoch'][:max_data_point][df['val_loss'] != 0][::skip_val], df['val_loss'][:max_data_point][df['val_loss'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('train loss')
    y_bottom_lim = min_val - (tenth_min_val - min_val) if y_bottom_lim is None else y_bottom_lim
    axes[1].set_ylim(y_bottom_lim, y_top_lim)
    # Cut the x-axis at the first wall time point of K-FAC that is larger than the last one of FOSI's
    #if kfac_wall_times is not None:
    #    for wall_time in kfac_wall_times:
    #        if wall_time > max_y_wall_time:
    #            max_y_wall_time = wall_time
    #            break
    #axes[1].set_xlim(right=max_y_wall_time)  # Cut the x-axis at the end of FOSI's graph, not L-BFGS

    axes[1].set_xlabel('wall time (seconds)')
    axes[1].set_ylabel('train loss')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel('validation loss')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(test_folders), columnspacing=1.0, handletextpad=0.29, handlelength=1.0)
    plt.subplots_adjust(top=0.89, bottom=0.24, left=0.15, right=0.98, wspace=0.23)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    fig_file_name = fig_file_name if not b_second_order_algos else fig_file_name.replace('.pdf', '_sec_ord.pdf')
    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_train_valid_loss_over_wall_sec_ord(test_result_root_folder, fig_file_name, y_top_lim, y_bottom_lim=None, max_data_point=None, skip_val=1, b_second_order_algos=False):
    scaling_factor = 1 if '.pdf' in fig_file_name else 1.5
    set_rc_params(scaling_factor)

    test_folders = get_test_folders(test_result_root_folder, b_second_order_algos)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=get_figsize(wf=scaling_factor, hf=0.35))

    min_val = np.inf
    tenth_min_val = np.inf
    max_y_wall_time = -np.inf
    kfac_wall_times = None

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)

        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        sorted_vals = np.sort(df['train_loss'][df['train_loss'] != 0][:max_data_point])
        if sorted_vals[0] < min_val:
            min_val = sorted_vals[0]
            tenth_min_val = sorted_vals[10]

        ax = axes[0]
        ax.plot(df['wall_time'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)
        '''if optimizer != 'lbfgs' and df['wall_time'][:max_data_point][df['train_loss'] != 0].iloc[-1] > max_y_wall_time:
            max_y_wall_time = df['wall_time'][:max_data_point][df['train_loss'] != 0].iloc[-1]
        if optimizer == 'kfac':
            kfac_wall_times = df['wall_time'][:max_data_point][df['train_loss'] != 0]'''

        ax = axes[1]
        ax.plot(df['wall_time'][:max_data_point][df['val_loss'] != 0][::skip_val], df['val_loss'][:max_data_point][df['val_loss'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

    axes[0].set_xlabel('wall time (seconds)')
    axes[0].set_ylabel('train loss')
    y_bottom_lim = min_val - (tenth_min_val - min_val) if y_bottom_lim is None else y_bottom_lim
    axes[1].set_ylim(y_bottom_lim, y_top_lim)
    # Cut the x-axis at the first wall time point of K-FAC that is larger than the last one of FOSI's
    '''if kfac_wall_times is not None:
        for wall_time in kfac_wall_times:
            if wall_time > max_y_wall_time:
                max_y_wall_time = wall_time
                break
    axes[1].set_xlim(right=max_y_wall_time)  # Cut the x-axis at the end of FOSI's graph, not L-BFGS
    '''
    axes[1].set_xlabel('wall time (seconds)')
    axes[1].set_ylabel('validation loss')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(test_folders), columnspacing=1.0, handletextpad=0.29, handlelength=1.0)
    plt.subplots_adjust(top=0.87, bottom=0.27, left=0.15, right=0.98, wspace=0.23)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    fig_file_name = fig_file_name if not b_second_order_algos else fig_file_name.replace('.pdf', '_sec_ord.pdf')
    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_train_loss_over_iterations_and_wall_time_and_validation_accuracy(test_result_root_folder, fig_file_name, y_top_lim, y_bottom_lim=None, max_data_point=None, x_label='epoch', skip_val=1):
    scaling_factor = 1 if '.pdf' in fig_file_name else 1.5
    suffix = '.pdf' if '.pdf' in fig_file_name else '.png'
    set_rc_params(scaling_factor)

    test_folders = get_test_folders(test_result_root_folder)

    fig, axes = plt.subplots(1, 2, figsize=get_figsize(columnwidth=397.48499*0.39, wf=1.0*scaling_factor, hf=0.3), sharey=True)

    min_val = np.inf
    tenth_min_val = np.inf
    kfac_wall_times = None

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)

        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        sorted_vals = np.sort(df['train_loss'][df['train_loss'] != 0][:max_data_point])
        if sorted_vals[0] < min_val:
            min_val = sorted_vals[0]
            tenth_min_val = sorted_vals[10]

        ax = axes[0]
        ax.plot(df['epoch'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[1]
        ax.plot(df['wall_time'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        if optimizer == 'kfac':
            kfac_wall_times = df['wall_time'][:max_data_point][df['train_loss'] != 0]

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('train loss')
    y_bottom_lim = min_val - (tenth_min_val - min_val) if y_bottom_lim is None else y_bottom_lim
    axes[1].set_ylim(y_bottom_lim, y_top_lim)
    axes[1].set_xlabel('wall time (sec.)')
    axes[0].set_yticks([0.01, 0.02, 0.03])
    if kfac_wall_times is not None:
        axes[1].set_xlim(right=np.max(kfac_wall_times))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(test_folders), columnspacing=0.7, handletextpad=0.29, handlelength=1.0)
    plt.subplots_adjust(top=0.86, bottom=0.28, left=0.21, right=0.99, wspace=0.2)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    plt.savefig(output_folder + fig_file_name)
    #plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=get_figsize(columnwidth=397.48499*0.39, wf=0.5*scaling_factor, hf=0.3/0.5))

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)

        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        ax.plot(df['epoch'][:max_data_point][df['val_acc'] != 0][::skip_val],
                df['val_acc'][:max_data_point][df['val_acc'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0],
                linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('validation acc.')
    plt.subplots_adjust(top=0.86, bottom=0.28, left=0.32, right=0.99, wspace=0.2)
    plt.savefig(output_folder + 'audioset_fosi_vs_base_val_acc' + suffix)
    #plt.show()
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def plot_train_loss_and_validation_accuracy_over_wall_time(test_result_root_folder, fig_file_name, y_top_lim, y_bottom_lim=None, max_data_point=None, skip_val=1):
    scaling_factor = 1 if '.pdf' in fig_file_name else 1.5
    set_rc_params(scaling_factor)

    test_folders = get_test_folders(test_result_root_folder)

    fig, axes = plt.subplots(1, 2, figsize=get_figsize(columnwidth=147.06749, wf=1.0*scaling_factor, hf=0.42))

    min_val = np.inf
    tenth_min_val = np.inf

    for test_folder in test_folders:
        optimizer = get_optimizer(test_folder)

        df = pd.read_csv(test_folder + '/train_stats.csv')
        max_data_point = len(df) if max_data_point is None else max_data_point

        sorted_vals = np.sort(df['train_loss'][df['train_loss'] != 0][:max_data_point])
        if sorted_vals[0] < min_val:
            min_val = sorted_vals[0]
            tenth_min_val = sorted_vals[10]

        ax = axes[0]
        ax.plot(df['wall_time'][:max_data_point][df['train_loss'] != 0], df['train_loss'][:max_data_point][df['train_loss'] != 0],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

        ax = axes[1]
        ax.plot(df['wall_time'][:max_data_point][df['val_acc'] != 0][::skip_val], df['val_acc'][:max_data_point][df['val_acc'] != 0][::skip_val],
                label=mapping[optimizer][2], color=mapping[optimizer][0], linestyle=mapping[optimizer][1], linewidth=0.7*scaling_factor)

    axes[0].set_ylabel('train loss')
    axes[1].set_ylabel('valid acc.')
    y_bottom_lim = min_val - (tenth_min_val - min_val) if y_bottom_lim is None else y_bottom_lim
    axes[0].set_ylim(y_bottom_lim, y_top_lim)
    axes[0].set_xlabel('wall time (sec.)')
    axes[1].set_xlabel('wall time (sec.)')
    #axes[0].set_yticks([0.01, 0.02, 0.03])

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(width=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, framealpha=0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=len(test_folders), columnspacing=0.7, handletextpad=0.29, handlelength=1.0)
    plt.subplots_adjust(top=0.83, bottom=0.35, left=0.2, right=0.99, wspace=0.65)
    plt.savefig(output_folder + fig_file_name)
    plt.close(fig)

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_result_folder = sys.argv[1] + '/'
    else:
        root_result_folder = '../dnn/'
    output_folder = './figures/'

    '''
    plot_train_loss_over_iterations_and_wall_time(root_result_folder + 'test_results_logistic_regression_mnist/', 'mnist_fosi_vs_base.pdf', 0.4)

    plot_train_loss_over_iterations_and_wall_time(root_result_folder + 'test_results_autoencoder_cifar10/', "autoencoder_cifar10_fosi_vs_base_128.pdf", 60)

    plot_train_loss_over_iterations_and_wall_time(root_result_folder + 'test_results_transfer_learning_cifar10/', 'transfer_learning_fosi_vs_base.pdf', 0.8)

    plot_train_loss_over_iterations_and_wall_time(root_result_folder + 'test_results_rnn_shakespeare/', 'rnn_shakespeare_fosi_vs_base.pdf', 2.4, x_label='iteration')
    '''

    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_logistic_regression_mnist/', 'mnist_fosi_vs_base_train_val.png', 0.3)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_logistic_regression_mnist/', 'mnist_fosi_vs_base_train_val.pdf', 0.3)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_logistic_regression_mnist/', 'mnist_fosi_vs_base_train_val.pdf', 0.3, b_second_order_algos=True)

    # L-BFGS diverges for any history_size at the first epoch, therefore not included
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_autoencoder_cifar10/', "autoencoder_cifar10_fosi_vs_base_train_val_128.png", y_top_lim=60)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_autoencoder_cifar10/', "autoencoder_cifar10_fosi_vs_base_train_val_128.pdf", y_top_lim=60)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_autoencoder_cifar10/', "autoencoder_cifar10_fosi_vs_base_train_val_128.pdf", y_top_lim=60, b_second_order_algos=True)

    # L-BFGS not included since its performance is bad (its loss is an order of magnitude larger and it runs 8x times slower) and distorts the figure
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_transfer_learning_cifar10/', 'transfer_learning_fosi_vs_base_train_val.png', 0.8)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_transfer_learning_cifar10/', 'transfer_learning_fosi_vs_base_train_val.pdf', 0.8)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_transfer_learning_cifar10/', 'transfer_learning_fosi_vs_base_train_val.pdf', 0.8, b_second_order_algos=True)

    # K-FAC throws exception for RNN.
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_rnn_shakespeare/', 'rnn_shakespeare_fosi_vs_base_train_val.png', 2.7,  x_label='iteration')
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_rnn_shakespeare/', 'rnn_shakespeare_fosi_vs_base_train_val.pdf', 2.7, x_label='iteration')
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_rnn_shakespeare/', 'rnn_shakespeare_fosi_vs_base_train_val.pdf', 2.7, x_label='iteration', b_second_order_algos=True)

    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val.png', 0.035, y_bottom_lim=0.013, skip_val=5)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val.pdf', 0.035, y_bottom_lim=0.013, skip_val=5)
    plot_train_loss_over_iterations_and_wall_time_and_validation_loss(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val.pdf', 0.035, y_bottom_lim=0.013, skip_val=5, b_second_order_algos=True)

    plot_loss_and_accuracy_for_mobilenet(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val_acc.pdf', 0.03, y_bottom_lim=0.013, skip_val=5)
    plot_loss_and_accuracy_for_mobilenet(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val_acc.png', 0.03, y_bottom_lim=0.013, skip_val=5)
    plot_train_loss_over_iterations_and_wall_time_and_validation_accuracy(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val_losses.pdf', 0.025, y_bottom_lim=0.013, skip_val=5)
    plot_train_loss_over_iterations_and_wall_time_and_validation_accuracy(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val_losses.png', 0.025, y_bottom_lim=0.013, skip_val=5)
    plot_train_loss_and_validation_accuracy_over_wall_time(root_result_folder + 'test_results_mobilenet_audioset/', 'audioset_fosi_vs_base_train_val_over_time.pdf', 0.025, y_bottom_lim=0.013, skip_val=5)

    # Only FOSI-HB, K-FAC, and L-BFGS
    plot_train_valid_loss_over_wall_sec_ord(root_result_folder + 'test_results_logistic_regression_mnist/', 'logistic_regression_mnist_second_order.pdf', 0.3, b_second_order_algos=True)
    plot_train_valid_loss_over_wall_sec_ord(root_result_folder + 'test_results_autoencoder_cifar10/', 'autoencoder_cifar10_second_order.pdf', 60, b_second_order_algos=True)
    plot_train_valid_loss_over_wall_sec_ord(root_result_folder + 'test_results_transfer_learning_cifar10/', 'transfer_learning_cifar10_second_order.pdf', 0.8, b_second_order_algos=True)
    plot_train_valid_loss_over_wall_sec_ord(root_result_folder + 'test_results_rnn_shakespeare/', 'rnn_shakespeare_second_order.pdf', 2.7, b_second_order_algos=True)
    plot_train_valid_loss_over_wall_sec_ord(root_result_folder + 'test_results_mobilenet_audioset/', 'mobilenet_audioset_second_order.pdf', 0.035, b_second_order_algos=True, skip_val=5)
