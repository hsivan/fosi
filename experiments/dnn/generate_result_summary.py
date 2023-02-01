import os
import numpy as np
import pandas as pd


def create_loss_summary_table(test_result_root_folders, dataset_type='val'):
    precision = '%0.3f'

    loss_summary_file_name = dataset_type + "_loss_summary.csv"

    with open(loss_summary_file_name, 'w') as f:
        f.write('Task & HB & FOSI HB & Adam & FOSI Adam' + r' \\' + '\n')
        f.write(r'\midrule' + '\n' + r'\arrayrulecolor{lightgray}' + '\n')

    for task, test_result_root_folder in test_result_root_folders.items():
        test_folders = [test_result_root_folder + f for f in os.listdir(test_result_root_folder) if f.startswith('results')]

        for test_folder in test_folders:
            df = pd.read_csv(test_folder + '/train_stats.csv')

            max_data_point = len(df) if task != 'LM' else 71

            loss = df[dataset_type + '_loss'][:max_data_point].values[-1]
            if 'results_momentum' in test_folder:
                loss_momentum = loss
            if 'results_my_momentum' in test_folder:
                loss_my_momentum = loss
            if 'results_adam' in test_folder:
                loss_adam = loss
            if 'results_my_adam' in test_folder:
                loss_my_adam = loss

        loss_momentum_str = precision % loss_momentum
        loss_my_momentum_str = precision % loss_my_momentum
        loss_adam_str = precision % loss_adam
        loss_my_adam_str = precision % loss_my_adam

        if loss_momentum < loss_my_momentum:
            loss_momentum_str = r'\textbf{' + loss_momentum_str + '}'
        else:
            loss_my_momentum_str = r'\textbf{' + loss_my_momentum_str + '}'

        if loss_adam < loss_my_adam:
            loss_adam_str = r'\textbf{' + loss_adam_str + '}'
        else:
            loss_my_adam_str = r'\textbf{' + loss_my_adam_str + '}'

        with open(loss_summary_file_name, 'a') as f:
            f.write(task + " & " +
                    loss_momentum_str + " & " + loss_my_momentum_str + " & " +
                    loss_adam_str + " & " + loss_my_adam_str + r' \\' + "\n")
            if task != "LR":
                f.write(r'\cmidrule(r){1-5}' + "\n")


def create_wall_time_to_loss_summary_table(dataset_type='val'):
    # Get the wall time of the best train/validation loss of the base optimizer and then find FOSI's wall time in which
    # its train/validation loss is a good as the best loss of the base optimizer.

    fosi_improvement = {'adam': [], 'momentum': []}

    wall_time_summary_file_name = dataset_type + "_wall_time_to_loss_summary.csv"
    with open(wall_time_summary_file_name, 'w') as f:
        f.write('Task & HB & FOSI HB & Adam & FOSI Adam' + r' \\' + '\n')
        f.write(r'\midrule' + '\n' + r'\arrayrulecolor{lightgray}' + '\n')

    for task, test_result_root_folder in test_result_root_folders.items():

        test_folders = [test_result_root_folder + f for f in os.listdir(test_result_root_folder) if f.startswith('results')]

        for optimizer_technique in ['adam', 'momentum']:
            base_test_folder = [f for f in test_folders if 'results_' + optimizer_technique in f][0]
            fosi_test_folder = [f for f in test_folders if 'results_my_' + optimizer_technique in f][0]
            df_base = pd.read_csv(base_test_folder + '/train_stats.csv')
            df_fosi = pd.read_csv(fosi_test_folder + '/train_stats.csv')

            max_data_point = len(df_base) if task != 'LM' else 71

            base_losses = df_base[dataset_type + '_loss'][:max_data_point][df_base[dataset_type +'_loss'] != 0].values
            min_loss_base_idx = np.argmin(base_losses)
            min_loss_base = base_losses[min_loss_base_idx]
            base_wall_times = df_base['wall_time'][:max_data_point][df_base[dataset_type + '_loss'] != 0].values
            min_loss_base_wall_time = base_wall_times[min_loss_base_idx]

            fosi_losses = df_fosi[dataset_type + '_loss'][:max_data_point][df_fosi[dataset_type + '_loss'] != 0].values
            min_loss_fosi_idx = np.argmin(fosi_losses)
            fosi_wall_times = df_fosi['wall_time'][:max_data_point][df_fosi[dataset_type + '_loss'] != 0].values
            min_loss_fosi_wall_time = fosi_wall_times[min_loss_fosi_idx]

            # Find the first index where FOSI's loss is a good as min_loss_base
            for idx, fosi_loss in enumerate(fosi_losses):
                if fosi_loss <= min_loss_base:
                    min_loss_fosi_wall_time = fosi_wall_times[idx]
                    break

            fosi_improvement[optimizer_technique].append(min_loss_fosi_wall_time / min_loss_base_wall_time)

            if min_loss_fosi_wall_time <= min_loss_base_wall_time:
                min_loss_base_wall_time = str(int(min_loss_base_wall_time))
                min_loss_fosi_wall_time = r'\textbf{' + str(int(min_loss_fosi_wall_time)) + '}'
            else:
                min_loss_base_wall_time = r'\textbf{' + str(int(min_loss_base_wall_time)) + '}'
                min_loss_fosi_wall_time = str(int(min_loss_fosi_wall_time))

            if optimizer_technique == 'adam':
                adam_min_loss_wall_time = min_loss_base_wall_time
                fosi_adam_min_loss_wall_time = min_loss_fosi_wall_time
            else:
                momentum_min_loss_wall_time = min_loss_base_wall_time
                fosi_momentum_min_loss_wall_time = min_loss_fosi_wall_time

        with open(wall_time_summary_file_name, 'a') as f:
            f.write(task + " & " +
                    momentum_min_loss_wall_time + " & " + fosi_momentum_min_loss_wall_time + " & " +
                    adam_min_loss_wall_time + " & " + fosi_adam_min_loss_wall_time + r' \\' + "\n")
            if task != "LR":
                f.write(r'\cmidrule(r){1-5}' + "\n")

    print("############", dataset_type, "time to loss ############")
    for optimizer_technique, fosi_improvements in fosi_improvement.items():
        print("FOSI", optimizer_technique, "avg relative time to convergence:", np.mean(fosi_improvements))

    all_improvements = fosi_improvement['adam'] + fosi_improvement['momentum']
    print("FOSI avg relative time to convergence:", np.mean(all_improvements))


def create_wall_time_to_acc_summary_table(dataset_type='val'):
    # Get the wall time of the best train/validation loss of the base optimizer and then find FOSI's wall time in which
    # its train/validation loss is a good as the best loss of the base optimizer.

    fosi_improvement = {'adam': [], 'momentum': []}

    wall_time_summary_file_name = dataset_type + "_wall_time_to_acc_summary.csv"
    with open(wall_time_summary_file_name, 'w') as f:
        f.write('Task & HB & FOSI HB & Adam & FOSI Adam' + r' \\' + '\n')
        f.write(r'\midrule' + '\n' + r'\arrayrulecolor{lightgray}' + '\n')

    for task, test_result_root_folder in test_result_root_folders.items():

        test_folders = [test_result_root_folder + f for f in os.listdir(test_result_root_folder) if f.startswith('results')]

        for optimizer_technique in ['adam', 'momentum']:
            base_test_folder = [f for f in test_folders if 'results_' + optimizer_technique in f][0]
            fosi_test_folder = [f for f in test_folders if 'results_my_' + optimizer_technique in f][0]
            df_base = pd.read_csv(base_test_folder + '/train_stats.csv')
            df_fosi = pd.read_csv(fosi_test_folder + '/train_stats.csv')

            max_data_point = len(df_base) if task != 'LM' else 71

            column = dataset_type + '_acc' if dataset_type + '_acc' in df_base.columns else dataset_type + '_loss'

            base_losses = df_base[column][:max_data_point][df_base[column] != 0].values
            if 'acc' in column:
                min_loss_base_idx = np.argmax(base_losses)
            else:
                min_loss_base_idx = np.argmin(base_losses)
            min_loss_base = base_losses[min_loss_base_idx]
            base_wall_times = df_base['wall_time'][:max_data_point][df_base[column] != 0].values
            min_loss_base_wall_time = base_wall_times[min_loss_base_idx]

            fosi_losses = df_fosi[column][:max_data_point][df_fosi[column] != 0].values
            if 'acc' in column:
                min_loss_fosi_idx = np.argmax(fosi_losses)
            else:
                min_loss_fosi_idx = np.argmin(fosi_losses)
            fosi_wall_times = df_fosi['wall_time'][:max_data_point][df_fosi[column] != 0].values
            min_loss_fosi_wall_time = fosi_wall_times[min_loss_fosi_idx]

            # Find the first index where FOSI's loss is a good as min_loss_base
            for idx, fosi_loss in enumerate(fosi_losses):
                if 'acc' in column:
                    if fosi_loss >= min_loss_base:
                        min_loss_fosi_wall_time = fosi_wall_times[idx]
                        break
                else:
                    if fosi_loss <= min_loss_base:
                        min_loss_fosi_wall_time = fosi_wall_times[idx]
                        break

            # Ignore Audio Classification (MobileNet) task with 'adam' and 'FOSI adam' results, as Adam overfits
            fosi_improvement[optimizer_technique].append(min_loss_fosi_wall_time / min_loss_base_wall_time)

            if min_loss_fosi_wall_time <= min_loss_base_wall_time:
                min_loss_base_wall_time = str(int(min_loss_base_wall_time))
                min_loss_fosi_wall_time = r'\textbf{' + str(int(min_loss_fosi_wall_time)) + '}'
            else:
                min_loss_base_wall_time = r'\textbf{' + str(int(min_loss_base_wall_time)) + '}'
                min_loss_fosi_wall_time = str(int(min_loss_fosi_wall_time))

            if optimizer_technique == 'adam':
                adam_min_loss = min_loss_base
                adam_min_loss_wall_time = min_loss_base_wall_time
                fosi_adam_min_loss_wall_time = min_loss_fosi_wall_time
            else:
                momentum_min_loss = min_loss_base
                momentum_min_loss_wall_time = min_loss_base_wall_time
                fosi_momentum_min_loss_wall_time = min_loss_fosi_wall_time

        with open(wall_time_summary_file_name, 'a') as f:
            f.write(task + " & " +
                    momentum_min_loss_wall_time + " & " + fosi_momentum_min_loss_wall_time + ' (' + '%0.3f' % momentum_min_loss + ')' + " & " +
                    adam_min_loss_wall_time + " & " + fosi_adam_min_loss_wall_time + ' (' + '%0.3f' % adam_min_loss + ')' + r' \\' + "\n")
            if task != "LR":
                f.write(r'\cmidrule(r){1-5}' + "\n")

    print("############", dataset_type, "time to accuracy ############")
    for optimizer_technique, fosi_improvements in fosi_improvement.items():
        print("FOSI", optimizer_technique, "avg relative time to convergence:", np.mean(fosi_improvements))

    all_improvements = fosi_improvement['adam'] + fosi_improvement['momentum']
    print("FOSI avg relative time to convergence:", np.mean(all_improvements))


if __name__ == "__main__":

    test_result_root_folders = {'AC': './test_results_mobilenet/',
                                'LM': './test_results_rnn/',
                                'AE': './test_results_autoencoder_128/',
                                'TL': './test_results_transfer_learning/',
                                'LR': './test_results_logistic_regression/'}

    create_loss_summary_table(test_result_root_folders, dataset_type='train')
    create_loss_summary_table(test_result_root_folders, dataset_type='val')

    create_wall_time_to_loss_summary_table(dataset_type='train')
    create_wall_time_to_loss_summary_table(dataset_type='val')

    create_wall_time_to_acc_summary_table(dataset_type='train')
    create_wall_time_to_acc_summary_table(dataset_type='val')