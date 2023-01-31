import os
import datetime
from timeit import default_timer as timer
import logging
import sys

#logger = logging.getLogger('lanczos')


def _prepare_test_folder(test_name, test_folder="", logging_level=logging.WARNING):
    if test_folder == "":
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), 'test_results/results_' + test_name + "_" + test_timestamp)
    else:
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), test_folder + '/results_' + test_name + "_" + test_timestamp)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    return test_folder


def start_test(test_name, test_folder="", logging_level=logging.WARNING):
    test_folder = _prepare_test_folder(test_name, test_folder, logging_level)
    return test_folder


def end_test():
    pass


def get_config(optimizer='adam', learning_rate=1e-3, num_epochs=100, batch_size=128,
               lanczos_order=20, num_batches_to_approx_lr=1, momentum=0.9,
               num_iterations_between_ese=2, approx_newton_k=5, approx_newton_l=0, num_warmup_iterations=None, alpha=1.0):
    conf = dict()
    conf["optimizer"] = optimizer  # 'sgd' / 'momentum' / 'adam'
    conf["learning_rate"] = learning_rate
    conf["num_epochs"] = num_epochs
    conf["batch_size"] = batch_size
    conf["lanczos_order"] = lanczos_order  # Num iterations of Lanczos
    conf["num_batches_to_approx_lr"] = num_batches_to_approx_lr
    conf["momentum"] = momentum
    conf["num_iterations_between_ese"] = num_iterations_between_ese  # Approximate lr using Lanczos every num_iterations_between_ese other epoch
    conf["approx_newton_k"] = approx_newton_k  # Num leading eigenvalues and eigenvectors to take out of Lanczos
    conf["approx_newton_l"] = approx_newton_l  # Num trailing eigenvalues and eigenvectors to take out of Lanczos
    conf["num_warmup_iterations"] = num_warmup_iterations  # Number of warmup iterations before running Lanczos for the first time. If None, will be considerd as num_iterations_between_ese
    conf["alpha"] = alpha  # FOSI's learning rate
    return conf


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def write_config_to_file(test_folder, conf):
    with open(test_folder + "/config.txt", 'w') as txt_file:
        txt_file.write(str(conf))
