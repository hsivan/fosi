import os
import datetime


def _prepare_test_folder(test_name, test_folder=""):
    if test_folder == "":
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), 'test_results/results_' + test_name + "_" + test_timestamp)
    else:
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        test_folder = os.path.join(os.getcwd(), test_folder + '/results_' + test_name + "_" + test_timestamp)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    return test_folder


def start_test(test_name, test_folder=""):
    test_folder = _prepare_test_folder(test_name, test_folder)
    return test_folder


def end_test():
    pass


def get_config(optimizer='adam', learning_rate=1e-3, num_epochs=100, batch_size=128, momentum=0.9,
               num_iterations_between_ese=2, approx_k=5, approx_l=0, num_warmup_iterations=None, alpha=1.0):
    conf = dict()
    conf["optimizer"] = optimizer  # Base optimizer. Could be 'sgd' / 'momentum' / 'adam'
    conf["learning_rate"] = learning_rate  # Learning rate of the base optimizer
    conf["num_epochs"] = num_epochs
    conf["batch_size"] = batch_size
    conf["momentum"] = momentum  # Momentum/decay parameter for evaluating the momentum (first moment) term by the base optimizer
    conf["num_iterations_between_ese"] = num_iterations_between_ese  # Call the ESE procedure once every num_iterations_between_ese
    conf["approx_k"] = approx_k  # Num leading eigenvalues and eigenvectors to take out of Lanczos
    conf["approx_l"] = approx_l  # Num trailing eigenvalues and eigenvectors to take out of Lanczos
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
