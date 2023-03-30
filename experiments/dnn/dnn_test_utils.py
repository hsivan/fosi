import os
import datetime
import optax

from fosi import fosi_momentum, fosi_adam, fosi_nesterov, fosi_sgd


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
               num_iterations_between_ese=100, approx_k=5, approx_l=0, num_warmup_iterations=None, alpha=0.1,
               learning_rate_clip=None):
    conf = dict()

    if learning_rate_clip is None:
        if 'adam' in optimizer:
            learning_rate_clip = 1.0  # Not used for Adam (hard coded to 1.0), but setting it for clarity.
        else:
            learning_rate_clip = 3.0  # Use 3.0 as the default value

    num_warmup_iterations = num_warmup_iterations if num_warmup_iterations is not None else num_iterations_between_ese

    conf["optimizer"] = optimizer  # Base optimizer. Could be 'sgd' / 'momentum' / 'adam' / 'nesterov' / 'fosi_sgd' / 'fosi_momentum' / 'fosi_adam' / 'fosi_nesterov
    conf["learning_rate"] = learning_rate  # Learning rate of the base optimizer
    conf["num_epochs"] = num_epochs
    conf["batch_size"] = batch_size
    conf["momentum"] = momentum  # Momentum/decay parameter for evaluating the momentum (first moment) term by the base optimizer
    conf["num_iterations_between_ese"] = num_iterations_between_ese  # Call the ESE procedure once every num_iterations_between_ese
    conf["approx_k"] = approx_k  # Num leading eigenvalues and eigenvectors to take out of Lanczos
    conf["approx_l"] = approx_l  # Num trailing eigenvalues and eigenvectors to take out of Lanczos
    conf["num_warmup_iterations"] = num_warmup_iterations  # Number of warmup iterations before running Lanczos for the first time. If None, will be considerd as num_iterations_between_ese
    conf["alpha"] = alpha  # FOSI's learning rate
    conf["learning_rate_clip"] = learning_rate_clip
    return conf


def read_config_file(test_folder):
    with open(test_folder + "/config.txt") as conf_file:
        conf = conf_file.read()
    conf = eval(conf)
    return conf


def write_config_to_file(test_folder, conf):
    with open(test_folder + "/config.txt", 'w') as txt_file:
        txt_file.write(str(conf))


def get_optimizer(conf, loss_fn, batch, b_call_ese_internally=True):
    if conf["optimizer"] == 'sgd':
        optimizer = optax.sgd(conf["learning_rate"])
    elif conf["optimizer"] == 'momentum':
        optimizer = optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False)
    elif conf["optimizer"] == 'nesterov':
        optimizer = optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=True)
    elif conf["optimizer"] == 'adam':
        optimizer = optax.adam(conf["learning_rate"])
    elif conf["optimizer"] == 'fosi_sgd':
        optimizer = fosi_sgd(optax.sgd(conf["learning_rate"]), loss_fn, batch,
                             num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                             approx_k=conf["approx_k"], approx_l=conf["approx_l"],
                             warmup_w=conf["num_warmup_iterations"], alpha=conf["alpha"],
                             learning_rate_clip=conf["learning_rate_clip"], b_call_ese_internally=b_call_ese_internally)
    elif conf["optimizer"] == 'fosi_momentum':
        optimizer = fosi_momentum(optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False), loss_fn,
                                  batch, decay=conf["momentum"],
                                  num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                                  approx_k=conf["approx_k"], approx_l=conf["approx_l"],
                                  warmup_w=conf["num_warmup_iterations"], alpha=conf["alpha"],
                                  learning_rate_clip=conf["learning_rate_clip"], b_call_ese_internally=b_call_ese_internally)
    elif conf["optimizer"] == 'fosi_nesterov':
        optimizer = fosi_nesterov(optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=True), loss_fn,
                                  batch, decay=conf["momentum"],
                                  num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                                  approx_k=conf["approx_k"], approx_l=conf["approx_l"],
                                  warmup_w=conf["num_warmup_iterations"], alpha=conf["alpha"],
                                  learning_rate_clip=conf["learning_rate_clip"], b_call_ese_internally=b_call_ese_internally)
    elif conf["optimizer"] == 'fosi_adam':
        optimizer = fosi_adam(optax.adam(conf["learning_rate"]), loss_fn, batch, decay=conf["momentum"],
                              num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                              approx_k=conf["approx_k"], approx_l=conf["approx_l"],
                              warmup_w=conf["num_warmup_iterations"], alpha=conf["alpha"], b_call_ese_internally=b_call_ese_internally)
    else:
        raise ValueError("Illegal optimizer " + conf["optimizer"])

    return optimizer
