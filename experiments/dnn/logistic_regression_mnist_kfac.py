import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178

from experiments.dnn.logistic_regression_mnist import get_normalized_dataset, Model, data_generator

os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
import jax
import numpy as np
from timeit import default_timer as timer

import jax.numpy as jnp
from jax import random
from jax import jit
import kfac_jax

from experiments.dnn.dnn_test_utils import start_test, get_config, write_config_to_file

print(jax.local_devices())


def train_mnist(optimizer_name, learning_rate, momentum):

    def loss_fn(params, batch_data):
        """ Implements cross-entropy loss function.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Loss calculated for the current batch
        """
        inputs, targets = batch_data
        logits = net_apply(params, inputs)
        kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(log_p * targets, axis=1))

    @jit
    def calculate_accuracy(params, batch_data):
        """ Implements accuracy metric.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Accuracy for the current batch
        """
        inputs, targets = batch_data
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(net_apply(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    @jit
    def inference(params, batch_data):
        """ Implements train step.

        Args:
            opt_state: Current state of the optimizer
            batch_data: A batch of data (images and labels)
        Returns:
            Batch loss, batch accuracy
        """
        batch_loss = loss_fn(params, batch_data)
        batch_accuracy = calculate_accuracy(params, batch_data)
        return batch_loss, batch_accuracy

    np.random.seed(1234)
    x_train_normalized, y_train_ohe, x_valid_normalized, y_valid_ohe = get_normalized_dataset()
    net_init, net_apply = Model(num_classes=10)

    batch_size = 1024
    num_train_batches = len(x_train_normalized) // batch_size
    num_valid_batches = len(x_valid_normalized) // batch_size
    print("num_train_batches:", num_train_batches, "num_valid_batches:", num_valid_batches)

    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=batch_size, momentum=momentum,
                      learning_rate=learning_rate, num_iterations_between_ese=800, approx_l=0, alpha=0.01,
                      num_warmup_iterations=num_train_batches)
    test_folder = start_test(conf["optimizer"], test_folder="test_results_logistic_regression_mnist")
    write_config_to_file(test_folder, conf)

    # We have defined our model. We need to initialize the params based on the input shape.
    # The images in our dataset are of shape (32, 32, 3). Hence we will initialize the
    # network with the input shape (-1, 32, 32, 3). -1 represents the batch dimension here.
    net_out_shape, net_params = net_init(random.PRNGKey(111), input_shape=(-1, 784))

    train_data_gen = data_generator(x_train_normalized, y_train_ohe, batch_size=conf["batch_size"], is_valid=True)

    # Use static learning_rate and momentum
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=0.0,
        value_func_has_aux=False,
        value_func_has_state=False,
        use_adaptive_learning_rate=True if conf["learning_rate"] is None else False,
        use_adaptive_momentum=True if conf["momentum"] is None else False,
        use_adaptive_damping=True,
        initial_damping=1.0,
        multi_device=False
    )
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    opt_state = optimizer.init(net_params, key, list(train_data_gen)[0])

    ###############################    Training    ###############################

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    start_time = 1e10
    for i in range(conf["num_epochs"]):
        if i == 1:
            start_time = timer()

        # Lists to store loss and accuracy for each batch
        train_batch_loss, train_batch_acc = [], []
        valid_batch_loss, valid_batch_acc = [], []
        iteration_latency = []

        print(f"Epoch: {i:<3}", end=" ")
        epoch_start = timer()

        # Training
        train_data_gen = data_generator(x_train_normalized, y_train_ohe, batch_size=conf["batch_size"], is_valid=False)
        for step in range(num_train_batches):
            iteration_start = timer()
            batch_data = next(train_data_gen)
            rng, key = jax.random.split(rng)
            net_params, opt_state, stats = optimizer.step(net_params, opt_state, key, batch=batch_data, global_step_int=i*num_train_batches+step,
                                                          learning_rate=conf["learning_rate"], momentum=conf["momentum"])
            loss_value = stats['loss']
            acc = calculate_accuracy(net_params, batch_data)
            iteration_end = timer()

            train_batch_loss.append(loss_value)
            train_batch_acc.append(acc)
            iteration_latency.append(iteration_end - iteration_start)

        epoch_end = timer()

        # Evaluation on validation data
        valid_data_gen = data_generator(x_valid_normalized, y_valid_ohe, batch_size=conf["batch_size"], is_valid=True)
        for step in range(num_valid_batches):
            batch_data = next(valid_data_gen)
            loss_value, acc = inference(net_params, batch_data)
            valid_batch_loss.append(loss_value)
            valid_batch_acc.append(acc)

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_valid_loss = np.mean(valid_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_acc)
        epoch_valid_acc = np.mean(valid_batch_acc)

        print(f"loss: {epoch_train_loss:.3f}  acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}  latency: {epoch_end - epoch_start}")
        with open(train_stats_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [i, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc, epoch_end - epoch_start,
                 np.maximum(0, timer() - start_time)])

        # Early stopping if loss_value is inf or None or just too high
        if jnp.isnan(epoch_train_loss) or not jnp.isfinite(epoch_train_loss) or epoch_train_loss > 1000:
            print("Early stopping with train loss", epoch_train_loss)
            break


if __name__ == "__main__":
    # None learning_rate indicates the optimizer to set use_adaptive_learning_rate to True and
    # None momentum to set use_adaptive_momentum to True
    learning_rates = [None, 1e-3, 1e-2, 1e-1]
    momentums = [None, 0.1, 0.5, 0.9]

    # min_loss: 0.22080655 best_learning_rate: 0.001 best_momentum: 0.9
    for learning_rate in learning_rates:
        for momentum in momentums:
            train_mnist('kfac', learning_rate, momentum)
