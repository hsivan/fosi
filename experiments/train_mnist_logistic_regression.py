import csv
import jax
import numpy as np
from timeit import default_timer as timer

import optax
import jax.numpy as jnp
from jax import random
from jax import value_and_grad
from jax import jit
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Flatten, LogSoftmax
from tensorflow.keras.datasets import mnist

from fosi.fosi_optimizer import fosi_adam, fosi_momentum
from test_utils import start_test, get_config, write_config_to_file

from jax.config import config

print(jax.local_devices())


def train_mnist(optimizer_name):
    config.update("jax_enable_x64", False)

    def data_generator(images, labels, batch_size=128, is_valid=False):
        # 1. Calculate the total number of batches
        num_batches = int(np.ceil(len(images) / batch_size))

        # 2. Get the indices and shuffle them
        indices = np.arange(len(images))

        if not is_valid:
            # Shuffle the data for training (not required for validation).
            np.random.shuffle(indices)

        for batch in range(num_batches):
            curr_idx = indices[batch * batch_size: (batch + 1) * batch_size]
            batch_images = images[curr_idx]
            batch_labels = labels[curr_idx]

            yield batch_images, batch_labels

    def get_normalized_dataset():
        # The downloaded dataset consists of two tuples. The first tuple represents the training data consisting
        # of pairs of images and labels. Similarly, the second tuple consists of validation/test data.
        (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
        print(f"\nNumber of training samples: {len(x_train)} with samples shape: {x_train.shape[1:]}")
        print(f"Number of validation samples: {len(x_valid)} with samples shape: {x_valid.shape[1:]}")

        # Normalize the image pixels in the range [0, 1]
        x_train_normalized = jnp.array(x_train / 255.)
        x_valid_normalized = jnp.array(x_valid / 255.)

        # One hot encoding applied to the labels. We have 10 classes in the dataset, hence the depth of OHE would be 10.
        y_train_ohe = jnp.squeeze(jax.nn.one_hot(y_train, num_classes=10))
        y_valid_ohe = jnp.squeeze(jax.nn.one_hot(y_valid, num_classes=10))

        print(f"Training images shape:   {x_train_normalized.shape}  Labels shape: {y_train_ohe.shape}")
        print(f"Validation images shape: {x_valid_normalized.shape}  Labels shape: {y_valid_ohe.shape}")

        return x_train_normalized, y_train_ohe, x_valid_normalized, y_valid_ohe

    def Model(num_classes=10):
        return stax.serial(
            Flatten, Dense(num_classes), LogSoftmax)

    @jit
    def loss_fn(params, batch_data):
        """ Implements cross-entropy loss function.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Loss calculated for the current batch
        """
        inputs, targets = batch_data
        preds = net_apply(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

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

    @jit
    def train_step(opt_state, params, batch_data):
        """ Implements train step.

        Args:
            opt_state: Current state of the optimizer
            params: Network parameters
            batch_data: A batch of data (images and labels)
        Returns:
            Batch loss, batch accuracy, updated optimizer state
        """
        batch_loss, batch_gradients = value_and_grad(loss_fn)(params, batch_data)
        batch_accuracy = calculate_accuracy(params, batch_data)

        deltas, opt_state = optimizer.update(batch_gradients, opt_state, params)
        params = optax.apply_updates(params, deltas)

        return batch_loss, batch_accuracy, opt_state, params

    np.random.seed(1234)
    x_train_normalized, y_train_ohe, x_valid_normalized, y_valid_ohe = get_normalized_dataset()
    net_init, net_apply = Model(num_classes=10)

    batch_size = 1024
    num_train_batches = len(x_train_normalized) // batch_size
    num_valid_batches = len(x_valid_normalized) // batch_size
    print("num_train_batches:", num_train_batches, "num_valid_batches:", num_valid_batches)

    learning_rate = 1e-1 if 'momentum' in optimizer_name else 1e-3
    conf = get_config(optimizer=optimizer_name, approx_newton_k=10, batch_size=batch_size,
                      learning_rate=learning_rate, num_iterations_between_ese=800, approx_newton_l=0, alpha=0.01,
                      num_warmup_iterations=num_train_batches)
    test_folder = start_test(conf["optimizer"], test_folder="test_results_logistic_regression")
    write_config_to_file(test_folder, conf)

    # We have defined our model. We need to initialize the params based on the input shape.
    # The images in our dataset are of shape (32, 32, 3). Hence we will initialize the
    # network with the input shape (-1, 32, 32, 3). -1 represents the batch dimension here.
    net_out_shape, net_params = net_init(random.PRNGKey(111), input_shape=(-1, 784))

    train_data_gen = data_generator(x_train_normalized, y_train_ohe, batch_size=conf["batch_size"], is_valid=True)
    batch = list(train_data_gen)[0]

    def get_optimizer():
        if conf["optimizer"] == 'momentum':
            return optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False)
        elif conf["optimizer"] == 'my_momentum':
            return fosi_momentum(optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False), loss_fn, batch,
                                 decay=conf["momentum"],
                                 num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                                 approx_newton_k=conf["approx_newton_k"],
                                 approx_newton_l=conf["approx_newton_l"], warmup_w=conf["num_warmup_iterations"],
                                 alpha=conf["alpha"], learning_rate_clip=3.0)
        elif conf["optimizer"] == 'adam':
            return optax.adam(conf["learning_rate"])
        elif conf["optimizer"] == 'my_adam':
            return fosi_adam(optax.adam(conf["learning_rate"]), loss_fn, batch,
                             decay=conf["momentum"],
                             num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                             approx_newton_k=conf["approx_newton_k"],
                             approx_newton_l=conf["approx_newton_l"], warmup_w=conf["num_warmup_iterations"],
                             alpha=conf["alpha"])
        else:
            raise "Illegal optimizer " + conf["optimizer"]

    optimizer = get_optimizer()
    opt_state = optimizer.init(net_params)

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
            loss_value, acc, opt_state, net_params = train_step(opt_state, net_params, batch_data)
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


if __name__ == "__main__":
    for optimizer_name in ['my_adam', 'my_momentum', 'adam', 'momentum']:
        train_mnist(optimizer_name)
