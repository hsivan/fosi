import csv
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import os
import pandas as pd

import tensorflow as tf
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import optax
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
from jax.config import config
from jax.lib import xla_bridge
import haiku as hk
from haiku.nets import MobileNetV1

from fosi.fosi_optimizer import fosi_momentum, fosi_adam
from test_utils import start_test, get_config, write_config_to_file


print(jax.local_devices())
print(xla_bridge.get_backend().platform)
tf.config.experimental.set_visible_devices([], "GPU")


def train_mobilenet(optimizer_name):
    config.update("jax_enable_x64", False)

    def sigmoid_cross_entropy(logits, labels):
        """ Computes sigmoid cross entropy given logits and multiple class labels. """
        logits = logits.astype(jnp.float32)
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits)
        loss = -labels * log_p - (1. - labels) * log_not_p
        return jnp.asarray(loss)

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
        preds, _ = model.apply(params, state, None, inputs, is_training=True)
        return sigmoid_cross_entropy(logits=preds, labels=targets).mean()

    @jit
    def loss_fn_with_state(params, batch_data, state):
        """ Implements cross-entropy loss function.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Loss calculated for the current batch
        """
        inputs, targets = batch_data
        preds, state = model.apply(params, state, None, inputs, is_training=True)
        return sigmoid_cross_entropy(logits=preds, labels=targets).mean(), state

    def calculate_accuracy(params, batch_data, state):
        """ Implements accuracy metric.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Accuracy for the current batch
        """
        inputs, targets = batch_data
        preds, _ = model.apply(params, state, None, inputs, is_training=False)
        predicted_class = jnp.argmax(preds, axis=1)
        indexes = (jnp.array(jnp.arange(0, targets.shape[0])), predicted_class)
        target_class = targets[indexes]
        return jnp.mean(target_class)

    @jit
    def inference(params, batch_data, state):
        """ Implements train step.

        Args:
            opt_state: Current state of the optimizer
            batch_data: A batch of data (images and labels)
        Returns:
            Batch loss, batch accuracy
        """
        batch_loss, _ = loss_fn_with_state(params, batch_data, state)
        batch_accuracy = calculate_accuracy(params, batch_data, state)
        return batch_loss, batch_accuracy

    @jit
    def train_step(opt_state, params, batch_data, state):
        """ Implements train step.

        Args:
            step: Integer representing the step index
            opt_state: Current state of the optimizer
            batch_data: A batch of data (images and labels)
        Returns:
            Batch loss, batch accuracy, updated optimizer state
        """
        (batch_loss, state), batch_gradients = value_and_grad(loss_fn_with_state, has_aux=True)(params, batch_data, state)
        batch_accuracy = calculate_accuracy(params, batch_data, state)

        deltas, opt_state = optimizer.update(batch_gradients, opt_state, params)
        params = optax.apply_updates(params, deltas)

        return batch_loss, batch_accuracy, opt_state, params, state

    np.random.seed(1234)

    class MyDataset(data.Dataset):

        def __init__(self, annotations_file_csv='./audio/audioset/index.csv', img_dir='./audioset_dataset/train_jpg',
                     transform=transforms.Compose([transforms.ToTensor()])):
            super().__init__()
            df = pd.read_csv(annotations_file_csv, converters={
                "labels_as_indices": lambda x: np.array(x.strip("[]").replace("'", "").split(", ")).astype(int)})
            self.df_data = df.values
            self.data_dir = img_dir
            self.transform = transform
            self.num_classes = 527

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name, label = self.df_data[index]
            img_path = os.path.join(self.data_dir, img_name)
            image = plt.imread(img_path)
            if self.transform is not None:
                image = self.transform(np.array(image))
            target = torch.zeros(self.num_classes)
            target[label] = 1.
            return image, target

    batch_size = 256

    trainset = MyDataset(annotations_file_csv='./audio/audioset/index_train.csv', img_dir='./audioset_dataset/train_jpg')
    train_ds = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    validset = MyDataset(annotations_file_csv='./audio/audioset/index_valid.csv', img_dir='./audioset_dataset/valid_jpg')
    val_ds = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    num_train_batches = len(train_ds)
    num_valid_batches = len(val_ds)

    print("num_train_batches:", num_train_batches, "num_valid_batches:", num_valid_batches)

    learning_rate = 1e-1 if 'momentum' in optimizer_name else 1e-3
    conf = get_config(optimizer=optimizer_name, approx_newton_k=10, lanczos_order=40, batch_size=batch_size, learning_rate=learning_rate, num_iterations_between_ese=800, approx_newton_l=0, alpha=0.01, num_warmup_iterations=num_train_batches)
    test_folder = start_test(conf["optimizer"], test_folder='test_results_mobilenet')
    write_config_to_file(test_folder, conf)

    def _model(images, is_training):
        net = MobileNetV1(num_classes=527)
        return net(images, is_training)

    model = hk.transform_with_state(_model)

    # We have defined our model. We need to initialize the params based on the input shape.
    batch = next(iter(train_ds))
    batch = (jnp.array(batch[0], jnp.float32), jnp.array(batch[1], jnp.int32))
    x = jnp.asarray(batch[0][0], dtype=jnp.float32)
    x = jnp.expand_dims(x, axis=0)
    net_params, state = model.init(random.PRNGKey(111), x, is_training=True)

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
        for batch_data in train_ds:
            bd = (jnp.array(batch_data[0], jnp.float32), jnp.array(batch_data[1], jnp.int32))
            iteration_start = timer()
            loss_value, acc, opt_state, net_params, state = train_step(opt_state, net_params, bd, state)
            iteration_end = timer()

            train_batch_loss.append(loss_value)
            train_batch_acc.append(acc)
            iteration_latency.append(iteration_end - iteration_start)

        epoch_end = timer()

        # Evaluation on validation data every 5 epochs
        if i % 5 == 0:
            for batch_data in val_ds:
                bd = (jnp.array(batch_data[0], jnp.float32), jnp.array(batch_data[1], jnp.int32))
                loss_value, acc = inference(net_params, bd, state)
                valid_batch_loss.append(loss_value)
                valid_batch_acc.append(acc)

            # Loss for the current epoch
            epoch_valid_loss = np.mean(valid_batch_loss)
            # Accuracy for the current epoch
            epoch_valid_acc = np.mean(valid_batch_acc)

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_acc)

        print(f"loss: {epoch_train_loss:.3f}  acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}  latency: {epoch_end - epoch_start}")
        with open(train_stats_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [i, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc, epoch_end - epoch_start,
                 np.maximum(0, timer() - start_time)])


if __name__ == "__main__":
    for optimizer_name in ['my_adam', 'my_momentum', 'adam', 'momentum']:
        train_mobilenet(optimizer_name)
