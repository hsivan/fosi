import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import pandas as pd

import tensorflow as tf
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.lib import xla_bridge
import haiku as hk
from haiku.nets import MobileNetV1
import kfac_jax

from experiments.dnn.dnn_test_utils import start_test, get_config, write_config_to_file

print(jax.local_devices())
print(xla_bridge.get_backend().platform)
tf.config.experimental.set_visible_devices([], "GPU")


def train_mobilenet(optimizer_name, learning_rate, momentum):

    def sigmoid_cross_entropy(logits, labels):
        """ Computes sigmoid cross entropy given logits and multiple class labels. """
        logits = logits.astype(jnp.float32)
        kfac_jax.register_sigmoid_cross_entropy_loss(logits, labels)
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits)
        loss = -labels * log_p - (1. - labels) * log_not_p
        return jnp.asarray(loss)

    def loss_fn_with_state(params, state, batch_data):
        """ Implements cross-entropy loss function.

        Args:
            params: Parameters of the network
            batch_data: A batch of data (images and labels)
        Returns:
            Loss calculated for the current batch
        """
        inputs, targets = batch_data
        preds, state = model.apply(params, state, None, inputs, is_training=True)
        loss = sigmoid_cross_entropy(logits=preds, labels=targets).mean()

        # This is a workaround to address a bug in the kfac library.
        # The bug is that kfac doesn't support loss functions that only return state in addition to the loss (without
        # additional auxiliary data), despite claiming to support it.
        # To address this, we need to use 'value_func_has_aux=True' when calling kfac_jax.Optimizer().
        # Otherwise, the method convert_value_and_grad_to_value_func() in kfac_jax/_src/optimizer.py
        # won't convert value_and_grad_func to value_func correctly.
        # However, the loss_fn_with_state function doesn't return auxiliary data, only state.
        # To handle this, we need to convert the returned values from '(loss, state)' to '(loss, (state, None))',
        # where None is for the auxiliary data.
        # Note: A more appropriate solution would be to modify kfac_jax/_src/optimizer.py and invoke
        # convert_value_and_grad_to_value_func() with the argument 'has_aux=value_func_has_aux or value_func_has_state'
        # instead of 'has_aux=value_func_has_aux'.
        return (loss, (state, None))

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
        batch_loss, _ = loss_fn_with_state(params, state, batch_data)
        batch_accuracy = calculate_accuracy(params, batch_data, state)
        return batch_loss, batch_accuracy

    np.random.seed(1234)

    class MyDataset(data.Dataset):

        def __init__(self, annotations_file_csv='./audioset_dataset/index.csv', img_dir='./audioset_dataset/train_jpg',
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

    trainset = MyDataset(annotations_file_csv='./audioset_dataset/index_train.csv', img_dir='./audioset_dataset/train_jpg')
    train_ds = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    validset = MyDataset(annotations_file_csv='./audioset_dataset/index_valid.csv', img_dir='./audioset_dataset/valid_jpg')
    val_ds = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    num_train_batches = len(train_ds)
    num_valid_batches = len(val_ds)

    print("num_train_batches:", num_train_batches, "num_valid_batches:", num_valid_batches)

    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum,
                      num_iterations_between_ese=800, approx_l=0, alpha=0.01, num_warmup_iterations=num_train_batches)
    test_folder = start_test(conf["optimizer"], test_folder='test_results_mobilenet_audioset')
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

    # Use adaptive_learning_rate and adaptive_momentum
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn_with_state, has_aux=True),
        l2_reg=0.0,
        value_func_has_aux=True,
        value_func_has_state=True,
        use_adaptive_learning_rate=True if conf["learning_rate"] is None else False,
        use_adaptive_momentum=True if conf["momentum"] is None else False,
        use_adaptive_damping=True,
        initial_damping=1.0,
        multi_device=False
    )
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    opt_state = optimizer.init(net_params, key, batch, state)

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
        for step, batch_data in enumerate(train_ds):
            bd = (jnp.array(batch_data[0], jnp.float32), jnp.array(batch_data[1], jnp.int32))
            iteration_start = timer()

            rng, key = jax.random.split(rng)
            net_params, opt_state, state, stats = optimizer.step(net_params, opt_state, key, batch=bd, func_state=state,
                                                                 global_step_int=i * num_train_batches + step,
                                                                 learning_rate=conf["learning_rate"], momentum=conf["momentum"])
            loss_value = stats['loss']
            acc = calculate_accuracy(net_params, bd, state)
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

        # Early stopping if loss_value is inf or None or just too high
        if jnp.isnan(epoch_train_loss) or not jnp.isfinite(epoch_train_loss) or epoch_train_loss > 1000:
            print("Early stopping with train loss", epoch_train_loss)
            break


if __name__ == "__main__":
    # None learning_rate indicates the optimizer to set use_adaptive_learning_rate to True and
    # None momentum to set use_adaptive_momentum to True
    learning_rates = [None, 1e-3, 1e-2, 1e-1]
    momentums = [None, 0.1, 0.5, 0.9]

    for learning_rate in learning_rates:
        for momentum in momentums:
            train_mobilenet('kfac', learning_rate, momentum)
