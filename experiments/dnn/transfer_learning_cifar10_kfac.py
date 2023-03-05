# Based on https://www.kaggle.com/code/yashvi/transfer-learning-using-jax-flax/notebook

import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
import numpy as np
import warnings
from functools import partial
from tqdm.auto import tqdm
from timeit import default_timer as timer

import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax import jit
from jax_resnet import pretrained_resnet, slice_variables, Sequential
from flax import linen as nn
from flax.core import FrozenDict
import optax
import kfac_jax

from experiments.dnn.dnn_test_utils import get_config, start_test, write_config_to_file

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.config.experimental.set_visible_devices([], "GPU")


def train_transfer_learning(optimizer_name, learning_rate, momentum):
    Config = {
        'NUM_LABELS': 10,
        'N_SPLITS': 5,
        'BATCH_SIZE': 128,
        'N_EPOCHS': 20,
        'LR': 0.001,
        'WIDTH': 32,
        'HEIGHT': 32,
        'IMAGE_SIZE': 128,
        'WEIGHT_DECAY': 1e-5,
        'FREEZE_BACKBONE': True
    }

    def transform_images(row, size):
        """
        Resize image
        INPUT row , size
        RETURNS resized image and its label
        """
        x_train = tf.image.resize(row['image'], (size, size))
        return x_train, row['label']

    def load_datasets():
        """
        load and transform dataset from tfds
        RETURNS train and test dataset

        """
        # Construct a tf.data.Dataset
        train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True)

        train_ds = train_ds.map(lambda row: transform_images(row, Config["IMAGE_SIZE"]))
        test_ds = test_ds.map(lambda row: transform_images(row, Config["IMAGE_SIZE"]))

        # Build your input pipeline
        train_dataset = train_ds.batch(Config["BATCH_SIZE"], drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_ds.batch(Config["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    class Head(nn.Module):
        """ head model """
        batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)

        @nn.compact
        def __call__(self, inputs, train: bool):
            x = nn.Dense(features=Config["NUM_LABELS"])(inputs)
            return x

    class Model(nn.Module):
        """ Combines backbone and head model """
        backbone: Sequential
        head: Head

        def __call__(self, inputs, train: bool):
            x = self.backbone(inputs)
            # average pool layer
            x = jnp.mean(x, axis=(1, 2))
            x = self.head(x, train)
            return x

    def _get_backbone_and_params(model_arch: str):
        """
        Get backbone and params
        1. Loads pretrained model (resnet18)
        2. Get model and param structure except last 2 layers
        3. Extract the corresponding subset of the variables dict
        INPUT : model_arch
        RETURNS backbone , backbone_params
        """
        if model_arch == 'resnet18':
            resnet_tmpl, params = pretrained_resnet(18)
            model = resnet_tmpl()
        else:
            raise NotImplementedError

        # get model & param structure for backbone
        start, end = 0, len(model.layers) - 2
        backbone = Sequential(model.layers[start:end])
        backbone_params = slice_variables(params, start, end)
        return backbone, backbone_params

    def get_model_and_variables(model_arch: str, head_init_key: int):
        """
        Get model and variables
        1. Initialise inputs(shape=(1,image_size,image_size,3))
        2. Get backbone and params
        3. Apply backbone model and get outputs
        4. Initialise head
        5. Create final model using backbone and head
        6. Combine params from backbone and head

        INPUT model_arch, head_init_key
        RETURNS  model, variables
        """

        # backbone
        inputs = jnp.ones((1, Config['IMAGE_SIZE'], Config['IMAGE_SIZE'], 3), jnp.float32)
        backbone, backbone_params = _get_backbone_and_params(model_arch)
        key = jax.random.PRNGKey(head_init_key)
        backbone_output = backbone.apply(backbone_params, inputs, mutable=False)

        # head
        head_inputs = jnp.ones((1, backbone_output.shape[-1]), jnp.float32)
        head = Head()
        head_params = head.init(key, head_inputs, train=False)

        # final model
        model = Model(backbone, head)
        variables = FrozenDict({
            'params': {
                'backbone': backbone_params['params'],
                'head': head_params['params']
            },
            'batch_stats': {
                'backbone': backbone_params['batch_stats'],
                # 'head': head_params['batch_stats']
            }
        })
        return model, variables

    model, variables = get_model_and_variables('resnet18', 0)
    inputs = jnp.ones((1, Config['IMAGE_SIZE'], Config['IMAGE_SIZE'], 3), jnp.float32)
    _ = model.apply(variables, inputs, train=False, mutable=False)

    train_dataset, test_dataset = load_datasets()
    print(len(train_dataset))

    num_train_steps = len(train_dataset)
    print(num_train_steps)

    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=Config['BATCH_SIZE'], momentum=momentum,
                      learning_rate=learning_rate, num_iterations_between_ese=800, approx_l=0,
                      num_warmup_iterations=len(train_dataset), alpha=0.01)

    test_folder = start_test(conf["optimizer"], test_folder='test_results_transfer_learning_cifar10')
    write_config_to_file(test_folder, conf)

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    batch = next(iter(train_dataset))
    batch = (jnp.array(batch[0], dtype=jnp.float32), jnp.array(batch[1], dtype=jnp.float32))

    print("Number of frozen parameters:", ravel_pytree(variables['params']['backbone'])[0].shape[0])

    def accuracy(logits, labels):
        """
        calculates accuracy based on logits and labels
        INPUT logits , labels
        RETURNS accuracy
        """
        return jnp.mean(jnp.argmax(logits, -1) == labels)

    params = variables['params']
    batch_stats = variables['batch_stats']

    def loss_function(params, batch_stats, batch_data):
        variables_ = {'params': {'backbone': variables['params']['backbone'], 'head': params['head']}, 'batch_stats': batch_stats}
        inputs, targets = batch_data
        logits, new_batch_stats = model.apply(variables_, inputs, train=True, mutable=['batch_stats'], rngs={'dropout': jax.random.PRNGKey(0)})
        # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
        one_hot = jax.nn.one_hot(targets, Config["NUM_LABELS"])
        kfac_jax.register_softmax_cross_entropy_loss(logits, one_hot)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        acc = accuracy(logits, targets)
        return loss, (new_batch_stats['batch_stats'], acc)

    # Use static learning_rate and momentum
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_function, has_aux=True),
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
    opt_state = optimizer.init(variables['params'], key, batch, variables['batch_stats'])

    @jit
    def val_step(params, batch_stats, batch, labels):
        variables = {'params': params, 'batch_stats': batch_stats}
        logits = model.apply(variables, batch, train=False)  # stack the model's forward pass with the logits function
        return accuracy(logits, labels), optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, Config["NUM_LABELS"])).mean()

    # control randomness on dropout and update inside train_step
    rng = jax.random.PRNGKey(0)

    print("len(train_dataset):", len(train_dataset), "len(test_dataset):", len(test_dataset))

    start_time = 1e10
    for epoch_i in tqdm(range(Config['N_EPOCHS']), desc=f"{Config['N_EPOCHS']} epochs", position=0, leave=True):
        # training set
        train_loss, train_accuracy = [], []
        iter_n = len(train_dataset)

        with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:

            epoch_start = timer()

            for batch_i, _batch in enumerate(train_dataset):
                batch = _batch[0]  # train_dataset is tuple containing (image,labels)
                labels = _batch[1]
                if epoch_i == 0 and batch_i == 1:
                    start_time = timer()

                batch = jnp.array(batch, dtype=jnp.float32)
                labels = jnp.array(labels, dtype=jnp.float32)

                # backprop and update param
                rng, key = jax.random.split(rng)
                params, opt_state, batch_stats, stats = optimizer.step(params, opt_state, key, batch=(batch, labels),
                                                                       func_state=batch_stats, global_step_int=epoch_i * iter_n + batch_i,
                                                                       learning_rate=conf["learning_rate"], momentum=conf["momentum"])
                _train_loss = stats['loss']
                _train_top1_acc = stats["aux"]

                # update train statistics
                train_loss.append(_train_loss)
                train_accuracy.append(_train_top1_acc)
                progress_bar.update(1)

        epoch_end = timer()
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        print(f"[{epoch_i + 1}/{Config['N_EPOCHS']}] Train Loss: {avg_train_loss:.03} | Train Accuracy: {avg_train_acc:.03}")

        # validation set
        valid_accuracy = []
        valid_loss = []
        iter_n = len(test_dataset)
        with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:
            for _batch in test_dataset:
                batch = _batch[0]
                labels = _batch[1]

                batch = jnp.array(batch, dtype=jnp.float32)
                labels = jnp.array(labels, dtype=jnp.float32)

                acc, loss = val_step(params, batch_stats, batch, labels)
                valid_accuracy.append(acc)
                valid_loss.append(loss)
                progress_bar.update(1)

        avg_valid_acc = sum(valid_accuracy) / len(valid_accuracy)
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"[{epoch_i + 1}/{Config['N_EPOCHS']}] Valid Accuracy: {avg_valid_acc:.03} Valid Loss: {avg_valid_loss:.03}")

        with open(train_stats_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch_i, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, epoch_end - epoch_start,
                 np.maximum(0, timer() - start_time)])

        # Early stopping if loss_value is inf or None or just too high
        if jnp.isnan(avg_train_loss) or not jnp.isfinite(avg_train_loss) or avg_train_loss > 100:
            print("Early stopping with train loss", avg_train_loss)
            break


if __name__ == "__main__":
    # None learning_rate indicates the optimizer to set use_adaptive_learning_rate to True and
    # None momentum to set use_adaptive_momentum to True
    learning_rates = [None, 1e-3, 1e-2, 1e-1]
    momentums = [None, 0.1, 0.5, 0.9]

    # min_loss: 0.5556641 best_learning_rate: 0.001 best_momentum: 0.1
    for learning_rate in learning_rates:
        for momentum in momentums:
            train_transfer_learning('kfac', learning_rate, momentum)
