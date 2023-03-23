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
from flax.training import train_state
from flax import linen as nn
from flax.core import FrozenDict
import optax

from experiments.dnn.dnn_test_utils import get_config, start_test, write_config_to_file, get_optimizer

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.config.experimental.set_visible_devices([], "GPU")


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

    return tfds.as_numpy(train_dataset), tfds.as_numpy(test_dataset)


class Head(nn.Module):
    """ head model """
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(features=Config["NUM_LABELS"])(inputs)
        return x


class Model(nn.Module):
    """ Combines backbone and head model """
    backbone: Sequential
    head: Head

    def __call__(self, inputs):
        x = self.backbone(inputs)
        # average pool layer
        x = jnp.mean(x, axis=(1, 2))
        x = self.head(x)
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
    head_params = head.init(key, head_inputs)

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


def accuracy(logits, labels):
    """
    calculates accuracy based on logits and labels
    INPUT logits , labels
    RETURNS accuracy
    """
    return jnp.mean(jnp.argmax(logits, -1) == labels)


train_dataset, test_dataset = load_datasets()
num_train_steps = len(train_dataset)
print("len(train_dataset):", len(train_dataset), "len(test_dataset):", len(test_dataset))


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict


@jit
def loss_fn(params, batch, batch_stats):
    inputs, targets = batch
    # Fix backbone with the backbone in variables['params']['backbone'] and use the given batch_stats (which is getting updated).
    # Only the head (params) is trained (only gradient w.r.t params are evaluated).
    variables_ = {'params': {'backbone': variables['params']['backbone'], 'head': params}, 'batch_stats': batch_stats}

    # return mutated states if mutable is specified
    logits, new_batch_stats = model.apply(variables_, inputs, mutable=['batch_stats'])
    # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
    one_hot = jax.nn.one_hot(targets, Config["NUM_LABELS"])
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss, (logits, new_batch_stats)


@jit
def train_step(state: TrainState, batch):
    inputs, targets = batch

    # backpropagation and update params & batch_stats
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # differentiate the loss function
    (loss, aux), grads = grad_fn(state.params, batch, state.batch_stats)
    logits, new_batch_stats = aux
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_batch_stats['batch_stats']  # applies the gradients to the weights.
    )

    # evaluation metrics
    acc = accuracy(logits, targets)

    # store metadata
    metadata = {'loss': loss, 'accuracy': acc}
    return new_state, metadata

@jit
def val_step(params, batch_stats, batch):
    inputs, targets = batch
    variables_ = {'params': {'backbone': variables['params']['backbone'], 'head': params}, 'batch_stats': batch_stats}
    logits = model.apply(variables_, inputs)  # stack the model's forward pass with the logits function
    return accuracy(logits, targets), optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, Config["NUM_LABELS"])).mean()


def train_transfer_learning(optimizer_name):
    # Reset the variables
    _, variables = get_model_and_variables('resnet18', 0)

    # Momentum with 1e-3 obtains 0.5758 train loss (after 20 epochs), while 1e-2 obtains 0.6037. Therefore, using 1e-3.
    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=Config['BATCH_SIZE'],
                      learning_rate=Config['LR'], num_iterations_between_ese=800, approx_l=0,
                      num_warmup_iterations=len(train_dataset), alpha=0.01)

    test_folder = start_test(conf["optimizer"], test_folder='test_results_transfer_learning_cifar10')
    write_config_to_file(test_folder, conf)

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    batch = next(iter(train_dataset))

    print("Number of frozen parameters:", ravel_pytree(variables['params']['backbone'])[0].shape[0])

    loss_f = lambda params, batch: loss_fn(params, batch, variables['batch_stats'])[0]
    optimizer = get_optimizer(conf, loss_f, batch)

    # Instantiate a TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params']['head'],
        tx=optimizer,
        batch_stats=variables['batch_stats']
    )

    start_time = 1e10
    for epoch_i in tqdm(range(Config['N_EPOCHS']), desc=f"{Config['N_EPOCHS']} epochs", position=0, leave=True):
        # training set
        train_loss, train_accuracy = [], []
        iter_n = len(train_dataset)

        with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:

            epoch_start = timer()

            for batch_i, batch in enumerate(train_dataset):
                if epoch_i == 0 and batch_i == 1:
                    start_time = timer()

                # backprop and update param & batch stats
                state, train_metadata = train_step(state, batch)

                # update train statistics
                train_loss.append(train_metadata['loss'])
                train_accuracy.append(train_metadata['accuracy'])
                progress_bar.update(1)

        epoch_end = timer()
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        print(
            f"[{epoch_i + 1}/{Config['N_EPOCHS']}] Train Loss: {avg_train_loss:.03} | Train Accuracy: {avg_train_acc:.03}")

        # validation set
        valid_accuracy = []
        valid_loss = []
        iter_n = len(test_dataset)
        with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:
            for batch in test_dataset:
                acc, loss = val_step(state.params, state.batch_stats, batch)
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


if __name__ == "__main__":
    for optimizer_name in ['fosi_adam', 'fosi_momentum', 'adam', 'momentum']:
        train_transfer_learning(optimizer_name)
