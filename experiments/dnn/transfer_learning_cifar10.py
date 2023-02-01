# Based on https://www.kaggle.com/code/yashvi/transfer-learning-using-jax-flax/notebook

import csv
import os
import numpy as np
from typing import Callable
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
from jax.config import config
from jax_resnet import pretrained_resnet, slice_variables, Sequential
import flax
from flax.training import train_state
from flax import linen as nn
from flax.core import FrozenDict, frozen_dict
import optax

from fosi.fosi_optimizer import fosi_momentum, fosi_adam
from experiments.utils.test_utils import get_config, start_test, write_config_to_file

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.config.experimental.set_visible_devices([], "GPU")


def train_transfer_learning(optimizer_name):
    config.update("jax_enable_x64", False)

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

    class TrainState(train_state.TrainState):
        batch_stats: FrozenDict
        loss_fn: Callable = flax.struct.field(pytree_node=False)
        eval_fn: Callable = flax.struct.field(pytree_node=False)

    def zero_grads():
        """
        Zero out the previous gradient computation
        """

        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()

        return optax.GradientTransformation(init_fn, update_fn)

    def create_mask(params, label_fn):
        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = 'zero'
                else:
                    if isinstance(params[k], FrozenDict):
                        mask[k] = {}
                        _map(params[k], mask[k], label_fn)
                    else:
                        mask[k] = 'adam'

        mask = {}
        _map(params, mask, label_fn)
        return frozen_dict.freeze(mask)

    def loss_f(params, batch_data):
        variables = {'params': {'backbone': state.params['backbone'], 'head': params['head']},
                     'batch_stats': state.batch_stats}
        inputs, targets = batch_data
        logits, new_batch_stats = state.apply_fn(variables, inputs, train=True, mutable=['batch_stats'],
                                                 rngs={'dropout': jax.random.PRNGKey(0)})
        # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
        one_hot = jax.nn.one_hot(targets, Config["NUM_LABELS"])
        loss = state.loss_fn(logits, one_hot).mean()
        return loss

    # Momentum with 1e-3 obtains 0.5758 train loss (after 20 epochs), while 1e-2 obtains 0.6037. Therefore, using 1e-3.
    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=Config['BATCH_SIZE'],
                      learning_rate=Config['LR'], num_iterations_between_ese=800, approx_l=0,
                      num_warmup_iterations=len(train_dataset), alpha=0.01)

    test_folder = start_test(conf["optimizer"], test_folder='test_results_transfer_learning')
    write_config_to_file(test_folder, conf)

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    batch = next(iter(train_dataset))
    batch = (jnp.array(batch[0], dtype=jnp.float32), jnp.array(batch[1], dtype=jnp.float32))

    print("Number of frozen parameters:", ravel_pytree(variables['params']['backbone'])[0].shape[0])

    if conf['optimizer'] == 'my_momentum':
        optim = fosi_momentum(optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False), loss_f, batch,
                              decay=conf["momentum"],
                              num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                              approx_k=conf["approx_k"],
                              approx_l=conf["approx_l"], warmup_w=conf["num_warmup_iterations"],
                              alpha=conf["alpha"], learning_rate_clip=3.0)
    elif conf['optimizer'] == 'my_adam':
        optim = fosi_adam(optax.adam(conf["learning_rate"]), loss_f, batch,
                          decay=conf["momentum"],
                          num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                          approx_k=conf["approx_k"],
                          approx_l=conf["approx_l"], warmup_w=conf["num_warmup_iterations"],
                          alpha=conf["alpha"])
    elif conf['optimizer'] == 'momentum':
        optim = optax.sgd(learning_rate=conf['learning_rate'], momentum=conf['momentum'], nesterov=False)
    elif conf['optimizer'] == 'adam':
        optim = optax.adam(learning_rate=conf['learning_rate'])
    else:
        raise "Illegal optimizer " + conf["optimizer"]

    optimizer = optax.multi_transform(
        {'adam': optim, 'zero': zero_grads()},
        create_mask(variables['params'], lambda s: s.startswith('backbone'))
    )

    def accuracy(logits, labels):
        """
        calculates accuracy based on logits and labels
        INPUT logits , labels
        RETURNS accuracy
        """
        return [jnp.mean(jnp.argmax(logits, -1) == labels)]

    loss_fn = optax.softmax_cross_entropy
    eval_fn = accuracy

    # Instantiate a TrainState.
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables['batch_stats'],
        loss_fn=loss_fn,
        eval_fn=eval_fn
    )

    @jit
    def train_step(state: TrainState, batch, labels, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        # params as input because we differentiate wrt it
        def loss_function(params):
            # if you set state.params, then params can't be backpropagated through!
            variables = {'params': params, 'batch_stats': state.batch_stats}

            # return mutated states if mutable is specified
            logits, new_batch_stats = state.apply_fn(
                variables, batch, train=True,
                mutable=['batch_stats'],
                rngs={'dropout': dropout_rng}
            )
            # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
            one_hot = jax.nn.one_hot(labels, Config["NUM_LABELS"])
            loss = state.loss_fn(logits, one_hot).mean()
            return loss, (logits, new_batch_stats)

        # backpropagation and update params & batch_stats
        grad_fn = jax.value_and_grad(loss_function, has_aux=True)  # differentiate the loss function
        (loss, aux), grads = grad_fn(state.params)
        logits, new_batch_stats = aux
        # grads = lax.pmean(grads, axis_name='batch') #compute the mean gradient over all devices
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_batch_stats['batch_stats']  # applies the gradients to the weights.
        )

        # evaluation metrics
        accuracy = state.eval_fn(logits, labels)

        # store metadata
        '''metadata = jax.lax.pmean(
            {'loss': loss, 'accuracy': accuracy},
            axis_name='batch'
        )'''
        metadata = {'loss': loss, 'accuracy': accuracy}
        return new_state, metadata, new_dropout_rng

    @jit
    def val_step(state: TrainState, batch, labels):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logits = state.apply_fn(variables, batch,
                                train=False)  # stack the model's forward pass with the logits function
        return state.eval_fn(logits, labels), state.loss_fn(logits, jax.nn.one_hot(labels, Config["NUM_LABELS"])).mean()

    def tes_step(state: TrainState, batch):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logits = state.apply_fn(variables, batch,
                                train=False)  # stack the model's forward pass with the logits function
        return logits

    parallel_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))
    parallel_val_step = jax.pmap(val_step, axis_name='batch', donate_argnums=(0,))
    parallel_tes_step = jax.pmap(tes_step, axis_name='batch', donate_argnums=(0,))

    # required for parallelism
    # state = replicate(state)

    # control randomness on dropout and update inside train_step
    rng = jax.random.PRNGKey(0)
    # dropout_rng = jax.random.split(rng, jax.local_device_count())  # for parallelism
    dropout_rng = rng

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

                # batch, labels = shard(batch), shard(labels)

                # backprop and update param & batch statsp

                # state, train_metadata, dropout_rng = parallel_train_step(state, batch, labels, dropout_rng)
                state, train_metadata, dropout_rng = train_step(state, batch, labels, dropout_rng)
                # train_metadata = unreplicate(train_metadata)

                # update train statistics
                _train_loss, _train_top1_acc = map(float, [train_metadata['loss'], *train_metadata['accuracy']])
                train_loss.append(_train_loss)
                train_accuracy.append(_train_top1_acc)
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
            for _batch in test_dataset:
                batch = _batch[0]
                labels = _batch[1]

                batch = jnp.array(batch, dtype=jnp.float32)
                labels = jnp.array(labels, dtype=jnp.float32)

                # batch, labels = shard(batch), shard(labels)
                # metric = parallel_val_step(state, batch, labels)[0]
                acc, loss = val_step(state, batch, labels)
                valid_accuracy.append(acc[0])
                valid_loss.append(loss)
                progress_bar.update(1)

        avg_valid_acc = sum(valid_accuracy) / len(valid_accuracy)
        # avg_valid_acc = np.array(avg_valid_acc)[0]
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"[{epoch_i + 1}/{Config['N_EPOCHS']}] Valid Accuracy: {avg_valid_acc:.03} Valid Loss: {avg_valid_loss:.03}")

        with open(train_stats_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch_i, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, epoch_end - epoch_start,
                 np.maximum(0, timer() - start_time)])


if __name__ == "__main__":
    for optimizer_name in ['my_adam', 'my_momentum', 'adam', 'momentum']:
        train_transfer_learning(optimizer_name)
