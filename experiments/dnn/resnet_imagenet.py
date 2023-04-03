"""Trains a ResNet-50 on the ImageNet dataset.
Based on: https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/examples/imagenet/train.py
"""

import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
import functools
from typing import Any
from timeit import default_timer as timer
import resource
from flax import jax_utils

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
import jax
import jax.numpy as jnp
from jax import random, lax
from jax.flatten_util import ravel_pytree
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from experiments.dnn import input_pipeline
from experiments.dnn import resnet_models
from experiments.dnn.dnn_test_utils import start_test, write_config_to_file, get_config, get_optimizer

# Hide any GPUs from TensorFlow. Otherwise, TF might reserve memory and make it unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')
print('JAX local devices: %r', jax.local_devices())

NUM_CLASSES = 1000
b_parallel = True


def create_model(*, model_cls, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    if b_parallel:
        metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=int(config.warmup_epochs * steps_per_epoch))
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=int(cosine_epochs * steps_per_epoch))
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[int(config.warmup_epochs * steps_per_epoch)])
    return schedule_fn


def loss_fn(params, batch, state):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['image'],
                                             mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_leaves(params)  # Change to jax.tree_util.tree_leaves
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)


def train_step(state, batch, learning_rate_fn):
    """Perform a single training step."""

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params, batch, state)
        if b_parallel:
            # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
            grads = lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(
        variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'])


# Relevant only for parallel training
def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
    ds = input_pipeline.create_split(
        dataset_builder, batch_size, image_size=image_size, dtype=dtype, train=train, cache=cache)
    if b_parallel:
        it = map(prepare_tf_data, ds)
        it = jax_utils.prefetch_to_device(it, 2)
    else:
        ds = tfds.as_numpy(ds)
        it = iter(ds)
    return it


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: DynamicScale


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        state = jax.device_get(state)
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


# Relevant only for parallel training.
# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    if b_parallel:
        # Each device has its own version of the running average batch statistics, and we sync them before evaluation.
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    else:
        return state.replace(batch_stats=state.batch_stats)


def create_train_state(rng, conf, model, image_size, learning_rate_fn, batch, half_precision):
    """Create initial training state."""
    platform = jax.local_devices()[0].platform
    if half_precision and platform == 'gpu':
        dynamic_scale = DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_size, model)
    '''tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=conf["momentum"],
        nesterov=True,
    )'''

    temp_state = ml_collections.ConfigDict()
    temp_state.apply_fn, temp_state.batch_stats = model.apply, batch_stats
    loss_f = lambda params, batch: loss_fn(params, batch, temp_state)[0]
    conf["learning_rate"] = learning_rate_fn
    conf["alpha"] = learning_rate_fn
    tx = get_optimizer(conf, loss_f, batch, b_call_ese_internally=False, b_parallel=b_parallel)

    num_params = ravel_pytree(params)[0].shape[0]
    print("num_params:", num_params)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=dynamic_scale)
    return state


def update_ese(state, batch):
    opt_state = state.tx.update_ese(state.params, state.opt_state, batch)
    new_state = state.replace(opt_state=opt_state)
    return new_state


def train_and_evaluate(config: ml_collections.ConfigDict, conf, test_folder, workdir: str) -> TrainState:
    """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    Final TrainState.
  """

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    rng = random.PRNGKey(0)

    image_size = 224

    if b_parallel:
        if config.batch_size % jax.device_count() > 0:
            raise ValueError('Batch size must be divisible by the number of devices')
        local_batch_size = config.batch_size // jax.process_count()
    else:
        local_batch_size = config.batch_size

    platform = jax.local_devices()[0].platform

    if config.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    if 'imagenet2012' in config.dataset or 'imagenette' in config.dataset:
        dataset_builder = tfds.builder(config.dataset)
        dataset_builder.download_and_prepare()
    else:  # load from local directory /datasets/ImageNet/
        dataset_builder = tfds.ImageFolder(config.dataset)
    print(dataset_builder.info)
    train_iter = create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=True,
        cache=config.cache)
    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=False,
        cache=config.cache)

    steps_per_epoch = (
            dataset_builder.info.splits['train'].num_examples // config.batch_size
    )

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits['val'].num_examples if 'image_folder' in dataset_builder.info.name else dataset_builder.info.splits['validation'].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    print("steps_per_epoch:", steps_per_epoch, "num_steps:", num_steps, "steps_per_eval:", steps_per_eval,
          "steps_per_checkpoint:", steps_per_checkpoint)

    base_learning_rate = config.learning_rate * config.batch_size / 256.

    model_cls = getattr(resnet_models, config.model)
    model = create_model(
        model_cls=model_cls, half_precision=config.half_precision)

    learning_rate_fn = create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch)

    batch_ = next(train_iter)
    state = create_train_state(rng, conf, model, image_size, learning_rate_fn, batch_, config.half_precision)
    # state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    if b_parallel:
        state = jax_utils.replicate(state)
        p_train_step = jax.pmap(functools.partial(train_step, learning_rate_fn=learning_rate_fn), axis_name='batch')
        p_eval_step = jax.pmap(eval_step, axis_name='batch')
        p_update_ese = jax.pmap(update_ese, axis_name='batch')
    else:
        p_train_step = jax.jit(functools.partial(train_step, learning_rate_fn=learning_rate_fn))
        p_eval_step = jax.jit(eval_step)
        p_update_ese = jax.jit(update_ese)

    train_metrics = []
    print('Initial compilation, this might take some minutes...')
    # for step, batch in zip(range(step_offset, num_steps), train_iter):
    print("step_offset:", step_offset, "num_steps:", num_steps)
    start_time = epoch_start = 1e10
    for step in range(step_offset, num_steps):
        batch = next(train_iter)

        if "fosi" in conf["optimizer"] and max(1, step + 1 - conf["num_warmup_iterations"]) % conf["num_iterations_between_ese"] == 0:
            state = p_update_ese(state, batch_)

        state, metrics = p_train_step(state, batch)
        if step == step_offset:
            print('Initial compilation completed.')

        train_metrics.append(metrics)

        if (step + 1) % config.log_every_steps == 0:
            # Reset train_metrics more than once per epoch, to have a better representation of current metrics
            train_metrics = []

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []
            epoch_end = timer()

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            if b_parallel:
                eval_metrics = common_utils.get_metrics(eval_metrics)
            else:
                eval_metrics = jax.device_get(eval_metrics)
                eval_metrics = common_utils.stack_forest(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            print('eval epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, summary['loss'], summary['accuracy'] * 100))

            if b_parallel:
                tm = common_utils.get_metrics(train_metrics)
            else:
                tm = jax.device_get(train_metrics)
                tm = common_utils.stack_forest(tm)
            train_summary = jax.tree_map(lambda x: x.mean(), tm)

            with open(train_stats_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [epoch, train_summary['loss'], train_summary['accuracy'], summary['loss'], summary['accuracy'],
                     jnp.maximum(0, epoch_end - epoch_start), jnp.maximum(0, timer() - start_time)])

            if epoch == 0:
                start_time = timer()
            epoch_start = timer()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            # save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state


if __name__ == '__main__':
    config = ml_collections.ConfigDict()
    config.model = 'ResNet50'
    config.dataset = '/datasets/ImageNet/'  # '/datasets/ImageNet/' (local directory)  or 'imagenet2012:5.*.*' or 'imagenette'
    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.batch_size = 128

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly, for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    conf = get_config(optimizer="fosi_momentum", approx_k=10, batch_size=config.batch_size, momentum=config.momentum,
                      learning_rate=config.learning_rate, num_iterations_between_ese=800, approx_l=0, alpha=0.01,
                      num_epochs=config.num_epochs)
    test_folder = start_test(conf["optimizer"], test_folder='test_results_resnet_imagenet')
    write_config_to_file(test_folder, conf)

    train_and_evaluate(config, conf, test_folder, './test_results_resnet_imagenet')
