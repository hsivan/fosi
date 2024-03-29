# Based on https://github.com/deepmind/dm-haiku/blob/main/examples/rnn/train.py :
# character-level language modelling with a recurrent network in JAX.

import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
from timeit import default_timer as timer

import tensorflow_datasets as tfds
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import jaxopt

from experiments.dnn.dnn_test_utils import get_config, start_test, write_config_to_file
from experiments.dnn.rnn_shakespeare import sample, TinyShakespeareDataset, make_network, TRAIN_BATCH_SIZE, \
    SEQUENCE_LENGTH, EVAL_BATCH_SIZE, SEED, TRAINING_STEPS, TrainingState, SAMPLING_INTERVAL, SAMPLE_LENGTH, \
    EVALUATION_INTERVAL


def sequence_loss(batch: TinyShakespeareDataset.Batch) -> jnp.ndarray:
    """Unrolls the network over a sequence of inputs & targets, gets loss."""
    # Note: this function is impure; we hk.transform() it below.
    core = make_network()
    sequence_length, batch_size = batch['input'].shape
    initial_state = core.initial_state(batch_size)
    logits, _ = hk.dynamic_unroll(core, batch['input'], initial_state)
    one_hot_labels = jax.nn.one_hot(batch['target'], num_classes=logits.shape[-1])
    kfac_jax.register_softmax_cross_entropy_loss(logits, one_hot_labels)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot_labels * log_probs) / (sequence_length * batch_size)


def main(optimizer_name, learning_rate, momentum):
    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=TRAIN_BATCH_SIZE, momentum=momentum,
                      learning_rate=learning_rate, num_iterations_between_ese=800, approx_l=0, alpha=0.01)
    test_folder = start_test(conf["optimizer"], test_folder='test_results_rnn_shakespeare')
    write_config_to_file(test_folder, conf)

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "latency", "wall_time"])
    start_time = 1e10

    # Make training dataset.
    train_data = TinyShakespeareDataset.load(
        tfds.Split.TRAIN,
        batch_size=TRAIN_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH)

    # Make evaluation dataset(s).
    eval_data = {  # pylint: disable=g-complex-comprehension
        split: TinyShakespeareDataset.load(
            split,
            batch_size=EVAL_BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH)
        for split in [tfds.Split.TRAIN, tfds.Split.TEST]
    }

    # Make loss, sampler, and optimizer.
    params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    _, sample_fn = hk.without_apply_rng(hk.transform(sample))

    sample_fn = jax.jit(sample_fn, static_argnums=[3])

    # Initialize training state.
    rng = hk.PRNGSequence(SEED)
    net_params = params_init(next(rng), next(train_data))

    if conf["optimizer"] == "kfac":
        # Throws exception "The following parameter indices were not assigned a block: {1, 3, 5, 7}.". When using
        # hk.static_unroll instead of hk.dynamic_unroll, the exception is that layers where assigned multiple tags.
        optimizer = kfac_jax.Optimizer(
            value_and_grad_func=jax.value_and_grad(loss_fn),
            l2_reg=0.0,
            value_func_has_aux=False,
            value_func_has_state=False,
            value_func_has_rng=False,
            use_adaptive_learning_rate=False,
            use_adaptive_momentum=False,
            use_adaptive_damping=True,
            initial_damping=1.0,
            multi_device=False,
        )
        rng = jax.random.PRNGKey(42)
        rng, key = jax.random.split(rng)
        opt_state = optimizer.init(net_params, key, next(train_data))
    else:  # lbfgs
        # Diverges with history_size=10, therefore use 20
        optimizer = jaxopt.LBFGS(fun=jax.value_and_grad(loss_fn), value_and_grad=True, jit=True, stepsize=learning_rate, history_size=momentum)
        opt_state = optimizer.init_state(net_params, next(train_data))

    epoch_start = timer()

    # Training loop.
    for step in range(TRAINING_STEPS + 1):

        # Do a batch of SGD.
        train_batch = next(train_data)

        if conf["optimizer"] == "kfac":
            rng, key = jax.random.split(rng)
            net_params, opt_state, stats = optimizer.step(net_params, opt_state, key, batch=train_batch,
                                                          global_step_int=step,
                                                          learning_rate=conf["learning_rate"], momentum=conf["momentum"])
        else:  # lbfgs
            net_params, opt_state = jax.jit(optimizer.update)(net_params, opt_state, train_batch)

        state = TrainingState(params=net_params, opt_state=opt_state)

        # Periodically generate samples.
        if step % SAMPLING_INTERVAL == 0:
            context = train_batch['input'][:, 0]  # First element of training batch.
            assert context.ndim == 1
            rng_key = next(rng)
            samples = sample_fn(state.params, rng_key, context, SAMPLE_LENGTH)

            prompt = TinyShakespeareDataset.decode(context)
            continuation = TinyShakespeareDataset.decode(samples)

            print('Prompt:', prompt)
            print('Continuation:', continuation)

        # Periodically evaluate training and test loss.
        if step % EVALUATION_INTERVAL == 0:
            epoch_end = timer()
            train_loss = 0.0
            valid_loss = 0.0
            for b_train, (split, ds) in enumerate(eval_data.items()):
                eval_batch = next(ds)
                loss = loss_fn(state.params, eval_batch)
                print({
                    'step': step,
                    'loss': float(loss),
                    'split': split,
                })
                if b_train == 0:
                    train_loss = float(loss)
                else:
                    valid_loss = float(loss)

            with open(train_stats_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [step, train_loss, valid_loss, epoch_end - epoch_start, np.maximum(0, timer() - start_time)])

            if step == EVALUATION_INTERVAL:
                start_time = timer()
            epoch_start = timer()


if __name__ == '__main__':
    history_sizes = [10, 20, 40, 80, 100]

    try:
        main('kfac', None, None)
    except Exception as e:
        # TODO: could not get K-FAC to work for RNN.
        print(e)

    # 0 learning rate means using line search
    for learning_rate in [0]:
        for history_size in history_sizes:  # We use momentum as a placeholder
            main('lbfgs', learning_rate, history_size)
