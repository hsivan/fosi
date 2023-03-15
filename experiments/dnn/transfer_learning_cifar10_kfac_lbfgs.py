# Based on https://www.kaggle.com/code/yashvi/transfer-learning-using-jax-flax/notebook

import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
# See: https://github.com/google/jax/issues/8178
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

import csv
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax
import kfac_jax
import jaxopt

from experiments.dnn.dnn_test_utils import get_config, start_test, write_config_to_file
from experiments.dnn.transfer_learning_cifar10 import get_model_and_variables, Config, train_dataset, accuracy, \
    model, variables, test_dataset, val_step


def loss_fn(params, batch_stats, batch):
    variables_ = {'params': {'backbone': variables['params']['backbone'], 'head': params['head']},
                  'batch_stats': batch_stats}
    inputs, targets = batch
    logits, new_batch_stats = model.apply(variables_, inputs, mutable=['batch_stats'])
    # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
    one_hot = jax.nn.one_hot(targets, Config["NUM_LABELS"])
    kfac_jax.register_softmax_cross_entropy_loss(logits, one_hot)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = accuracy(logits, targets)
    return loss, (new_batch_stats['batch_stats'], acc)


def train_transfer_learning(optimizer_name, learning_rate, momentum):
    # Reset the variables
    _, variables = get_model_and_variables('resnet18', 0)

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

    if conf["optimizer"] == "kfac":
        optimizer = kfac_jax.Optimizer(
            value_and_grad_func=jax.value_and_grad(loss_fn, has_aux=True),
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
    else:  # lbfgs
        # Diverges with history_size=10, therefore use 20
        optimizer = jaxopt.LBFGS(fun=jax.value_and_grad(loss_fn, has_aux=True), value_and_grad=True, has_aux=True, jit=True, stepsize=learning_rate, history_size=momentum)
        opt_state = optimizer.init_state(variables['params'], variables['batch_stats'], batch)

    params = variables['params']
    batch_stats = variables['batch_stats']

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

                # batch is a tuple containing (image,labels)
                batch_ = (jnp.array(batch[0], dtype=jnp.float32), jnp.array(batch[1], dtype=jnp.float32))

                # backprop and update param & batch stats
                if conf["optimizer"] == "kfac":
                    rng, key = jax.random.split(rng)
                    params, opt_state, batch_stats, stats = optimizer.step(params, opt_state, key, batch=batch_,
                                                                           func_state=batch_stats, global_step_int=epoch_i * iter_n + batch_i,
                                                                           learning_rate=conf["learning_rate"], momentum=conf["momentum"])
                    # update train statistics
                    train_loss.append(stats['loss'])
                    train_accuracy.append(stats["aux"])
                else:  # lbfgs
                    net_params, opt_state = jax.jit(optimizer.update)(params, opt_state, batch_stats, batch_)
                    batch_stats, acc = opt_state.aux
                    train_loss.append(opt_state.value)
                    train_accuracy.append(acc)
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
            for batch in test_dataset:
                batch_ = (jnp.array(batch[0], dtype=jnp.float32), jnp.array(batch[1], dtype=jnp.float32))

                acc, loss = val_step(params, batch_stats, batch_)
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
    # None learning_rate indicates K-FAC to set use_adaptive_learning_rate to True and
    # None momentum to set use_adaptive_momentum to True
    learning_rates = [None, 1e-3, 1e-2, 1e-1]
    momentums = [None, 0.1, 0.5, 0.9]
    history_sizes = [10, 20, 40, 80, 100]

    # min_loss: 0.5556641 best_learning_rate: 0.001 best_momentum: 0.1
    for learning_rate in learning_rates:
        for momentum in momentums:
            train_transfer_learning('kfac', learning_rate, momentum)

    # 0 learning rate means using line search
    for learning_rate in [0]:
        for history_size in history_sizes:  # We use momentum as a placeholder
            train_transfer_learning('lbfgs', learning_rate, history_size)
