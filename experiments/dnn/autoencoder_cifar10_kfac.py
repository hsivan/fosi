# Based on: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html

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
import kfac_jax

from experiments.dnn.dnn_test_utils import start_test, get_config, write_config_to_file
from experiments.dnn.autoencoder_cifar10 import Autoencoder, val_loader, train_loader, test_loader


class TrainerModule:

    def __init__(self, c_hid, latent_dim, conf, seed=42):
        super().__init__()
        self.num_epochs = 200
        self.c_hid = c_hid
        self.latent_dim = latent_dim
        self.conf = conf
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = Autoencoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        # Prepare logging
        self.exmp_imgs = next(iter(val_loader))[0][:8]
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Loss function: MSE reconstruction loss
        def loss_fn(params, batch):
            imgs, _ = batch
            recon_imgs = self.model.apply({'params': params}, imgs)
            kfac_jax.register_squared_error_loss(recon_imgs, imgs)
            loss = ((recon_imgs - imgs) ** 2).mean(axis=0).sum()  # Mean over batch, sum over pixels
            return loss

        self.loss_fn = loss_fn  # jitted by the K-FAC library
        self.jitted_loss_fn = jax.jit(loss_fn)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp_imgs)['params']

        # Initialize learning rate schedule and optimizer
        print("len(train_loader)", len(train_loader))

        # Use adaptive_learning_rate and adaptive_momentum
        optimizer = kfac_jax.Optimizer(
            value_and_grad_func=jax.value_and_grad(self.loss_fn),
            l2_reg=0.0,
            value_func_has_aux=False,
            value_func_has_state=False,
            use_adaptive_learning_rate=True if self.conf["learning_rate"] is None else False,
            use_adaptive_momentum=True if self.conf["momentum"] is None else False,
            use_adaptive_damping=True,
            initial_damping=1.0,
            multi_device=False
        )
        self.optimizer = optimizer
        rng = jax.random.PRNGKey(42)
        rng, key = jax.random.split(rng)
        self.opt_state = optimizer.init(params, key, next(iter(train_loader)))
        self.params = params

    def train_model(self):
        # Train model for defined number of epochs
        best_eval = 1e6
        start_time = 1e10
        for epoch_idx in tqdm(range(1, self.num_epochs + 1)):
            if epoch_idx == 1:
                start_time = timer()
            epoch_start = timer()
            train_loss = self.train_epoch(epoch=epoch_idx)
            epoch_end = timer()
            eval_loss = 0.0
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_loader)
                if eval_loss < best_eval:
                    best_eval = eval_loss
                print("eval_loss", eval_loss)
            with open(train_stats_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch_idx, train_loss, eval_loss, epoch_end - epoch_start, np.maximum(0, timer() - start_time)])

            # Early stopping if loss_value is inf or None or just too high
            if np.isnan(train_loss) or not np.isfinite(train_loss) or train_loss > 1000:
                print("Early stopping with train loss", train_loss)
                break

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        num_train_batches = len(train_loader)
        rng = jax.random.PRNGKey(42)

        for step, batch in enumerate(train_loader):
            rng, key = jax.random.split(rng)
            self.params, self.opt_state, stats = self.optimizer.step(self.params, self.opt_state, key, batch=batch,
                                                                     global_step_int=epoch * num_train_batches + step,
                                                                     learning_rate=self.conf["learning_rate"], momentum=self.conf["momentum"])
            loss = stats['loss']
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        return avg_loss

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.jitted_loss_fn(self.params, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss


def train_cifar(latent_dim, conf=None):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(c_hid=32, latent_dim=latent_dim, conf=conf)
    trainer.train_model()
    test_loss = trainer.eval_model(test_loader)
    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({'params': trainer.params})
    return trainer, test_loss


if __name__ == "__main__":
    # None learning_rate indicates the optimizer to set use_adaptive_learning_rate to True and
    # None momentum to set use_adaptive_momentum to True
    learning_rates = [None, 1e-3, 1e-2, 1e-1]
    momentums = [None, 0.1, 0.5, 0.9]

    # min_loss: 49.601337 best_learning_rate: None best_momentum: None
    for learning_rate in learning_rates:
        for momentum in momentums:
            conf = get_config(optimizer='kfac', approx_k=10, batch_size=256, learning_rate=learning_rate, momentum=momentum,
                              num_iterations_between_ese=800, approx_l=0, alpha=0.01, learning_rate_clip=1.0)
            test_folder = start_test(conf["optimizer"], test_folder='test_results_autoencoder_cifar10')
            write_config_to_file(test_folder, conf)

            train_stats_file = test_folder + "/train_stats.csv"
            with open(train_stats_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "latency", "wall_time"])

            _, _ = train_cifar(128, conf=conf)
