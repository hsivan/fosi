# Based on: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html

import csv
import os
import numpy as np
from jax.config import config
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
import torch
from torch.utils.tensorboard import SummaryWriter
import jax
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

from fosi import fosi_adam, fosi_momentum
from experiments.utils.test_utils import start_test, get_config, write_config_to_file

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/tutorial9_jax"

print("Device:", jax.devices()[0])

torch.manual_seed(0)


# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2. - 1.
    return img


# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=image_to_numpy, download=True)
train_set, val_set = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=image_to_numpy, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4,
                               collate_fn=numpy_collate, persistent_workers=True)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4,
                             collate_fn=numpy_collate)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4,
                              collate_fn=numpy_collate)


class Encoder(nn.Module):
    c_hid: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 32x32 => 16x16
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 16x16 => 8x8
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 8x8 => 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)
        return x


# Test encoder implementation.
rng = random.PRNGKey(0)
# Example images as input
imgs = next(iter(train_loader))[0]
# Create encoder
encoder = Encoder(c_hid=32, latent_dim=128)
# Initialize parameters of encoder with random key and images
params = encoder.init(rng, imgs)['params']
# Apply encoder with parameters on the images
out = encoder.apply({'params': params}, imgs)
print(out.shape)

del out, encoder, params


class Decoder(nn.Module):
    c_out: int
    c_hid: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2 * 16 * self.c_hid)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.tanh(x)
        return x


# Test decoder implementation.
rng = random.PRNGKey(0)
# Example latents as input
rng, lat_rng = random.split(rng)
latents = random.normal(lat_rng, (16, 128))
# Create decoder
decoder = Decoder(c_hid=32, latent_dim=128, c_out=3)
# Initialize parameters of decoder with random key and latents
rng, init_rng = random.split(rng)
params = decoder.init(init_rng, latents)['params']
# Apply decoder with parameters on the images
out = decoder.apply({'params': params}, latents)
print(out.shape)

del out, decoder, params


class Autoencoder(nn.Module):
    c_hid: int
    latent_dim: int

    def setup(self):
        # Alternative to @nn.compact -> explicitly define modules
        # Better for later when we want to access the encoder and decoder explicitly
        self.encoder = Encoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        self.decoder = Decoder(c_hid=self.c_hid, latent_dim=self.latent_dim, c_out=3)

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# Test Autoencoder implementation.
rng = random.PRNGKey(0)
# Example images as input
imgs = next(iter(train_loader))[0]
# Create encoder
autoencoder = Autoencoder(c_hid=32, latent_dim=128)
# Initialize parameters of encoder with random key and images
params = autoencoder.init(rng, imgs)['params']
# Apply encoder with parameters on the images
out = autoencoder.apply({'params': params}, imgs)
print(out.shape)

del out, autoencoder, params


def mse_recon_loss(model, params, batch):
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((recon_imgs - imgs) ** 2).mean(axis=0).sum()  # Mean over batch, sum over pixels
    return loss


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = ((img1 - img2) ** 2).sum()
    # Plot images for visual comparison
    imgs = jax_to_torch(np.stack([img1, img2], axis=0))
    grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    fig = plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    # plt.show()
    plt.close(fig)


for i in range(2):
    # Load example image
    img, _ = train_dataset[i]
    img_mean = img.mean(axis=(0, 1), keepdims=True)

    # Shift image by one pixel
    SHIFT = 1
    img_shifted = np.roll(img, shift=SHIFT, axis=0)
    img_shifted = np.roll(img_shifted, shift=SHIFT, axis=1)
    img_shifted[:1, :, :] = img_mean
    img_shifted[:, :1, :] = img_mean
    compare_imgs(img, img_shifted, "Shifted -")

    # Set half of the image to zero
    img_masked = np.copy(img)
    img_masked[:img_masked.shape[1] // 2, :, :] = img_mean
    compare_imgs(img, img_masked, "Masked -")


class GenerateCallback:

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_imgs = model.apply({'params': state.params}, self.input_imgs)
            reconst_imgs = jax.device_get(reconst_imgs)

            # Plot and add to tensorboard
            imgs = np.stack([self.input_imgs, reconst_imgs], axis=1).reshape(-1, *self.input_imgs.shape[1:])
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1))
            logger.add_image("Reconstructions", grid, global_step=epoch)


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
        self.log_dir = os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}')
        self.generate_callback = GenerateCallback(self.exmp_imgs, every_n_epochs=50)
        self.logger = SummaryWriter(log_dir=self.log_dir, filename_suffix='.' + conf["optimizer"])
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Training function
        def train_step(state, batch):
            loss_fn = lambda params: mse_recon_loss(self.model, params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, loss

        self.train_step = jax.jit(train_step)

        # Eval function
        def eval_step(state, batch):
            return mse_recon_loss(self.model, state.params, batch)

        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp_imgs)['params']

        batch = next(iter(train_loader))
        loss_fn = lambda params, batch: mse_recon_loss(self.model, params, batch)

        # Initialize learning rate schedule and optimizer
        print("len(train_loader)", len(train_loader))

        if conf["optimizer"] == 'adam':
            optimizer = optax.chain(
                optax.clip(1.0),  # Clip gradients at 1
                optax.adam(self.conf['learning_rate'])
            )
        elif conf["optimizer"] == 'my_adam':
            optimizer = optax.chain(
                optax.clip(1.0),  # Clip gradients at 1
                fosi_adam(optax.adam(conf["learning_rate"]), loss_fn, batch,
                          decay=conf["momentum"],
                          num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                          approx_k=conf["approx_k"],
                          approx_l=conf["approx_l"], warmup_w=conf["num_warmup_iterations"],
                          alpha=conf["alpha"])
            )
        elif conf["optimizer"] == 'momentum':
            optimizer = optax.chain(
                optax.clip(1.0),  # Clip gradients at 1
                optax.sgd(conf["learning_rate"], momentum=0.9, nesterov=False)
            )
        elif conf["optimizer"] == 'my_momentum':
            optimizer = optax.chain(
                optax.clip(1.0),  # Clip gradients at 1
                fosi_momentum(optax.sgd(conf["learning_rate"], momentum=conf["momentum"], nesterov=False), loss_fn,
                              batch,
                              decay=conf["momentum"],
                              num_iters_to_approx_eigs=conf["num_iterations_between_ese"],
                              approx_k=conf["approx_k"],
                              approx_l=conf["approx_l"], warmup_w=conf["num_warmup_iterations"],
                              alpha=conf["alpha"], learning_rate_clip=1.0)
            )
        else:
            raise "Illegal optimizer " + conf["optimizer"]
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

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
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    # self.save_model(step=epoch_idx)
                self.generate_callback.log_generations(self.model, self.state, logger=self.logger, epoch=epoch_idx)
                self.logger.flush()
                print("eval_loss", eval_loss)
            with open(train_stats_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch_idx, train_loss, eval_loss, epoch_end - epoch_start, np.maximum(0, timer() - start_time)])

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        for batch in train_loader:
            self.state, loss = self.train_step(self.state, batch)
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
        return avg_loss

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.eval_step(self.state, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params,
                                    prefix=f'cifar10_{self.latent_dim}_', step=step)

    def delete_model(self):
        # Save current model at certain training iteration
        old_checkpoint = checkpoints.latest_checkpoint(ckpt_dir=self.log_dir, prefix=f'cifar10_{self.latent_dim}_')
        if old_checkpoint:
            os.remove(old_checkpoint)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params,
                                                    prefix=f'cifar10_{self.latent_dim}_')
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'))


def train_cifar(latent_dim, conf=None):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(c_hid=32, latent_dim=latent_dim, conf=conf)
    trainer.delete_model()
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model()
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    test_loss = trainer.eval_model(test_loader)
    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss


for optimizer_name in ['my_momentum', 'momentum', 'my_adam', 'adam']:
    config.update("jax_enable_x64", False)

    # Heavy-Ball (momentum) diverges with learning rate 1e-2. Using 1e-3 instead.
    conf = get_config(optimizer=optimizer_name, approx_k=10, batch_size=256, learning_rate=1e-3,
                      num_iterations_between_ese=800, approx_l=0, alpha=0.01)
    test_folder = start_test(conf["optimizer"], test_folder='test_results_autoencoder_cifar10')
    write_config_to_file(test_folder, conf)

    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "latency", "wall_time"])

    _, _ = train_cifar(128, conf=conf)
