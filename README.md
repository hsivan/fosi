# FOSI

FOSI is a library for improving first order optimizers with second order information.
TODO.

For more information, see our ICML 2023 paper, [FOSI: Hybrid First and Second Order Optimization](TODO).

## Installation

FOSI is written in pure Python.
Use the following instructions to install a
binary package with `pip`, or to download FOSI's source code.
We support installing `fosi` package on Linux (Ubuntu 18.04 or later).
**The installation requires Python >=3.9, <3.11**.

To download FOSI's source code run:
```bash
git clone https://github.com/hsivan/fosi
```
Let `fosi_root` be the root folder of the project on your local computer, for example `/home/username/fosi`.

To install FOSI run:
```bash
pip install git+https://github.com/hsivan/fosi.git
```
Or, download the code and then run:
```bash
pip install <fosi_root>
```

## Usage example

The following example shows how to apply FOSI with the base optimizer Adam.

```python
import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'
from fosi import fosi_adam
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

# CIFAR-10 dataset
train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True)
train_dataset = train_ds.batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batch = next(iter(train_dataset))

# Define the loss function
def mse_recon_loss(model, params, batch):
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((recon_imgs - imgs) ** 2).mean(axis=0).sum()  # Mean over batch, sum over pixels
    return loss

# Convert the loss function into a functions of the form f(params, batch)
loss_fn = lambda params, batch: mse_recon_loss(model, params, batch)

# Construct the FOSI-Adam optimizer
optimizer = fosi_adam(optax.adam(1e-3), loss_fn, batch)

# Using the optimizer is identical to using any other optax optimizer, with the
# optimizer.init() and optimizer.update() methods.
```

More examples can be found in the `experiments/dnn` folder.

## Reproduce paper's experimental results

We provide detailed instructions for reproducing the experiments from our paper.
The full [instructions](experiments/README.md) and scripts are in the `experiments` folder.

## Citing AutoMon

If AutoMon has been useful for your research, and you would like to cite it in an academic
publication, please use the following Bibtex entry:
```bibtex
@inproceedings{sivan_fosi_2023,
  author    = {Sivan, Hadar and Gabel, Moshe and Schuster, Assaf},
  title     = {{FOSI}: Hybrid First and Second Order Optimization},
  year      = {2023},
  series    = {ICML '23},
  booktitle = {Proceedings of the 2022 {ICML} International Conference on Machine Learning},
  note      = {to appear}
}
```
