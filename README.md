# FOSI

FOSI is a library for improving first order optimizers with second order information.
Given a first-order base optimizer, 
FOSI works by iteratively splitting the function to minimize into pairs of quadratic problems on orthogonal subspaces,
then using Newton's method to optimize one and the base optimizer to optimize the other.

Our analysis of FOSI’s preconditioner and effective Hessian proves that FOSI improves the condition number for a large family of optimizers.
Our empirical evaluation demonstrates that FOSI improves the convergence rate and optimization time of GD, Heavy-Ball, and Adam when applied to several deep neural networks training tasks such as audio classification, transfer learning, and object classification and when applied to convex functions.

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

## Basic usage of FOSI

The following example shows how to apply FOSI with the base optimizer Adam.
FOSI is implemented in [JAX](https://github.com/google/jax) and supports [Optax](https://github.com/deepmind/optax)
optimizers as base optimizers.
FOSI's API is also similar to that of Optax optimizers.


```python
import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'
from fosi import fosi_adam
import jax.numpy as jnp
import jax
import optax

def network(params, x):
    return jnp.dot(x, params)

def loss_fn(params, batch):
    x, y = batch
    y_pred = network(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss

def data_generator(key, target_params, n_dim):
    while True:
        key, subkey = jax.random.split(key)
        batch_xs = jax.random.normal(subkey, (16, n_dim))
        batch_ys = jnp.sum(batch_xs * target_params, axis=-1)
        yield batch_xs, batch_ys

# Generate random data
n_dim = 100
key = jax.random.PRNGKey(42)
target_params = 0.5
data_gen = data_generator(key, target_params, n_dim)

# Construct the FOSI-Adam optimizer. The usage after construction is identical to that of Optax optimizers,
# with the optimizer.init() and optimizer.update() methods.
optimizer = fosi_adam(optax.adam(1e-3), loss_fn, next(data_gen))

# Initialize parameters of the model and optimizer
params = jnp.zeros(n_dim)
opt_state = optimizer.init(params)

# A simple update loop.
for _ in range(5000):
  grads = jax.grad(loss_fn)(params, next(data_gen))
  updates, opt_state = jax.jit(optimizer.update)(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  print(params)

assert jnp.allclose(params, target_params), 'Optimization should retrieve the target params used to generate the data.'
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
