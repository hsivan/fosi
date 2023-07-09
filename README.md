# FOSI

FOSI is a library for improving first order optimizers with second order information.
Given a first-order base optimizer, 
FOSI works by iteratively splitting the function to minimize into pairs of quadratic problems on orthogonal subspaces,
then using Newton's method to optimize one and the base optimizer to optimize the other.

Our analysis of FOSI’s preconditioner and effective Hessian proves that FOSI improves the condition number for a large family of optimizers.
Our empirical evaluation demonstrates that FOSI improves the convergence rate and optimization time of GD, Heavy-Ball, and Adam when applied to several deep neural networks training tasks such as audio classification, transfer learning, and object classification and when applied to convex functions.

For more information, see our paper, [FOSI: Hybrid First and Second Order Optimization](https://arxiv.org/pdf/2302.08484.pdf).

## Installation

FOSI is written in pure Python.
We support installing `fosi` package on Linux (Ubuntu 20.04 or later) and the installation requires Python >=3.8.

### CUDA toolkit

To run FOSI with GPU, CUDA toolkit must be installed.
If using conda environment, the installation command is:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda
```
Otherwise, a global installation is required:
```bash
sudo apt-get install cuda-11-8
```
After installing CUDA toolkit, follow [NVIDIA's environment setup instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup)
to set the environment variables PATH and LD_LIBRARY_PATH.
To find the lib/bin folders in case of conda environment use `find ~ -name 'libcusolver.so.11'` and in case of a
global installation with apt-get `find /usr/ -name 'libcusolver.so.11'` and use the containing folder.

Note: CUDA toolkit installation is not required when using the Docker container to run the experiments, or if running on the CPU.


### FOSI package

Use the following instructions to install a
binary package with `pip`, or to download FOSI's source code.

To download FOSI's source code run:
```bash
git clone https://github.com/hsivan/fosi
```
Let `fosi_root` be the root folder of the project on your local computer, for example `/home/username/fosi`.

To install FOSI run:
```bash
pip install git+https://github.com/hsivan/fosi.git -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Or, download the code and then run:
```bash
pip install <fosi_root> -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Basic usage of FOSI

FOSI can work with both [JAX](https://github.com/google/jax) and [PyTorch](https://github.com/pytorch/pytorch) frameworks.
When using JAX, FOSI uses [Optax](https://github.com/deepmind/optax) optimizers as base optimizers,
and its API is designed to be similar to that of Optax optimizers.
In the case of PyTorch, FOSI utilizes [TorchOpt](https://github.com/metaopt/torchopt) optimizers as base optimizers,
and its API is designed to be similar to that of TorchOpt optimizers.

*Note: Within the FOSI package, you will find implementations of the Lanczos algorithm in both JAX and PyTorch frameworks.
Both implementations utilize the forward-over-reverse technique to efficiently compute the Hessian-vector product.
It is worth mentioning that the just-in-time (jit) compilation time of JAX is minimal, even when dealing with large models and functions containing up to 100 million parameters.*

### JAX
This example demonstrates the application of FOSI with the Adam base optimizer for a program based on JAX.

```python
import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

from fosi import fosi_adam
import jax.numpy as jnp
import jax
from jax.example_libraries import stax
from jax.nn.initializers import zeros
import optax

key = jax.random.PRNGKey(42)
n_dim = 100
target_params = 0.5

# Single linear layer equals inner product between the input and the network parameters
init_fn, apply_fn = stax.serial(stax.Dense(1, W_init=zeros, b_init=zeros))

def loss_fn(params, batch):
    x, y = batch
    y_pred = apply_fn(params, x).squeeze()
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss

def data_generator(key, target_params, n_dim):
    while True:
        key, subkey = jax.random.split(key)
        batch_xs = jax.random.normal(subkey, (16, n_dim))
        batch_ys = jnp.sum(batch_xs * target_params, axis=-1)
        yield batch_xs, batch_ys

# Generate random data
data_gen = data_generator(key, target_params, n_dim)

# Construct the FOSI-Adam optimizer. The usage after construction is identical to that of Optax optimizers,
# with the optimizer.init() and optimizer.update() methods.
optimizer = fosi_adam(optax.adam(1e-3), loss_fn, next(data_gen))

# Initialize parameters of the model and optimizer
_, params = init_fn(key, next(data_gen)[0].shape)
opt_state = optimizer.init(params)

@jax.jit
def step(params, batch, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# A simple update loop.
for i in range(5000):
    params, opt_state, loss = step(params, next(data_gen), opt_state)
    if i % 100 == 0:
        print("loss:", loss)

assert jnp.allclose(params[0][0], target_params), 'Optimization should retrieve the target params used to generate the data.'
```

### PyTorch
This example, which is similar to the previous JAX example, illustrates how to apply FOSI with the Adam base
optimizer for a program based on PyTorch.

```python
from fosi import fosi_adam_torch
import torch
import torchopt
import functorch

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")  # "cpu" or "cuda"
n_dim = 100
target_params = 0.5

# Single linear layer equals inner product between the input and the network parameters
model = torch.nn.Linear(n_dim, 1).to(device)
model.weight.data.fill_(0.0)
model.bias.data.fill_(0.0)
apply_fn, params = functorch.make_functional(model)

def loss_fn(params, batch):
    x, y = batch
    y_pred = apply_fn(params, x)
    # TODO: using torch.nn.MSELoss causes 'RuntimeError: ZeroTensors are immutable' when calling torch.autograd.grad
    #  in lanczos_algorithm::hvp_forward_ad()
    #loss = torch.nn.MSELoss()(y_pred, y)
    loss = torch.mean((y_pred - batch[1])**2)
    return loss

def data_generator(target_params, n_dim):
    while True:
        batch_xs = torch.normal(0.0, 1.0, size=(16, n_dim)).to(device)
        batch_ys = torch.unsqueeze(torch.sum(batch_xs * target_params, dim=-1).to(device), -1)
        yield batch_xs, batch_ys

# Generate random data
data_gen = data_generator(target_params, n_dim)

# Construct the FOSI-Adam optimizer. The usage after construction is identical to that of TorchOpt optimizers,
# with the optimizer.init() and optimizer.update() methods.
optimizer = fosi_adam_torch(torchopt.adam(lr=1e-3), loss_fn, next(data_gen))

# Initialize the optimizer
opt_state = optimizer.init(params)

def step(params, batch, opt_state):
    loss = loss_fn(params, batch)
    grads = torch.autograd.grad(loss, params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = torchopt.apply_updates(params, updates, inplace=True)
    return params, opt_state, loss

# A simple update loop.
for i in range(5000):
    params, opt_state, loss = step(params, next(data_gen), opt_state)
    if i % 100 == 0:
        print("loss:", loss.item())

assert torch.allclose(params[0], torch.tensor(target_params)), 'Optimization should retrieve the target params used to generate the data.'
```

More examples can be found in the `examples` folder.

## Reproduce paper's experimental results

We provide detailed instructions for reproducing the experiments from our paper.
The full [instructions](experiments/README.md) and scripts are in the `experiments` folder.

In the paper, we presented the results of five DNN training tasks.
Our study involved a comparison of FOSI against various optimization methods,
including first-order methods Adam and Heavy-Ball (HB) and partially second-order methods K-FAC and L-BFGS.
We utilized the K-FAC implementation from the [KFAC-JAX](https://github.com/deepmind/kfac-jax) library and the L-BFGS
implementation from the [JAXopt](https://github.com/google/jaxopt) library.
As a base optimizer, FOSI employs Adam and HB.
For further information regarding the experiments, please refer to the paper for the full details.

*Note: Additionally, we offer a [script](experiments/aws_ec2_configure.py) that initiates and configures an AWS EC2 instance with a GPU and the necessary drivers.
This script handles the cloning of the FOSI project onto the instance and installs all the required dependencies.
Before executing the script, it is important to ensure that the prerequisites mentioned at the beginning of the script are met.
Once satisfied, the user can establish an SSH connection to the EC2 instance and promptly execute the provided examples or run the experiments.*

## Citing FOSI

If FOSI has been useful for your research, and you would like to cite it in an academic
publication, please use the following Bibtex entry:
```bibtex
@misc{sivan_fosi_2023,
  author = {Sivan, Hadar and Gabel, Moshe and Schuster, Assaf},
  title = {FOSI: Hybrid First and Second Order Optimization},
  year = {2023},
  doi = {10.48550/ARXIV.2302.08484},
  url = {https://arxiv.org/abs/2302.08484},
  publisher = {arXiv},
}
```
