import os
# Note: To maintain the default precision as 32-bit and not switch to 64-bit, set the following flag prior to any
# imports of JAX. This is necessary as the jax_enable_x64 flag is later set to True inside the Lanczos algorithm.
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'

from fosi import fosi_adam
import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import optax

key = jax.random.PRNGKey(42)
n_dim = 100
target_params = 0.5

# Single linear layer equals inner product between the input and the network parameters
init_fn, apply_fn = stax.serial(stax.Dense(1, W_init=jax.nn.initializers.zeros, b_init=jax.nn.initializers.zeros))

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