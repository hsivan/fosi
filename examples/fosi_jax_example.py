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
for i in range(5000):
  loss, grads = jax.value_and_grad(loss_fn)(params, next(data_gen))
  updates, opt_state = jax.jit(optimizer.update)(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  if i % 100 == 0:
    print("loss:", loss)

assert jnp.allclose(params, target_params), 'Optimization should retrieve the target params used to generate the data.'
