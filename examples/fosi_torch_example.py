from fosi import fosi_adam_torch
import torch
import torchopt
import functorch

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")  # "cpu" or "cuda"


class Net(torch.nn.Module):
    def __init__(self, in_dim):
        super(Net, self).__init__()
        self.hid = torch.nn.Linear(in_dim, 1, bias=False)

    def forward(self, x):
        z = self.hid(x)
        return z


n_dim = 100
net = Net(n_dim).to(device)
net.hid.weight.data.fill_(0.0)
model, params = functorch.make_functional(net)


def loss_fn(params, batch):
    x, y = batch
    y_pred = model(params, x)
    loss = torch.nn.MSELoss()(y_pred, y)
    return loss


def data_generator(target_params, n_dim):
    while True:
        batch_xs = torch.normal(0.0, 1.0, size=(16, n_dim)).to(device)
        batch_ys = torch.unsqueeze(torch.sum(batch_xs * target_params, dim=-1).to(device), -1)
        yield batch_xs, batch_ys


# Generate random data
target_params = 0.5
data_gen = data_generator(target_params, n_dim)

# Construct the FOSI-Adam optimizer. The usage after construction is identical to that of Optax optimizers,
# with the optimizer.init() and optimizer.update() methods.
optimizer = fosi_adam_torch(torchopt.adam(lr=1e-3), loss_fn, next(data_gen), device=device)

# Initialize parameters of the model and optimizer
opt_state = optimizer.init(params)

# A simple update loop.
for i in range(5000):
    loss = loss_fn(params, next(data_gen))
    grads = torch.autograd.grad(loss, params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = torchopt.apply_updates(params, updates, inplace=True)
    if i % 100 == 0:
        print("loss:", loss.item())

assert torch.allclose(params[0], torch.tensor(target_params)), 'Optimization should retrieve the target params used to generate the data.'
