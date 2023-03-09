# Minimal Iris example

import numpy as np
import torch
import torchopt
import functorch

from fosi import fosi_adam_torch

torch.set_default_dtype(torch.float32)


device = torch.device("cuda")  # "cpu" or "cuda"


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(4, 7)  # 4->7->3
        self.oupt = torch.nn.Linear(7, 3)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = self.oupt(z)
        return z


def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)


def main():
    # 0. get started
    print("\nBegin minimal PyTorch Iris demo ")
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. set up training data
    print("\nLoading Iris train data ")

    train_x = np.array([
        [5.0, 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [5.5, 2.6, 4.4, 1.2],
        [6.1, 3.0, 4.6, 1.4],
        [6.7, 3.1, 5.6, 2.4],
        [6.9, 3.1, 5.1, 2.3]], dtype=np.float32)

    train_y = np.array([0, 0, 1, 1, 2, 2], dtype=np.float32)

    print("\nTraining predictors:")
    print(train_x)
    print("\nTraining class labels: ")
    print(train_y)

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)

    # 2. create network
    net = Net().to(device)

    # 3. train model
    max_epochs = 100
    lrn_rate = 0.04

    model, params = functorch.make_functional(net)

    def loss_fn(params, batch):
        preds = model(params, batch[0])
        loss = torch.nn.CrossEntropyLoss()(preds, batch[1])
        return loss

    batch = (train_x[0].reshape(1, 4), train_y[0].reshape(1, ))

    base_optimizer = torchopt.adam(lr=lrn_rate)
    optimizer = fosi_adam_torch(base_optimizer, loss_fn, batch, num_iters_to_approx_eigs=100, alpha=0.01, device=device)
    opt_state = optimizer.init(params)

    print("Num trainable parameters", count_parameters(params))

    print("\nStarting training ")
    indices = np.arange(6)
    for epoch in range(0, max_epochs):
        np.random.shuffle(indices)
        for i in indices:
            X = train_x[i].reshape(1, 4)
            Y = train_y[i].reshape(1, )
            loss = loss_fn(params, (X, Y))
            if i % 200 == 0:
                print(loss)
            # Update the gradient of net to update the parameters
            grads = torch.autograd.grad(loss, params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = torchopt.apply_updates(params, updates, inplace=True)
        # (monitor error)
    print("Done training ")

    # 4. use the model to make a prediction
    print("\nPredicting species for [5.8, 2.8, 4.5, 1.3]: ")
    unk = np.array([[5.8, 2.8, 4.5, 1.3]], dtype=np.float32)
    unk = torch.tensor(unk, dtype=torch.float32).to(device)
    logits = model(params, unk).to(device)
    probs = torch.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()  # allows print options

    np.set_printoptions(precision=4)
    print(probs)  # Should predict class 1

    print("\nEnd Iris demo")


if __name__ == "__main__":
    main()
