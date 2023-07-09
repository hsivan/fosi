import torchvision
import torchvision.transforms as transforms
import functorch
import torchopt
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F

from fosi import fosi_adam_torch

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")  # "cpu" or "cuda"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                                          collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,
                                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
model, params = functorch.make_functional(net)


def loss_fn(params, batch):
    preds = model(params, batch[0])
    loss = nn.CrossEntropyLoss()(preds, batch[1])
    return loss


def accuracy(params, batch):
    preds = model(params, batch[0])
    predicted_class = torch.argmax(preds, dim=1)
    return torch.sum(predicted_class == batch[1])


base_optimizer = torchopt.adam(lr=0.001)  # torchopt.sgd(net.parameters(), lr=0.001, momentum=0.9)
optimizer = fosi_adam_torch(base_optimizer, loss_fn, next(iter(trainloader)), num_iters_to_approx_eigs=500, alpha=0.01)
opt_state = optimizer.init(params)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # forward + backward + optimize
        loss = loss_fn(params, batch)
        grads = torch.autograd.grad(loss, params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = torchopt.apply_updates(params, updates, inplace=True)

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    acc = 0.0
    num_samples = 0
    for i, batch in enumerate(testloader, 0):
        acc += accuracy(params, batch)
        num_samples += batch[0].shape[0]
    acc /= num_samples
    print(f'Test accuracy: {acc}')

print('Finished Training')
