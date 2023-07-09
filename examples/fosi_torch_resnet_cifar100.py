# Based on https://github.com/weiaicunzai/pytorch-cifar100
import argparse
import csv
import time
import sys
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchopt
from torchopt.typing import Numeric, Scalar
from torch._functorch.make_functional import make_functional_with_buffers
from torch.optim.lr_scheduler import LRScheduler as Schedule

from experiments.dnn.dnn_test_utils import start_test
from fosi import fosi_adam_torch, fosi_momentum_torch

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MILESTONES = [60, 120, 160]
EPOCH = 200


torch.set_default_dtype(torch.float32)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # we use a different input size than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)  # Lanczos default is hvp_forward_ad, however, if switching to hvp_backward_ad should set the stride to 1, as 2 causes exception.
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def get_network(args):
    if args.net == 'resnet18':
        net = resnet18()
    elif args.net == 'resnet34':
        net = resnet34()
    elif args.net == 'resnet50':
        net = resnet50()
    elif args.net == 'resnet101':
        net = resnet101()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)

    return cifar100_test_loader


def loss_function(params, batch):
    outputs = model(params, buffers, batch[0])
    loss = nn.CrossEntropyLoss()(outputs, batch[1])
    # TODO: Weight decay should be added to the loss directly as an L2 regularization (as done here), rather than
    #  indirectly through the optimizer step, i.e., the optimizer weight_decay must be 0.
    #  The reason is that using weight_decay indirectly through the optimizer step, and not directly through the
    #  loss function, results in ESE inaccurate eigenvectors and eigenvalues estimation; the gradient of the loss
    #  is not the real gradient used for the update step.
    l2_norm = weight_decay * 0.5 * sum(p.pow(2.0).sum() for p in params)
    return loss + l2_norm


def loss_function_with_pred(params, batch):
    outputs = model(params, buffers, batch[0])
    loss = nn.CrossEntropyLoss()(outputs, batch[1])
    return loss, outputs


def train(epoch, params, opt_state):
    start = time.time()
    loss_val = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        # forward + backward + optimize
        loss = loss_function(params, (images, labels))
        grads = torch.autograd.grad(loss, params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = torchopt.apply_updates(params, updates, inplace=True)
        loss_val += loss.item()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return params, opt_state, loss_val / len(cifar100_training_loader)


@torch.no_grad()
def eval_training(epoch, params):
    start = time.time()
    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        loss, outputs = loss_function_with_pred(params, (images, labels))

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return test_loss / len(cifar100_test_loader.dataset), correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    weight_decay = 5e-4  # Used inside the loss function

    test_folder = start_test("fosi_momentum", test_folder='test_results_resnet_cifar100')
    train_stats_file = test_folder + "/train_stats.csv"
    with open(train_stats_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "latency", "wall_time"])

    net = get_network(args)
    model, params, buffers = make_functional_with_buffers(net)
    del net

    num_params = sum(p.numel() for p in params)
    print("num_params:", num_params)

    cifar100_training_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=2,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=2,
        batch_size=args.b,
        shuffle=False
    )

    print("cifar100_training_loader:", len(cifar100_training_loader))

    def piecewise_constant_schedule(
            init_value: Scalar,
            boundaries_and_scales: dict,
    ) -> Schedule:
        def schedule(count: Numeric) -> Numeric:
            v = init_value
            if boundaries_and_scales is not None:
                for threshold, scale in sorted(boundaries_and_scales.items()):
                    indicator = torch.maximum(torch.tensor(0.), torch.sign(threshold - count))
                    v = v * indicator + (1 - indicator) * scale * v
            return v

        return schedule


    iter_per_epoch = len(cifar100_training_loader)
    boundaries_and_scales = {int(iter_per_epoch * EPOCH * 0.30): 0.2,
                             int(iter_per_epoch * EPOCH * 0.60): 0.2,
                             int(iter_per_epoch * EPOCH * 0.80): 0.2}
    base_optimizer = torchopt.sgd(lr=piecewise_constant_schedule(args.lr, boundaries_and_scales), momentum=0.9, weight_decay=0)  # TODO: weight_decay > 0 is not supported. It interfere with FOSI's convergence.
    batch = next(iter(cifar100_training_loader))
    batch = (batch[0].cuda(), batch[1].cuda())
    optimizer = fosi_momentum_torch(base_optimizer, loss_function, batch, num_iters_to_approx_eigs=800, alpha=0.01, approx_k=10, learning_rate_clip=1)
    opt_state = optimizer.init(params)

    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()

    best_acc = 0.0
    start_time = 1e10

    for epoch in range(1, EPOCH + 1):
        if epoch == 2:
            start_time = timer()
        epoch_start = timer()
        params, opt_state, train_loss = train(epoch, params, opt_state)
        epoch_end = timer()
        model.eval()
        test_loss, test_acc = eval_training(epoch, params)
        model.train()

        with open(train_stats_file, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [epoch, train_loss, None, test_loss, test_acc, epoch_end - epoch_start, max(0., timer() - start_time)])
