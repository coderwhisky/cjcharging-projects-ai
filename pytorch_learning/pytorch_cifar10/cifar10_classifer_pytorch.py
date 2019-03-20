#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File Name : cifar10_classifer_pytorch.py
# @Purpose :
# @Creation Date : 2019-03-20 11:37:23
# @Last Modified : 2019-03-20 11:41:13
# @Created By :  chenjiang
# @Modified By :  chenjiang


from os import sys, path
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as optim

# ---- data load
transform = transforms.Compose(
        [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root="./data", train=True, 
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, 
        num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, 
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
        num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(Function.relu(self.conv1(x)))
        x = self.pool(Function.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = Function.relu(self.fc1(x))
        x = Function.relu(self.fc2(x))
        x = self.fc3(x)

net = Net()


criterion = nn.CrossEntropyloss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    runing_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        runing_loss += loss.item()
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' % 
                    (epoch + 1, i + 1, running_loss / 2000))
            runing_loss = 0.0
print("finish training")





