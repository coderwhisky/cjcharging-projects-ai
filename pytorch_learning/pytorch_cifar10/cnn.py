#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File Name : cnn.py
# @Purpose :
# @Creation Date : 2019-03-20 10:16:14
# @Last Modified : 2019-03-20 11:37:10
# @Created By :  chenjiang
# @Modified By :  chenjiang


from os import sys, path


import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as optim

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





