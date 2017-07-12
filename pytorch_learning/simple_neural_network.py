#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File Name : simple_neural_network.py
# @Purpose :
# @Creation Date : 2017-07-09 08:09:12
# @Last Modified : 2017-07-10 14:50:22
# @Created By :  chenjiang


import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
reload(sys)
sys.setdefaultencoding("utf8")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), (2, 2))
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print net


parameters = list(net.parameters())
print len(parameters)
print parameters[0].size()

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)


net.zero_grad()
out.backward(torch.randn(1, 10))
