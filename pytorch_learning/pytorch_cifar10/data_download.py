#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File Name : cifar10_classifer_pytorch.py
# @Purpose :
# @Creation Date : 2019-03-19 21:46:24
# @Last Modified : 2019-03-20 10:28:08
# @Created By :  chenjiang
# @Modified By :  chenjiang


from os import sys, path


import torch
import torchvision
import torchvision.transforms as transforms

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



