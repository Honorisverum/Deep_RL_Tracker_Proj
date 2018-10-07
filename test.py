import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

"""
================================
PARAMETRS AND LOAD FRAMES
================================
"""

sigma = 0.01
N = 15
T = 20

out = torch.randn(2, 4)

# sample predictions
# network_output : (T, 4)
# return size : (T, N, 4)
def sample_predictions(network_output):
    tens_list = []
    for i in range(network_output.size(0)):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(network_output[i], sigma * torch.eye(4))
        sample = dist.rsample(sample_shape=torch.Size([N])).unsqueeze(0)
        tens_list.append(sample)
    return torch.cat(tens_list, 0)


pred = sample_predictions(out)
print(pred.size())