import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

"""
=================================================
        LOADING FRAMES AND GROUND TRUTH
=================================================
"""

f = torch.randn(4, 3)

print(f)

for i in f:
    print(i)