import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import datasets
from torchvision import transforms

length = 320
width = 240


batch_size = 100
path = "/Users/OTECFIZIKI/Visual_Tracking/python/Vid_A_balll/" + "gt.txt"

# load to numpy
arr = np.loadtxt(path, delimiter=',')

# convert to torch
gt_tens = torch.from_numpy(arr)


# normalization
def gt_normalize(gt_tens, len, wid):
    first_col = gt_tens[:, 0].unsqueeze(1)/len
    secnd_col = gt_tens[:, 1].unsqueeze(1) / wid
    third_col = gt_tens[:, 2].unsqueeze(1) / len
    fouth_col = gt_tens[:, 3].unsqueeze(1) / wid
    return torch.cat((first_col, secnd_col, third_col, fouth_col), 1)

# apply normalization
gt_tens = gt_normalize(gt_tens, length, width)

# put to loader
gt_loader = torch.utils.data.DataLoader(gt_tens, batch_size=batch_size, shuffle=False)

# location vec for training
def gt_location_vec(gt_seq):
    len = gt_seq.size(0)
    return torch.cat((gt_seq[1].unsqueeze(0), torch.zeros((len - 1, 4), dtype=torch.float64)), 0)

# FOR SINGLE NEXT BATCH
# next(iter(data_loader))
