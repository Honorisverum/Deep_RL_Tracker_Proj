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

image_size = (32, 32)

length = 320
width = 240
sigma = 0.01
N = 30
T = 20
initial_stage = 15
late_stage = 15
epochs = initial_stage + late_stage
rewards_curve = []
#602 frames
DataSet_Name = "/Vid_A_ball"
root = os.getcwd() + "/videos" + DataSet_Name
gt_path = os.getcwd() + "/videos" + DataSet_Name + "/frames/gt.txt"

# Normalize : mean and std for 3 channels
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

frame_set = datasets.ImageFolder(root=root, transform=transform)
frame_loader = torch.utils.data.DataLoader(frame_set, batch_size=T, shuffle=False)

"""
================================
LOAD AND MANAGE GROUND TRUTH
================================
"""

# load gt to numpy
arr = np.loadtxt(gt_path, delimiter=',', dtype=np.float32)

# convert to torch
gt_tens = torch.from_numpy(arr)


# normalization
def gt_norm(gt, len, wid):
    gt[:, 0] /= len
    gt[:, 1] /= wid
    gt[:, 2] /= len
    gt[:, 3] /= wid
    return gt


# apply normalization
gt_tens = gt_norm(gt_tens, length, width)

# put to loader
gt_loader = torch.utils.data.DataLoader(gt_tens, batch_size=T, shuffle=False)


# location vec for training
def gt_location_vec(gt_seq):
    len = gt_seq.size(0)
    return torch.cat((gt_seq[0].unsqueeze(0), torch.zeros((len - 1, 4), dtype=torch.float32)), 0)


"""
================================
CNN + LSTM CLASS
================================
"""


class CNN_LSTM(nn.Module):
    def __init__(self, o_t_dim, hidden_dim, output_dim):
        super(CNN_LSTM, self).__init__()

        # output_dim
        self.output_dim = output_dim

        # feature vec dim
        self.i_t_dim = o_t_dim - 4

        # feature and location combo dim
        self.o_t_dim = o_t_dim

        # convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # polling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected layer
        self.fc_cnn = nn.Linear(32 * 5 * 5, self.i_t_dim, bias=True)

        # hidden dimension
        self.hidden_dim = hidden_dim

        # number of lstm layers
        self.layer_dim = 1

        """
        Input: seq_length x batch_size x input_size
        Output: seq_length x batch_size x hidden_size
        """
        # batch_first=True causes input/output tensors
        # to be of shape (batch_dim, seq_dim, feature_dim)
        # layer_dim : number of lstm layers
        self.lstm = nn.LSTM(self.o_t_dim, self.hidden_dim,
                            self.layer_dim, batch_first=True)

        # Readout layer
        self.fc_rnn = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x, ground_truth):
        # initialize hidden/cell state with zeros
        # since x is a vec, size(0) yield his len
        # 1 is batch size
        h_0 = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
        c_0 = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))

        # convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # pooling 1
        out = self.pool1(out)

        # convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # pooling 2
        out = self.pool2(out)

        # Resize
        # Original size: (100, 32, 5, 5)
        # New out size: (100, 32*5*5)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc_cnn(out)

        # concatenates ground_truth
        out = torch.cat((out, ground_truth), 1)

        # batch_size = 1 in our case
        out = out.unsqueeze(0)

        # lstm
        out, (_, _) = self.lstm(out, (h_0, c_0))

        # rid of first batch_dim
        out = self.fc_rnn(out[-1, :])

        # slice last four elements
        out = out[:, -4:]

        return out


"""
================================
CREATING NET
================================
"""

# create net
net = CNN_LSTM(304, 304, 304)

# Learning rate
learning_rate = 0.01

# choose optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

"""
# print model parametrs
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size(), param_tensor)

for param in net.parameters():
    print(param.requires_grad)
"""

"""
================================
SAMPLE PREDICTIONS AND CALCULATE REWARDS
================================
"""


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


# first type of reward
def reward1(pred, gt):
    subtraction = abs(pred - gt)
    return - subtraction.mean().item() - max(subtraction).item()


# second type of reward
def reward2(pred, gt):
    # calculating length and width of intersect area
    dx = min(pred[0].item() + pred[2].item() / 2, gt[0].item() + gt[2].item() / 2) - \
         max(pred[0].item() - pred[2].item() / 2, gt[0].item() - gt[2].item() / 2)
    dy = min(pred[1].item() + pred[3].item() / 2, gt[1].item() + gt[3].item() / 2) - \
         max(pred[1].item() - pred[3].item() / 2, gt[1].item() - gt[3].item() / 2)

    # intersect square
    if (dx >= 0) and (dy >= 0):
        intersect = dx * dy
    else:
        return 0

    # union area
    union = pred[2].item() * pred[3].item() + gt[2].item() * gt[3].item() - intersect

    return intersect / union


# predictions : (T, N, 4)
# ground truth : (T, 4)
# out : (T, N)
def compute_rewards(predictions, ground_truth, reward_func):
    out_rewards = torch.zeros(ground_truth.size(0), N)
    for i in range(ground_truth.size(0)):
        for j in range(N):
            out_rewards[i][j] = reward_func(predictions[i][j], ground_truth[i])
    return out_rewards


"""
================================
BASELINES AND LOSS
================================
"""


# rewards : (T, N)
# out : (T)
def compute_baselines(rewards):
    return torch.mean(rewards, dim=1)


# rewards : (T, N)
# baselines : (T)
# net_out : (T, 4)
# predictions : (T, N, 4)
# out : number
def compute_loss(rewards, baselines, net_out, predictions):
    baselines = baselines.unsqueeze(1).expand_as(rewards)
    shifted_rewards = baselines - rewards
    net_out = net_out.unsqueeze(1).expand_as(predictions)
    squares = (net_out - predictions) ** 2 / (2 * sigma ** 2)
    squares = torch.sum(squares, dim=2)
    return torch.sum(squares * shifted_rewards)



for epoch in range(1, epochs + 1):

    # initial frame num for print info
    frame_num = 1

    # reward func depend on stage
    if epoch <= initial_stage:
        reward_func = reward1
    else:
        reward_func = reward2

    for gt, (images, _) in zip(gt_loader, frame_loader):

        # images and gt
        images = Variable(images, requires_grad=True)
        gt = Variable(gt, requires_grad=False)

        # compute location vec
        loc_vec = Variable(gt_location_vec(gt), requires_grad=True)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output
        outputs = Variable(net(images, loc_vec), requires_grad=True)

        # sample predictions
        predictions = Variable(sample_predictions(outputs), requires_grad=False)

        # calculate rewards
        rewards = Variable(compute_rewards(predictions, gt, reward_func), requires_grad=False)

        # calculate baselines
        baselines = Variable(compute_baselines(rewards), requires_grad=False)

        # calculate reward fot iteration for learning curve
        ep_reward = sum(baselines).item()
        rewards_curve.append(ep_reward)

        # Calculate Loss
        loss = Variable(compute_loss(rewards, baselines, outputs, predictions), requires_grad=True)

        # Getting gradients
        loss.backward()

        # Updating parameters
        optimizer.step()

        # print info for iteration
        iteration_info_format = {
            'fst_frame': frame_num,
            'lst_frame': frame_num + T - 1,
            'epoch_num': epoch,
            'total_ep_reward': round(ep_reward, 3)
        }
        print("Training end for {fst_frame}...{lst_frame} frammes |"
              " Epoch: {epoch_num} |"
              " Total Reward: {total_ep_reward}".format(**iteration_info_format))

        # shift frame number
        frame_num += images.size(0)


# learning curve
plt.plot(rewards_curve, color='orange')
plt.xlabel('iteration each T frames')
plt.ylabel('cumulative reward')
plt.savefig('learning_curve')
plt.close()
