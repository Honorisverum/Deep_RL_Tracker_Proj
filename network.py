import torch.nn as nn
from torch.autograd import Variable
import torch


"""
=================================================
        CREATE NETWORK CNN + LSTM
=================================================
"""


class CNN_LSTM(nn.Module):
    def __init__(self, dim):
        super(CNN_LSTM, self).__init__()

        # output_dim
        self.output_dim = dim

        # feature vec dim
        self.i_t_dim = dim - 4

        # feature and location combo dim
        self.o_t_dim = dim

        # hidden dimension
        self.hidden_dim = dim

        # number of lstm layers
        self.layer_dim = 1

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
        # self.fc_rnn = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

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
        # out = self.fc_rnn(out[-1, :])
        out = out[-1, :]

        # slice last four elements
        out = out[:, -4:]

        return out