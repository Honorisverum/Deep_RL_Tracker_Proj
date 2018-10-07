import torch.nn as nn
from torch.autograd import Variable
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()

        # hidden dimension
        self.hidden_dim = hidden_dim

        # number of hidden layers
        self.layer_dim = layer_dim

        """
        Input: seq_length x batch_size x input_size
        Output: seq_length x batch_size x hidden_size
        """
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        # layer_dim : number of lstm layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):

        # initialize hidden/cell state with zeros
        # since x is a vec, size(0) yield his len
        h_0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c_0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, (_, _) = self.lstm(x, (h_0, c_0))

        # rid of first batch_dim
        out = self.fc(out[-1, :])

        #slice last four elements
        out = out[:, -4:]

        return out

T=10
batch_size=1

in_ten = torch.randn(batch_size, T, 50)
print(in_ten.size())

net = LSTM(50, 50, 1, 50)
#print(in_ten)
out_ten = net(in_ten)
#print(out_ten)
print(out_ten.size())

