import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        self.output_dim = output_dim

        # convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # polling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected layer
        self.fc = nn.Linear(32 * 5 * 5, self.output_dim, bias=True)

    def forward(self, x, ground_truth):

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
        out = self.fc(out)

        #concatenates ground_truth
        out = torch.cat((out, ground_truth), 1)

        return out

model = CNN(100)


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



"""
HOW TO SAVE
#torch.save(model.state_dict(), PATH)

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()
"""

print(model)

# seq_len, channels (color), lenght x width
input = torch.randn(10, 3, 32, 32)

# seq_len, bounding_box_coordinates
ground_truth = torch.randn(10, 4)

out = model(input, ground_truth)

print(out.size())
