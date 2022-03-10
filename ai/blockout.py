import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding="same") #conv2d(in_channels, out_chnnels, kernel_size, stride=1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128) # = 64 * 15 * 15 = out_channels * pooled_img
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x): # 256
        x = self.conv1(x) # 254
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 127
        x = self.conv2(x) # 125
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 62
        x = self.conv3(x) # 60
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 30
        x = self.conv3(x) # 28
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 14
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
