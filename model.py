import torch
import torch.nn as nn
import torch.nn.functional as F

# Xarxa MNIST


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 4)
        # self.conv2 = nn.Conv2d(20, 30, 5)
        self.drop = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(103680,50)
        # self.fc_extra = nn.Linear(6000, 50)
        # self.fc2 = nn.Linear(50, 4)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.float() #passem a float pq els parametres de conv son float, sino problemes (x es double)
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.pool(x)
        x = F.relu(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc_extra(x)
        # x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim = 1)








