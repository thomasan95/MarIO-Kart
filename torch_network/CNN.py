import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    """
    Instantiate the graph. We create a placeholder to feed into the network which is then
    created to use through tf.Session()
    Structure:
        Conv1 - Kernel (5,5), Stride (2,2) out-depth 24 (200x66x3 -> 98x31x24)
        Conv2 - Kernel (5,5), Stride (2,2) out-depth 36 (98x31x24 -> 47x14x36)
        Conv3 - Kernel (5,5), Stride (2,2) out-depth 48 (47x14x36 -> 22x5x48)
        Conv4 - Kernel (3,3), Stride (1,1) out-depth 64 (22x5x48 -> 20x3x64)
        Conv5 - Kernel (3,3), Stride (1,1) out-depth 64 (20x3x64 -> 18x1x64)
        Fc1 - [1152, 1164]
        Fc2 - [1164, 100]
        Fc3 = [100, 50]
        Fc4 = [50, 10]
        Fc5 (output) = [10, conf.OUTPUT_SIZE]
    The network along with all important nodes are then then returned so they can then be used to train
    the graph
    :return: graph, input_placeholder, max_actions placeholder, optimal_action, out, action, loss, optimizer
    :rtype: tf.Graph(), tf.placeholder(), tf.placeholder(), tf.placeholder(), tf.Tensor, tf.Tensor, loss, tf optimizer
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5,stride=2,bias=True)
        self.conv2 = nn.Conv2d(24, 36, 5,stride=2,bias=True)
        self.conv3 = nn.Conv2d(36,48,5,stride=2,bias=True)
        self.conv4 = nn.Conv2d(48,64,3,stride=1,bias=True)
        self.conv5 = nn.Conv2d(64,64,3,stride=1,bias=True)
        
        #1152, 1164
        self.fc1 = nn.Linear(1152, 1164,bias=True)
        self.fc2 = nn.Linear(1164, 100,bias=True)
        self.fc3 = nn.Linear(100, 50,bias=True)
        self.fc4 = nn.Linear(50, 10,bias=True)
        self.fc5 = nn.Linear(10,6,bias=True)
        

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = (F.relu(self.conv5(x)))
        x = x.view(-1, 1152)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc3(x)
        return x