
import math

import torch
from torch import nn as nn

class LunaModel(nn.Module):
    
    def __init__(self, in_channels=1, conv_channels=8, bn_features=1):
        super().__init__()

        # normalize input
        self.bn = nn.BatchNorm3d(num_features=bn_features)

        # conv blocks
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.fc = nn.Linear(in_features=conv_channels * 8 * 2 * 3 * 3, out_features=2)  # input voxel dim is 32x48x48 (from dataset class), which is halved x4 becomes 2x3x3

        self.init_weights()

    def forward(self, x):
        out = self.bn(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(out.size(0), -1))    # flatten conv out for FC layer => (N, ...)

        # return both raw logits and softmax probability for each class
        return out, torch.softmax(out, dim=1)
    
    def init_weights(self):
        # init weights and bias of each layer
        for module in self.modules():
            if type(module) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(module.bias, -bound, bound)




# each block is two conv layers then maxpool layer
class LunaBlock(nn.Module):

    def __init__(self, in_channels, conv_channels, kernel_size=3, padding=1, maxpool_dim=2):    # kernel=3, padding=1 keeps same dimension. maxpool_dim=2 halves
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.maxpool = nn.MaxPool3d(maxpool_dim, maxpool_dim)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        return self.maxpool(out)