import math
import torch
import torch.nn as nn


class Dense_B(nn.Module):
    '''
    A DenseNet-B as described in the paper, the output size is always 4*k
    '''

    def __init__(self, in_channel, k=32):
        super(Dense_B, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_channel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, 4 * k, (1, 1)),
                                  nn.BatchNorm2d(4 * k),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(4 * k, k, (3, 3)))

    def forward(self, x):
        out = self.conv(x)
        return out


class DenseBlock(nn.Module):
    '''
    A denseblock in the densenet
    '''

    def __init__(self, in_channel, layers, k=32):
        super(DenseBlock, self).__init__()
        self.layers = layers
        self.net = [Dense_B(in_channel + i * k, k) for i in range(layers)]

    def forward(self, x):
        x_out = [x]
        for i in range(self.layers):
            if not i:
                x_out.append(self.net[i](torch.cat(tuple(x_out), 1)))
            else:
                x_out.append(self.net[i](x))
        return x_out


class Transition(nn.Module):
    '''
    The trasition layer with parameter theta
    '''

    def __init__(self, in_channel, theta):
        super(Transition, self).__init__()
        self.pooling = nn.Sequential(nn.Conv2d(in_channel, int(theta * in_channel), (1, 1)),
                                     nn.AvgPool2d((2, 2), 2))

    def forward(self, x):
        x = torch.cat(tuple(x), 1)
        out = self.pooling(x)
        return out


class DenseNet121(nn.Module):
    '''
    An implementation of densenet121
    '''

    def __init__(self, in_channel, out_class, theta, k=32):
        super(DenseNet121, self).__init__()
        # size below may be wrong
        self.trans_size1 = 6 * k + in_channel
        self.trans_size2 = int(self.trans_size1 * theta) + 12 * k
        self.trans_size3 = int(self.trans_size1 * theta) + 24 * k
        self.fc_size = int(self.trans_size3 * theta) + 12 * k
        self.net = nn.Sequential(nn.Conv2d(in_channel, 2 * k, (7, 7), 2),
                                 nn.MaxPool2d((3, 3), 2),
                                 DenseBlock(2 * k, 6, k),
                                 Transition(self.trans_size1, theta),
                                 DenseBlock(int(self.trans_size1 * theta), 12, k),
                                 Transition(self.trans_size2, theta),
                                 DenseBlock(int(self.trans_size2 * theta), 24, k),
                                 Transition(self.trans_size3, theta),
                                 DenseBlock(int(self.trans_size3 * theta), 12, k),
                                 nn.AvgPool2d((7, 7)))
        self.fc = nn.Linear(self.fc_size, out_class)

    def forward(self, x):
        x = self.net(x)
        out = nn.Softmax(self.fc(x))
        return out
