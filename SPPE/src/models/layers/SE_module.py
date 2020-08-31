from torch import nn
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, m1, m2 = x.size()
        ####修改####
        inputsz = np.array([m1,m2])
        outputsz = np.array([1,1])
        stride = np.floor(inputsz/outputsz).astype(np.int32)
        kernel = inputsz-(outputsz-1)*stride
        avg_pool = nn.AvgPool2d(kernel_size=list(kernel),stride=list(stride))
        y = avg_pool(x)
        #####
        #y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
