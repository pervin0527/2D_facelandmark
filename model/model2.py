import torch
from torch import nn
from model.layers import MBConv, conv_bn_silu

class LandmarkEstimationV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Replace InvertedResidual with MBConv
        self.conv3_1 = MBConv(64, 64, stride=2, expand_ratio=2, se_ratio=0.25)

        self.block3_2 = MBConv(64, 64, stride=1, expand_ratio=2, se_ratio=0.25, skip_connection=True)
        self.block3_3 = MBConv(64, 64, stride=1, expand_ratio=2, se_ratio=0.25, skip_connection=True)
        self.block3_4 = MBConv(64, 64, stride=1, expand_ratio=2, se_ratio=0.25, skip_connection=True)
        self.block3_5 = MBConv(64, 64, stride=1, expand_ratio=2, se_ratio=0.25, skip_connection=True)

        self.conv4_1 = MBConv(64, 128, stride=2, expand_ratio=2, se_ratio=0.25)

        self.conv5_1 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25)
        self.block5_2 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, skip_connection=True)
        self.block5_3 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, skip_connection=True)
        self.block5_4 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, skip_connection=True)
        self.block5_5 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, skip_connection=True)
        self.block5_6 = MBConv(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, skip_connection=True)

        self.conv6_1 = MBConv(128, 16, stride=1, expand_ratio=2, se_ratio=0.25)

        self.conv7 = conv_bn_silu(16, 32, 3, 2, activation=True)
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.bn8(self.conv8(x)))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks
    
class AuxiliaryNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn_silu(64, 128, 3, 2, activation=True)
        self.conv2 = conv_bn_silu(128, 128, 3, 1, activation=True)
        self.conv3 = conv_bn_silu(128, 32, 3, 2, activation=True)
        self.conv4 = conv_bn_silu(32, 128, 7, 1, activation=False)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x