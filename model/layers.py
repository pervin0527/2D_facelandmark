import torch
from torch import nn

def conv_bn_relu(in_channels, out_channels, kernel=3, stride=1, padding=1):
    conv_bn_relu_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))
    
    return conv_bn_relu_block


def conv_bn_silu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
    conv_bn_silu_block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                          nn.BatchNorm2d(out_channels)]
    
    if activation:
        conv_bn_silu_block.append(nn.SiLU())

    return nn.Sequential(*conv_bn_silu_block)



class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(in_channels * expand_ratio),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1, groups=in_channels * expand_ratio, bias=False),
                                  nn.BatchNorm2d(in_channels * expand_ratio),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, se_ratio=None, skip_connection=True, drop_connect_rate=0.2):
        super().__init__()
        self.stride = stride
        self.skip_connection = skip_connection and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        self.drop_connect_rate = drop_connect_rate

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(hidden_dim),
                                            nn.SiLU())
        else: 
            self.expand_conv = nn.Identity()

        # Depthwise convolution
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.SiLU())

        # Squeeze-and-Excitation layer
        if se_ratio is not None:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(hidden_dim, num_squeezed_channels, 1, bias=True),
                                    nn.SiLU(),
                                    nn.Conv2d(num_squeezed_channels, hidden_dim, 1, bias=True),
                                    nn.Sigmoid())
        else:
            self.se = nn.Identity()

        # Output phase
        self.project_conv = nn.Sequential(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.BatchNorm2d(out_channels))

        # Matching dimension for skip connection, if needed
        if self.stride != 1 or in_channels != out_channels:
            self.identity_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                                               nn.BatchNorm2d(out_channels))
        else:
            self.identity_conv = None

    def forward(self, x):
        identity = x

        if self.expand_conv is not nn.Identity:
            x = self.expand_conv(x)
        x = self.dwconv(x)

        if self.se is not nn.Identity:
            x = x * self.se(x)
        x = self.project_conv(x)

        if self.drop_connect_rate > 0 and self.training:
            x = self.drop_connect(x, p=self.drop_connect_rate)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)

        if self.skip_connection:
            x = x + identity

        return x

    def drop_connect(self, inputs, p):
        """Apply drop connect."""
        if not self.training:
            return inputs
        batch_size = inputs.size(0)
        keep_prob = 1 - p
        random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor

        return output