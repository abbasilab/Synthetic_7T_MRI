# SRGAN style model with SynthSeg Loss as the perceptual loss function
import torch
import torch.nn as nn
from .VNet_parts import InputTransition, DownTransition, UpTransition


class VNet_generator(nn.Module):
    def __init__(self):
        super(VNet_generator, self).__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 2)
        self.down_tr64 = DownTransition(32, 3)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.up_tr256 = UpTransition(256, 256, 3)
        self.up_tr128 = UpTransition(256, 128, 3)
        self.up_tr64 = UpTransition(128, 64, 2)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class VNet_discriminator(nn.Module):
    def __init__(self):
        super(VNet_discriminator, self).__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 2)
        self.down_tr64 = DownTransition(32, 3)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.linear1 = nn.Linear(int(256*64**3 / (16**3)), 256)
        self.relu1 = nn.PReLU(256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x, y=None):
        out = self.in_tr(x)
        out = self.down_tr32(out)
        out = self.down_tr64(out)
        out = self.down_tr128(out)
        out = self.down_tr256(out)
        # get bottleneck scalar output
        # global sum pooling
        out = torch.sum(out, [2, 3, 4])
        out = self.relu1(out)
        out = self.linear2(out)
        return out
