import torch
import torch.nn as nn
from .WatNet_3D_parts import *


class WatNet3D(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        self.c1 = conv_init_3D()
        self.d1 = DBlock_3D()
        self.wat12 = wat_layer_1_12_3D()
        self.wat3 = wat_layer_1_3_3D()
        self.inter = DBlock_interim_3D()
        self.recon = DBlock_recon_3D()
        self.final_up = nn.Conv3d(64, 1, 17, stride=1, padding=8)


    def forward(self, x):
        watp1, watp2, watp3 = wat_3_3D(x,'haar')
        t = self.c1(x)
        t = self.d1(t)
        t = self.wat12(t, watp1)
        t = self.d1(t)
        t = self.wat12(t, watp2)
        t = self.d1(t)
        t = self.wat3(t, watp3)
        t = self.inter(t)
        t = self.inter(t)
        t = self.recon(t)
        t = self.recon(t)
        t = self.recon(t)
        t = self.final_up(t)
        t = torch.add(t, x)
        return t