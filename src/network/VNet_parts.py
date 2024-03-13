# %%
import torch
import torch.nn as nn
# import torch.nn.functional as F

# %%
# parts of design based on https://github.com/mattmacy/vnet.pytorch
def _make_nConv(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)

def passthrough(x, **kwargs):
    return x

class LUConv(nn.Module):
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(nchan)
        self.relu1 = nn.PReLU(nchan)
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out
    
class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.relu1 = nn.PReLU(outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = torch.add(out, x16)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = nn.PReLU(outChans)
#         self.relu2 = nn.PReLU(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = torch.add(out, down)
        return out
    
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv = nn.Conv3d(inChans, outChans//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(outChans//2)
        self.do1 = passthrough
        self.do2 = passthrough
        self.relu1 = nn.PReLU(outChans//2)
        if dropout:
            self.do1 = nn.Dropout3d()
            self.do2 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        up = self.relu1(self.bn1(self.up_conv(self.upsample(out))))
        up = torch.cat((up, skipxdo), 1)
        out = self.ops(up)
        out = torch.add(out, up)
        return out
    
