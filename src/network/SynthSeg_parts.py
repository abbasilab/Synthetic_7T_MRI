import torch
import torch.nn as nn


class InputTransitionSS(nn.Module):
    def __init__(self, outChans=24):
        super(InputTransitionSS, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, padding=1)
        self.elu1 = nn.ELU(alpha=1)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, padding=1)
        self.elu2 = nn.ELU(alpha=1)

    def forward(self, x):
        out = self.elu2(self.conv2(self.elu1(self.conv1(x))))
        return out


class BnMpConvConv(nn.Module):
    def __init__(self, nchan):
        super(BnMpConvConv, self).__init__()
        self.bn1 = nn.BatchNorm3d(nchan, eps=1e-03, momentum=0.99)
        self.mp1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv3d(nchan, nchan * 2, kernel_size=3, padding=1)
        self.elu1 = nn.ELU(alpha=1)
        self.conv2 = nn.Conv3d(nchan * 2, nchan * 2, kernel_size=3, padding=1)
        self.elu2 = nn.ELU(alpha=1)

    def forward(self, x):
        out = self.elu2(self.conv2(self.elu1(self.conv1(self.mp1(self.bn1(x))))))
        return out


class ConvConvBnUp(nn.Module):
    def __init__(self, nchan):
        super(ConvConvBnUp, self).__init__()
        outchan = int(nchan / 2)
        nchan = int(nchan * 1.5)
        self.conv1 = nn.Conv3d(nchan, outchan, kernel_size=3, padding=1)
        self.elu1 = nn.ELU(alpha=1)
        self.conv2 = nn.Conv3d(outchan, outchan, kernel_size=3, padding=1)
        self.elu2 = nn.ELU(alpha=1)
        self.bn1 = nn.BatchNorm3d(outchan, eps=1e-03, momentum=0.99)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skipx):
        out = torch.cat((x, skipx), 1)
        out = self.up1(self.bn1(self.elu2(self.conv2(self.elu1(self.conv1(out))))))
        return out


class SynthSegModel(nn.Module):
    def __init__(self):
        super(SynthSeg, self).__init__()
        self.in_tr = InputTransitionSS()
        self.down_tr48 = BnMpConvConv(24)
        self.down_tr96 = BnMpConvConv(48)
        self.down_tr192 = BnMpConvConv(96)
        self.down_tr384 = BnMpConvConv(192)
        self.bn_bottleneck = nn.BatchNorm3d(384, eps=1e-03, momentum=0.99)
        self.up_bottleneck = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_tr192 = ConvConvBnUp(384)
        self.up_tr96 = ConvConvBnUp(192)
        self.up_tr48 = ConvConvBnUp(96)

        # output transition blocks
        self.conv1 = nn.Conv3d(72, 24, kernel_size=3, padding=1)
        self.elu1 = nn.ELU(alpha=1)
        self.conv2 = nn.Conv3d(24, 24, kernel_size=3, padding=1)
        self.elu2 = nn.ELU(alpha=1)
        self.bn1 = nn.BatchNorm3d(24, eps=1e-03, momentum=0.99)
        self.conv_likelihood = nn.Conv3d(24, 33, kernel_size=1)
        self.conv_predict = nn.Softmax(dim=1)

        self.add_module('in_tr', self.in_tr)
        self.add_module('down_tr48', self.down_tr48)
        self.add_module('down_tr96', self.down_tr96)
        self.add_module('down_tr192', self.down_tr192)
        self.add_module('down_tr384', self.down_tr384)
        self.add_module('up_tr96', self.up_tr96)
        self.add_module('up_tr192', self.up_tr192)
        self.add_module('up_tr48', self.up_tr48)

        self.add_module('bn_bottleneck', self.bn_bottleneck)
        self.add_module('up_bottleneck', self.up_bottleneck)
        self.add_module('conv1', self.conv1)
        self.add_module('elu1', self.elu1)
        self.add_module('conv2', self.conv2)
        self.add_module('elu2', self.elu2)
        self.add_module('bn1', self.bn1)
        self.add_module('conv_likelihood', self.conv_likelihood)
        self.add_module('conv_predict', self.conv_predict)

    def forward(self, x):
        out24 = self.in_tr(x)
        out48 = self.down_tr48(out24)
        out96 = self.down_tr96(out48)
        out192 = self.down_tr192(out96)
        out = self.down_tr384(out192)
        out = self.bn_bottleneck(out)
        out = self.up_bottleneck(out)
        out = self.up_tr192(out, out192)
        out = self.up_tr96(out, out96)
        out = self.up_tr48(out, out48)
        out = torch.cat((out, out24), 1)
        out = self.bn1(self.elu2(self.conv2(self.elu1(self.conv1(out)))))
        out = self.conv_likelihood(out)
        out = self.conv_predict(out)
        return out
