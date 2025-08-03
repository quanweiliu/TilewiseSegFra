import torch
import torch.nn as nn


class BasicBlock1DConv3(nn.Module):
    def __init__(self, inplanes, planes, bias=False, m=1):
        super(BasicBlock1DConv3, self).__init__()
        ks = m * 2 + 1
        pad = m
        dim_out = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, ks),
                                 stride=1, padding=(0, pad), bias=bias)
        self.conv2_2 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(ks, 1),
                                 stride=1, padding=(pad, 0), bias=bias)
        self.conv2_3 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(ks, 1),
                                 stride=1, padding=(pad, 0), bias=bias)
        self.conv2_4 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, ks),
                                 stride=1, padding=(0, pad), bias=bias)
        self.conv=nn.Sequential(
            nn.Conv2d(planes*2,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x0 = self.conv2(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(self.h_transform(x))
        x3 = self.inv_h_transform(x3)
        x4 = self.conv2_4(self.v_transform(x))
        x4 = self.inv_v_transform(x4)

        x = torch.cat((x0, x1, x2, x3, x4), 1)
        out = self.conv(x)

        out = out + residual
        out = self.relu(out)
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class DecoderBlock6(nn.Module):
    def __init__(self,ch_in,ch_out,m):
        super(DecoderBlock6, self).__init__()
        self.conv=BasicBlock1DConv3(inplanes=ch_in,planes=ch_in,m=m)
        self.conv1=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.up=nn.Sequential(
            nn.ConvTranspose2d(ch_out,ch_out,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.conv(x)
        x = self.up(self.conv1(x))
        return x
    

class Cross_identity(nn.Module):
    def __init__(self):
        super(Cross_identity, self).__init__()

    def forward(self,x,y):
        return x,y

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class CSEBlock(nn.Module):
    def __init__(self,in_channels,ratio=16):
        super(CSEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if in_channels < ratio:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels * ratio, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels * ratio, in_channels, bias=False),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels // ratio, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels // ratio, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self,x,y):
        cat=torch.cat((x, y), dim=1)
        batch, channels, _, _ = cat.shape
        theta = self.squeeze(cat).view(batch, channels)
        theta = self.excitation(theta).view(batch, channels, 1, 1)
        # Tensor is weighted with theta
        cat = cat * theta
        x_features, y_features = torch.chunk(cat, 2, dim=1)
        cat =x_features + y_features
        return cat


#1.门控卷积的模块
class Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.ELU()):
        super(Gated_Conv, self).__init__()
        padding=int(rate*(ksize-1)/2)
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=nn.Conv2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=rate)
        self.activation=activation

    def forward(self,x):
        raw=self.conv(x)
        x1=raw.split(int(raw.shape[1]/2),dim=1)#将特征图分成两半，其中一半是做学习
        gate=torch.sigmoid(x1[0])#将值限制在0-1之间
        out=self.activation(x1[1])*gate
        return out




