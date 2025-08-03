'''
https://github.com/YimianDai/open-aff
cvpr: Attentional Feature Fusion
'''

import torch
import torch.nn as nn
# from torchsummary import summary
# from functools import partial
import torch.nn.functional as F

norm_layer=nn.BatchNorm2d
# nonlinearity = partial(F.relu, inplace=True)


class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class SP(nn.Module):   # end to end
    def __init__(self,ch_in, ch_out, input_size, r_size=1, norm_layer=None):
        super(SP, self).__init__()
        ch_mid = ch_in//4
        H = input_size[0]
        W = input_size[1]

        self.conv1 = nn.Conv2d(ch_in,ch_mid,kernel_size=(3,1),padding=(1,0),bias=False)
        self.bn1 = norm_layer(ch_mid)
        self.up1 = nn.Upsample((H,W),mode='bilinear',align_corners=True)

        self.conv2 = nn.Conv2d(ch_in, ch_mid, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(ch_mid)
        self.up2 = nn.Upsample((H, W), mode='bilinear', align_corners=True)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(ch_mid, ch_out, kernel_size=1, bias=True)
        self.bn3 = norm_layer(ch_out)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool1 = nn.AdaptiveAvgPool2d((H//r_size, r_size))
        self.pool2 = nn.AdaptiveAvgPool2d((r_size, W//r_size))


    def forward(self,x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.up1(x1)

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.up2(x2)

        x3 = self.relu(x1+x2)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        final = self.relu3(x3)

        return final


# class StripPooling(nn.Module):
#     """
#     Reference:
#     """
#     def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
#         super(StripPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
#         self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
#         self.pool3 = nn.AdaptiveAvgPool2d((1, None))
#         self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
#
#         inter_channels = int(in_channels/4)
#         self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#
#
#         self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
#                                 norm_layer(inter_channels))
#
#
#         self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
#                                 norm_layer(in_channels))
#         # bilinear interpolate options
#         self._up_kwargs = up_kwargs
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         x1 = self.conv1_1(x)
#         x2 = self.conv1_2(x)
#         x2_1 = self.conv2_0(x1)
#         x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
#         x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
#         x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
#         x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
#         x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
#         x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
#         out = self.conv3(torch.cat([x1, x2], dim=1))
#         return F.relu_(x + out)


class MSPcam(nn.Module):
    def __init__(self,ch_in,ch_out,input_size,norm_layer=None):
        super(MSPcam, self).__init__()
        ch_mid=ch_in//4
        self.SP1=SP(ch_in,ch_mid,input_size,r_size=1,norm_layer=norm_layer)
        self.SP2=SP(ch_in,ch_mid,input_size,r_size=2,norm_layer=norm_layer)
        self.SP3=SP(ch_in, ch_mid, input_size, r_size=4, norm_layer=norm_layer)
        self.conv1=nn.Conv2d(ch_mid*3,ch_mid,kernel_size=1,bias=False)
        self.bn1=norm_layer(ch_mid)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv=nn.Conv2d(ch_mid,ch_out,kernel_size=1,bias=False)
        self.bn = norm_layer(ch_out)
        # self.relu = nn.ReLU(inplace=False)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_mid, kernel_size=1, stride=1, padding=0),
            norm_layer(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=1, stride=1, padding=0),
            norm_layer(ch_out),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.SP1(x)
        x2=self.SP2(x)
        x3=self.SP3(x)
        xl_cat=torch.cat((x1,x2,x3),dim=1)
        xl=self.relu1(self.bn1(self.conv1(xl_cat)))
        xl=self.bn(self.conv(xl))
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


## author
class SP_ASPP(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(SP_ASPP, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))


        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))


        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


# by me
class Strip_Pooling(nn.Module):
    def __init__(self,in_channels,input_size,r_size=1,up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(Strip_Pooling, self).__init__()
        H=input_size[0]
        W=input_size[1]
        self.pool1 = nn.AdaptiveAvgPool2d((r_size, W // r_size)) # r * W/r
        self.pool2 = nn.AdaptiveAvgPool2d((H//r_size, r_size)) # H/r * r

        if in_channels>=4:
            inter_channels = int(in_channels / 4)
        else:
            inter_channels=in_channels

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, 1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   )
        # self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
        #                            nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        out = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        # out = self.conv5(x4)
        return F.relu_(x + out)  # 将输出的特征与原始输入特征结合



class Muti_SP(nn.Module):
    def __init__(self, channel, input_size):
        super(Muti_SP, self).__init__()
        inter_channel=channel
        self.SP1=Strip_Pooling(inter_channel, input_size,r_size=1,
                               up_kwargs={'mode': 'bilinear', 'align_corners': True})
        self.SP2 = Strip_Pooling(inter_channel, input_size, r_size=3,
                                 up_kwargs={'mode': 'bilinear', 'align_corners': True})
        self.SP3 = Strip_Pooling(inter_channel, input_size, r_size=7,
                                 up_kwargs={'mode': 'bilinear', 'align_corners': True})
        self.conv1=nn.Sequential(
            nn.Conv2d(inter_channel*3, channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(inter_channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sp1 = self.SP1(input)
        sp2 = self.SP2(input)
        sp3 = self.SP3(input)

        out = self.conv1(torch.cat((sp1, sp2, sp3), dim=1))

        wei = self.sigmoid(out)
        final = input * wei

        return final


class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)#结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)#结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))#结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)#将输出的特征与原始输入特征结合



# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     device = torch.device("cuda:0")
#
#     x, residual= torch.ones(8,64, 32, 32).to(device),torch.ones(8,64, 32, 32).to(device)
#     channels=x.shape[1]
#
#     model=AFF(channels=channels)
#     model=model.to(device).train()
#     output = model(x, residual)
#     print(output.shape)

if __name__=='__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(4, 224, 128, 128, device = device)
    y = torch.randn(4, 3, 128, 128, device = device)

    # model=MS_CAM()
    # summary(model.cuda(), (64, 512, 512))

    # model=SP(64,64,(512,512),2,norm_layer)
    # summary(model.cuda(), (64, 512, 512))

    model = Muti_SP(channel = 224, input_size = (128, 128)).to(device)
    output = model(x)
    print("output", output.shape)

    model = Muti_SP(channel = 3, input_size = (128, 128)).to(device)
    output = model(y)
    print("output", output.shape)

    # summary(model.cuda(), (64, 512, 512))


