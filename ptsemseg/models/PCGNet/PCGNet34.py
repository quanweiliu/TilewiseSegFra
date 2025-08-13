
## code by me
"""
# PCGNet
Pyramid-Context Guided Feature Fusion for RGB-D Semantic Segmentation
https://github.com/hmdliu/PCGNet
"""

import torch
from torch import nn
from torchvision import models
# from torchsummary import summary
# from torchsummaryX import summary
from functools import partial
import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DecoderBlock, self).__init__()

        self.conv1=nn.Conv2d(ch_in,ch_in//4,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(ch_in//4)
        self.relu1=nonlinearity

        self.deconv2=nn.ConvTranspose2d(ch_in//4,ch_in//4,kernel_size=4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(ch_in//4)
        self.relu2=nonlinearity

        self.conv3=nn.Conv2d(ch_in//4,ch_out,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(ch_out)
        self.relu3=nonlinearity

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.deconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)

        return x


# Pyramid-Context Guided Fusion Module
class PCGF_Module(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), descriptor=8, mid_feats=16, sp_feats='u'):
        super().__init__()
        # print('[PCGF]: sp = %s, pp = %s, t = %d, m = %d.' % (sp_feats, pp_size, descriptor, mid_feats))

        self.sp_feats = sp_feats
        self.pp_size = pp_size
        self.feats_size = sum([(s ** 2) for s in self.pp_size])
        self.descriptor = descriptor

        # without dim reduction
        if (descriptor == -1) or (self.feats_size < descriptor):
            self.des = nn.Identity()
            self.fc = nn.Sequential(nn.Linear(in_feats * self.feats_size, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))
        # with dim reduction
        else:
            self.des = nn.Conv2d(self.feats_size, self.descriptor, kernel_size=1)
            self.fc = nn.Sequential(nn.Linear(in_feats * descriptor, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))

        self.fc_x = nn.Linear(mid_feats, in_feats)
        self.fc_y = nn.Linear(mid_feats, in_feats)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, ch, _, _ = x.size()
        sp_dict = {'x': x, 'y': y, 'u': x + y}

        pooling_pyramid = []
        for s in self.pp_size:
            l = F.adaptive_avg_pool2d(sp_dict[self.sp_feats], s).view(batch_size, ch, 1, -1)
            pooling_pyramid.append(l)  # [b, c, 1, s^2]
        z = torch.cat(tuple(pooling_pyramid), dim=-1)  # [b, c, 1, f]
        z = z.reshape(batch_size * ch, -1, 1, 1)  # [bc, f, 1, 1]
        z = self.des(z).view(batch_size, -1)  # [bc, d, 1, 1] => [b, cd]
        z = self.fc(z)  # [b, m]

        z_x, z_y = self.fc_x(z), self.fc_y(z)  # [b, c]
        w_x, w_y = self.sigmoid(z_x), self.sigmoid(z_y)  # [b, c]
        rf_x = x * w_x.view(batch_size, ch, 1, 1)  # [b, c, h, w]
        rf_y = y * w_y.view(batch_size, ch, 1, 1)  # [b, c, h, w]
        out_feats = rf_x + rf_y  # [b, c, h, w]

        return out_feats, rf_x, rf_y


# Multi-Level General Fusion Module
class MLGF_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='par', att_setting={}):
        super().__init__()
        module_dict = {
            'se': SE_Block,
            'par': PAR_Block,
            'idt': IDT_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Fuse_Block(in_feats, **fuse_setting)

    def forward(self, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(x, y), x, y


# Attention Refinement Blocks

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x


class PAR_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor  # d: descriptor num (for one channel)

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)  # [b,  c, 1, f]
        y = y.reshape(b * c, f, 1, 1)  # [bc, f, 1, 1]
        y = self.des(y).view(b, c * d)  # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)  # [b,  c, 1, 1] => [b, c, 1, 1]
        return w * x


class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class General_Fuse_Block(nn.Module):
    def __init__(self, in_feats, merge_mode='grp', init=True, civ=1):
        super().__init__()
        merge_dict = {
            'add': Add_Merge(in_feats),
            'cc3': CC3_Merge(in_feats),
            'lma': LMA_Merge(in_feats),
            'grp': nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats)
        }
        self.merge_mode = merge_mode
        self.merge = merge_dict[merge_mode]
        if init and isinstance(self.merge, nn.Conv2d):
            self.merge.weight.data.fill_(civ)

    def forward(self, x, y):
        if self.merge_mode != 'grp':
            return self.merge(x, y)
        b, c, h, w = x.size()
        feats = torch.cat((x, y), dim=-2).reshape(b, 2 * c, h, w)  # [b, c, 2h, w] => [b, 2c, h, w]
        return self.merge(feats)


# Merge Modules

class Add_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y


class LMA_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return x + self.lamb * y


class CC3_Merge(nn.Module):
    def __init__(self, in_feats, *args, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if in_channels < r:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels * r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels * r, in_channels, bias=False),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels // r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels // r, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, rgb, lidar, cross_modal=None):
        # Single tensor for squeeze excitation ops
        if cross_modal is not None:
            x = torch.cat((rgb, lidar, cross_modal), dim=1)
        else:
            x = torch.cat((rgb, lidar), dim=1)
        batch, channels, _, _ = x.shape
        theta = self.squeeze(x).view(batch, channels)
        theta = self.excitation(theta).view(batch, channels, 1, 1)
        # Tensor is weighted with theta
        x = x * theta
        # Channel-wise summation
        if cross_modal is not None:
            rgb_features, dsm_features, cm_features = torch.chunk(x, 3, dim=1)
            x = rgb_features + dsm_features + cm_features
        else:
            rgb_features, dsm_features = torch.chunk(x, 2, dim=1)
            x = rgb_features + dsm_features
        return x

class Attention_Block1(nn.Module):
    def __init__(self, channel):
        super(Attention_Block1, self).__init__()

        # self.seblock = SELayer(channel, channel)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()

        )
        #
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channel, 1, 3, 1, 1, bias=False),
        #     nn.Sigmoid()
        #
        # )

        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.weights = nn.Parameter(torch.randn(2))
        # self.nums = 2
        # self._reset_parameters()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    # def _reset_parameters(self):
    #     init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        # out = self.seblock(x)
        # out = x * out
        master = self.conv1x1(x)
        out = self.conv(x)
        out = out * x
        # finalout = self.weights[0]*out + self.weights[1]*x
        # finalout = self.alpha * out + x
        finalout = out + master
        return finalout

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Cat(nn.Module):
    def __init__(self,channels):
        super(Cat, self).__init__()
        self.channels=channels
        self.conv=nn.Sequential(
            nn.Conv2d(self.channels*2,self.channels,3,1,1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x,y):
        out=torch.cat([x,y],dim=1)
        out=self.conv(out)

        return out

class CAM(nn.Module):
    def __init__(self,channel):
        super(CAM, self).__init__()
        self.channel=channel
        self.conv=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel,self.channel,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        atten = self.conv(x)
        x = x.mul(atten)
        return x


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return s_h,s_w


class PCGNet34(nn.Module):
    def __init__(self, bands1, bands2, n_classes=2, classification="Multi", is_pretrained="ResNet34_Weights.IMAGENET1K_V1"):
        super(PCGNet34, self).__init__()

        filters = [64, 128, 256, 512]  # ResNet34
        reduction=[1, 2, 4, 8, 16]
        resnet = models.resnet34(weights=is_pretrained)

        # rgb-decoder
        # self.rgb_first=nn.Sequential(
        #     nn.Conv2d(4, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(filters[0]),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.rgb_first=nn.Sequential(
            ConvBNReLU(bands1, filters[0],ks=3,stride=1,padding=1),
            ConvBNReLU(filters[0], filters[0], ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0], filters[0], ks=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.rgb_encoder1=resnet.layer1
        self.rgb_encoder2=resnet.layer2
        self.rgb_encoder3=resnet.layer3
        self.rgb_encoder4=resnet.layer4

        # lidar-decoder
        # self.lidar_first=nn.Sequential(
        #     nn.Conv2d(2, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(filters[0]),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.lidar_first=nn.Sequential(
            ConvBNReLU(bands2, filters[0],ks=3,stride=1,padding=1),
            ConvBNReLU(filters[0], filters[0], ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0], filters[0], ks=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.lidar_encoder1 = resnet.layer1
        self.lidar_encoder2 = resnet.layer2
        self.lidar_encoder3 = resnet.layer3
        self.lidar_encoder4 = resnet.layer4

        # cross_block
        self.cross_first=PCGF_Module(filters[0])
        self.cross_block1=PCGF_Module(filters[0])
        self.cross_block2 = PCGF_Module(filters[1])
        self.cross_block3 = PCGF_Module(filters[2])
        # self.cross_block1=Cat(filters[0])
        # self.cross_block2 = Cat(filters[1])
        # self.cross_block3 = Cat(filters[2])


        #Center
        # self.center=DifPyramidBlock(filters[3]) # cahnnel:512
        # self.center=Dblock(filters[3])
        # self.cross_center = Cat(filters[3])
        self.cross_center=PCGF_Module(filters[3])

        self.encoder1_1=MLGF_Module(filters[0])
        self.encoder2_2=MLGF_Module(filters[1])
        self.encoder3_3=MLGF_Module(filters[2])

        # self.attention1=Attention_Block1(filters[0])
        # self.attention2 = Attention_Block1(filters[1])
        # self.attention3 = Attention_Block1(filters[2])

        #decoder
        self.decoder4=DecoderBlock(filters[3],filters[2])

        self.decoder3=DecoderBlock(filters[2],filters[1])

        self.decoder2=DecoderBlock(filters[1],filters[0])

        # self.decoder1=DecoderBlock(filters[0],filters[0])

        # final
        self.final=nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )

        self.classification = classification

    def forward(self,x,y):
        # encoder
        x_first=self.rgb_first(x)
        # xe1=self.rgb_encoder1(x)
        # xe2=self.rgb_encoder2(xe1)
        # xe3=self.rgb_encoder3(xe2)

        y_first=self.lidar_first(y)
        # ye1=self.lidar_encoder1(y)
        # ye2=self.lidar_encoder2(ye1)
        # ye3=self.lidar_encoder3(ye2)

        x_first,_,_=self.cross_first(x_first,y_first)
        xe1=self.rgb_encoder1(x_first)
        ye1=self.lidar_encoder1(y_first)

        xe1,_,_=self.cross_block1(xe1,ye1)
        xe2=self.rgb_encoder2(xe1)
        ye2=self.lidar_encoder2(ye1)

        xe2,_,_=self.cross_block2(xe2,ye2)
        xe3=self.rgb_encoder3(xe2)
        ye3=self.lidar_encoder3(ye2)

        xe3, _,_ = self.cross_block3(xe3, ye3)
        xe4 = self.rgb_encoder4(xe3)
        ye4 = self.lidar_encoder4(ye3)


        ## center
        # center=self.center(xe4+ye4)
        # center=xe4+ye4
        center,_,_ = self.cross_center(xe4, ye4)

        # decoder
        d4=self.decoder4(center)
        d4,_,_=self.encoder3_3(xe3,d4)

        # d3=self.decoder3(self.attention3(e3+d4))
        d3=self.decoder3(d4)
        d3,_,_=self.encoder2_2(xe2,d3)

        # d2 = self.decoder2(self.attention2(e2 + d3))
        d2=self.decoder2(d3)
        d2,_,_=self.encoder1_1(xe1,d2)

        # d1 = self.decoder1(self.attention1(e1 + d2))
        d1=d2
        ## final classification
        out=self.final(d1)

        if self.classification == "Multi":
            return out
        elif self.classification == "Binary":
            return F.sigmoid(out)


if __name__=="__main__":
    # model = SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    bands1 = 16  # gaofen
    bands2 = 3
    x = torch.randn(4, bands1, 128, 128, device=device)
    y = torch.randn(4, bands2, 128, 128, device=device)
    model = PCGNet34(bands1, bands2, n_classes=2, classification="Multi", is_pretrained="ResNet34_Weights.IMAGENET1K_V1").to(device)

    output = model(x, y)
    print("output", output.shape)
    # model=rectification(64,128,128)
    # model=crossFusionMoudle1(64,512,512,reduction=8)
    # summary(model.to(device),x,y)
