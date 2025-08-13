import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import collections
import time
from PIL import Image, ImageFilter
# from torchsummaryX import summary
from torchvision import transforms
import torch.utils.model_zoo as model_zoo

from .net_util import SAGate
# from net_util import SAGate
# from .backbone import VisualizeFlow,  VisualizeFeatureMapPCA

# __all__ = ['DualResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    #mmcv resnet101_v1c  mmcv/model_zoo/open_mmlab.json:
    'resnet101_v1c': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


class DualBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(DualBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.hha_conv1 = conv3x3(inplanes, planes, stride)
        self.hha_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_conv2 = conv3x3(planes, planes)
        self.hha_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.downsample = downsample
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]

        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)

        out1 += residual1
        out2 += residual2

        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class DualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(DualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.hha_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.hha_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.hha_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                   bias=False)
        self.hha_bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                                  momentum=bn_momentum)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv3(out2)
        out2 = self.hha_bn3(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)
        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class DualResNet(nn.Module):
    def __init__(self, bands1, bands2, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        super(DualResNet, self).__init__()
        
        self.inplanes = stem_width * 2 if deep_stem else 64
        
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(bands1, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
            self.hha_conv1 = nn.Sequential(
                nn.Conv2d(bands2, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(bands1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.hha_conv1 = nn.Conv2d(bands2, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.hha_bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                                  momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hha_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

        self.sagates = nn.ModuleList([
            SAGate(in_planes=256, out_planes=256, bn_momentum=bn_momentum),
            SAGate(in_planes=512, out_planes=512, bn_momentum=bn_momentum),
            SAGate(in_planes=1024, out_planes=1024, bn_momentum=bn_momentum),
            SAGate(in_planes=2048, out_planes=2048, bn_momentum=bn_momentum)
        ])

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.hha_conv1(x2)
        x2 = self.hha_bn1(x2)
        x2 = self.hha_relu(x2)
        x2 = self.hha_maxpool(x2)

        x = [x1, x2]
        blocks = []
        merges = []
        x = self.layer1(x)
        x, merge = self.sagates[0](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer2(x)
        x, merge = self.sagates[1](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer3(x)
        x, merge = self.sagates[2](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer4(x)
        x, merge = self.sagates[3](x)
        blocks.append(x)
        merges.append(merge)

        return blocks, merges


def load_dualpath_model(model, is_restore=False):
    # load raw state_dict

    model_url = model_urls['resnet101_v1c']
    cache_dir = 'pretrains'
    
    raw_state_dict = model_zoo.load_url(model_url, model_dir=cache_dir)
    # if isinstance(model_file, str):
    #     raw_state_dict = torch.load(model_file)

        # if 'model' in raw_state_dict.keys():
        #     raw_state_dict = raw_state_dict['model']
    # else:
    #     raw_state_dict = model_file
    # copy to hha backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('conv1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv1', 'hha_conv1')] = v
        if k.find('conv2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv2', 'hha_conv2')] = v
        if k.find('conv3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv3', 'hha_conv3')] = v
        if k.find('bn1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn1', 'hha_bn1')] = v
        if k.find('bn2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn2', 'hha_bn2')] = v
        if k.find('bn3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn3', 'hha_bn3')] = v
        if k.find('downsample') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample', 'hha_downsample')] = v

    # ## 删除第一个卷积预训练权重
    # del state_dict['conv1.0.weight']
    # del state_dict['hha_conv1.0.weight']
    # print(state_dict["state_dict"].keys())

    if is_restore:
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict

    return model


def dual_resnet101(bands1, bands2, pretrained_model=None, **kwargs):
    model = DualResNet(bands1, bands2, DualBottleneck, [3, 4, 23, 3], **kwargs)

    # print(pretrained_model)
    if pretrained_model is not None:
        model = load_dualpath_model(model)
    return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid(),
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(in_planes*2, out_planes, reduction)

    def forward(self, x):
        rgb, hha = x
        channel_weight = self.filter(torch.cat([rgb, hha], dim=1))
        out = channel_weight * (rgb+hha)
        return out


class PSPModule(nn.Module):

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features,
                      kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(
            h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]

        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class RGBDAlignModule(nn.Module):
    def __init__(self, inplane):
        super(RGBDAlignModule, self).__init__()
        self.flow_make = nn.Conv2d(
            inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make2 = nn.Conv2d(
            inplane * 2, 2, kernel_size=3, padding=1, bias=False)

        self.fsp1 = FSP(inplane, inplane)
        self.fsp2 = FSP(inplane, inplane)
        self.count = 0

    def forward(self, x):
        rgb, hha = x
        size = rgb.size()[2:]

        rgb_merged_feat = self.fsp1([rgb, hha])
        hha_merged_feat = self.fsp2([rgb, hha])
        flow2 = self.flow_make2(
            torch.cat([rgb_merged_feat, hha_merged_feat], 1))
        hha_merged_feat = self.flow_warp(hha_merged_feat, flow2, size=size)

        rgb = (rgb+rgb_merged_feat)/2
        hha = (hha+hha_merged_feat)/2

        return [rgb, hha], (rgb_merged_feat+hha_merged_feat)

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size
        n, c, h, w = inputs.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(
            inputs).to(inputs.device)

        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)

        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(inputs, grid, align_corners=False)
        return output


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(
            outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_origin = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size,
                                  mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))

        h_feature = self.flow_warp(h_feature_origin, flow, size=size)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(
            input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.ppm = PSPModule(
            inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3,
                                  stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1,
                                  stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class TestNet(nn.Module):
    def __init__(self, backbone, num_classes, classification="Multi", fpn_dsn=False):
        super(TestNet, self).__init__()
        self.backbone = backbone
        self.dilate = 2

        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2
        self.backbone.sagates = nn.ModuleList([
            RGBDAlignModule(256),
            RGBDAlignModule(512),
            RGBDAlignModule(1024),
            RGBDAlignModule(2048),
        ])
        self.head = UperNetAlignHead(
            2048, num_class=num_classes, norm_layer=nn.BatchNorm2d, fpn_dsn=fpn_dsn)

        self.classification = classification

    def forward(self, rgb, hha):
        b, c, h, w = rgb.shape
        blocks, merges = self.backbone(rgb, hha)
        pred = self.head(merges)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.classification == "Multi":
            return pred
        elif self.classification == "Single":
            return torch.sigmoid(pred)

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_params(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key or 'sagates' in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [bb_lr, nbb_lr]
        return params


def FAFNet(bands1, bands2, num_classes, classification, pretrained=True):
    # bkb=None
    # if pretrained == True:
    #     bkb = '/home/wzj/PycharmProjects/multi_road_extraction/pretrained/resnet101_v1c.pth'
    backbone = dual_resnet101(bands1, 
                              bands2, 
                              pretrained_model=pretrained, 
                              norm_layer=nn.BatchNorm2d,
                              bn_eps=1e-5,
                              bn_momentum=0.1,
                              deep_stem=True, 
                              stem_width=64)
    model = TestNet(backbone, num_classes, classification)
    return model


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bands1 = 4
    bands2 = 2
    x = torch.randn(4, bands1, 512, 512, device=device)
    y = torch.randn(4, bands2, 512, 512, device=device)

    model = FAFNet(bands1, bands2, num_classes=1, pretrained=True)
    model = model.to(device)
    output = model(x, y)
    print(output.shape)
    # summary(model.to(device), x,y)
