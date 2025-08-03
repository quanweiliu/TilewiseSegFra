import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, \
                 fist_dilation=1, multi_grid=1, bn_momentum=0.0003):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    
    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x, start_module=1, end_module=5):
        if start_module <= 1:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            start_module = 2
        features = []
        for i in range(start_module, end_module+1):
            x = eval('self.layer%d'%(i-1))(x)
            features.append(x)

        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x

def get_resnet50(dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False, is_pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    
    if is_pretrained:
        # 定义模型URL和缓存目录
        model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        cache_dir = 'pretrains' # 模型将被下载到项目根目录下的 'pretrains' 文件夹

        # 从URL加载预训练权重，如果本地已存在则直接读取
        pretrained_dict = model_zoo.load_url(model_url, model_dir=cache_dir)

        # 获取你当前模型的权重字典
        model_dict = model.state_dict()

        # 过滤掉预训练权重中 key 为 'conv1.weight' 的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['conv1.weight']}

        # 合并权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('loadding pretrained model:ResNet50')

    # del model.avgpool
    # del model.fc
    return model


def get_resnet101(dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False, is_pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    
    if is_pretrained:
        # 定义模型URL和缓存目录
        model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        cache_dir = 'pretrains' # 模型将被下载到项目根目录下的 'pretrains' 文件夹

        # 从URL加载预训练权重，如果本地已存在则直接读取
        pretrained_dict = model_zoo.load_url(model_url, model_dir=cache_dir)

        # 获取你当前模型的权重字典
        model_dict = model.state_dict()

        # 过滤掉预训练权重中 key 为 'conv1.weight' 的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['conv1.weight']}
        print("pretrained_dict keys:", pretrained_dict.keys())

        # 合并权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('loadding pretrained model:ResNet101')

    # del model.avgpool
    # del model.fc
    return model


if __name__ == '__main__':
    # net = get_resnet50()
    # x = torch.randn(4, 3, 128, 128)
    # print(net(x).shape)

    net = get_resnet101()
    # x = torch.randn(4, 3, 128, 128)
    x = torch.randn(4, 3, 256, 256)
    # x = torch.randn(4, 3, 512, 512)
    print(net(x).shape)
