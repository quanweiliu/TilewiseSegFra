'''
first-conv: 3* 3x3 conv
layer:1-3
cross: identity
skip_connection: +
center: +
'''

import torch
from torch import nn
from torchvision import models
# from torchsummary import summary
# from torchsummaryX import summary
from functools import partial
import torch.nn.functional as F

# from ptsemseg.models.Baseline.decoder import DecoderBlock
# from ptsemseg.models.Baseline.decoder import decoder
# from ptsemseg.models.Baseline.utils import ConvBNReLU

from .decoder_zoos import DecoderBlock
from .utils import ConvBNReLU

# from decoder_zoos import DecoderBlock
# from utils import ConvBNReLU

nonlinearity = partial(F.relu, inplace=True)


class Cross_identity(nn.Module):
    def __init__(self):
        super(Cross_identity, self).__init__()

    def forward(self,x,y):
        return x,y


class Resnet_base34_double(nn.Module):
    def __init__(self, bands1, bands2, n_classes=2, classification="Multi", is_pretrained="ResNet34_Weights.DEFAULT"):
        super(Resnet_base34_double, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet34
        rgb_resnet = models.resnet34(weights=is_pretrained)
        lidar_resnet = models.resnet34(weights=is_pretrained)

        # rgb-decoder
        self.rgb_first = nn.Sequential(
            ConvBNReLU(bands1, filters[0]//2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0]//2, filters[0]//2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0]//2, filters[0], ks=3, stride=1, padding=1),
        )

        self.rgb_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            rgb_resnet.layer1
        )
        self.rgb_encoder2 = rgb_resnet.layer2
        self.rgb_encoder3 = rgb_resnet.layer3
        self.rgb_encoder4 = rgb_resnet.layer4

        # lidar-decoder
        self.lidar_first = nn.Sequential(
            ConvBNReLU(bands2, filters[0]//2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0]//2, filters[0]//2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0]//2, filters[0], ks=3, stride=1, padding=1),
        )

        self.lidar_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            lidar_resnet.layer1
        )
        self.lidar_encoder2 = lidar_resnet.layer2
        self.lidar_encoder3 = lidar_resnet.layer3
        self.lidar_encoder4 = lidar_resnet.layer4

        # cross_block
        self.cross_block0 = Cross_identity()
        self.cross_block1 = Cross_identity()
        self.cross_block2 = Cross_identity()
        self.cross_block3 = Cross_identity()
        # decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )

        self.classification = classification

        self._initalize_weights()

    def _initalize_weights(self):
        init_set = {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}
        for module in self.modules():
            if type(module) in init_set:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, y):
        # encoder
        x_first, y_first = self.cross_block0(self.rgb_first(x), self.lidar_first(y))
        xe1, ye1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first))
        xe2, ye2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1))
        xe3, ye3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2))

        xe4 = self.rgb_encoder4(xe3)
        ye4 = self.lidar_encoder4(ye3)

        ## center
        center = xe4 + ye4

        # decoder
        d4 = self.decoder4(center)
        e3 = xe3 + ye3
        d3 = self.decoder3(e3 + d4)
        e2 = xe2 + ye2
        d2 = self.decoder2(e2 + d3)
        e1 = xe1 + ye1
        d1 = self.decoder1(e1 + d2)

        ## final classification
        out = self.final(d1)

        if self.classification == "Multi":
            return out
        elif self.classification == "Binary":
            return F.sigmoid(out)


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bands1 = 224
    bands2 = 3
    x = torch.randn(4, bands1, 256, 256, device=device)
    y = torch.randn(4, bands2, 256, 256, device=device)

    model = Resnet_base34_double(bands1, bands2, n_classes=1, classification="Multi").to(device)
    output = model(x, y)
    print("output", output.shape)

