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

from .decoder_zoos import DecoderBlock
from .utils import ConvBNReLU

# from decoder_zoos import DecoderBlock
# from utils import ConvBNReLU

nonlinearity = partial(F.relu, inplace=True)


class CRFN_base18_single(nn.Module):
    def __init__(self, bands, n_classes=2, is_pretrained="ResNet18_Weights.DEFAULT"):
        super(CRFN_base18_single, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet18
        resnet = models.resnet18(weights=is_pretrained)

        # rgb and lidar -decoder
        self.first = nn.Sequential(
            ConvBNReLU(bands, filters[0]//2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0]//2, filters[0]//2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0]//2, filters[0], ks=3, stride=1, padding=1),
        )

        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet.layer1
        )
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # decoder
        self.decoder4 = DecoderBlock(filters[3],filters[2])
        self.decoder3 = DecoderBlock(filters[2],filters[1])
        self.decoder2 = DecoderBlock(filters[1],filters[0])
        self.decoder1 = DecoderBlock(filters[0],filters[0])

        # final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )

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

    def forward(self, x):
        # encoder
        x_first = self.first(x)
        e1 = self.encoder1(x_first)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        ## center
        # center = xe3+ye3
        center = e4

        # decoder
        d4 = self.decoder4(center)
        d3 = self.decoder3(e3 + d4)
        d2 = self.decoder2(e2 + d3)
        d1 = self.decoder1(e1 + d2)

        ## final classification
        # print("d1", d1.shape)   # [32, 64, 64, 64
        out = self.final(d1)

        return F.sigmoid(out)


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bands = 193
    x = torch.randn(4, bands, 128, 128, device=device)

    model = CRFN_base18_single(bands).to(device)
    output = model(x)
    print("output", output.shape)

