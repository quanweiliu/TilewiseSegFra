import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .CMFs.CMF_re6 import CMF_re6, CMF_re6_2
from .utils import ConvBNReLU, DecoderBlock6

# from CMFs.CMF_re6 import CMF_re6
# from utils import ConvBNReLU, DecoderBlock6
# # from decoder import DecoderBlock, DecoderBlock6


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, 
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, \
                      kernel_size=kernel_size, bias=bias, \
                      dilation=dilation, stride=stride, \
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class DE_CCFNet18(nn.Module):
    def __init__(self, bands1=16, bands2=3, n_classes=1, classification="Multi", is_pretrained=False):
        super(DE_CCFNet18, self).__init__()

        filters = [64, 128, 256, 512]  # ResNet34
        # reduction = [1, 2, 4, 8, 16]
        if is_pretrained:
            rgb_resnet = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
            lidar_resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        else:
            rgb_resnet = models.resnet18(weights=None)
            lidar_resnet = models.resnet18(weights=None)

        self.rgb_first = nn.Sequential(
            ConvBNReLU(bands1, filters[0] // 2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
        )

        self.rgb_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            rgb_resnet.layer1,
        )

        self.rgb_encoder2 = rgb_resnet.layer2
        self.rgb_encoder3 = rgb_resnet.layer3
        self.rgb_encoder4 = rgb_resnet.layer4

        self.lidar_first = nn.Sequential(
            ConvBNReLU(bands2, filters[0] // 2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.lidar_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            lidar_resnet.layer1,
        )
        self.lidar_encoder2 = lidar_resnet.layer2
        self.lidar_encoder3 = lidar_resnet.layer3
        self.lidar_encoder4 = lidar_resnet.layer4

        # cross_block
        self.cross_block0 = CMF_re6(filters[0], filters[0], r=8)
        self.cross_block1 = CMF_re6(filters[0], filters[0], r=8)
        # self.cross_block0 = CMF_sp(filters[0],r=8)  #(*,64,256,256)
        # self.cross_block1 = CMF_sp(filters[0],r=8)  #(*,64,128,128)
        self.cross_block2 = CMF_re6(filters[1], filters[0], r=16)
        self.cross_block3 = CMF_re6(filters[2], filters[1], r=16)

        # Center
        # self.center = nn.Sequential(
        #     ConvBNReLU(filters[3]*2,filters[3]),
        #     TwofoldGCN(filters[3],filters[3],filters[3])
        # )
        self.center = CMF_re6(filters[3], filters[2], r=16)
        # self.center=CCAM_Module(filters[3])

        # decoder
        self.decoder4 = DecoderBlock6(filters[3], filters[2], m=4)
        # self.att4 = Attention_Block(filters[2])

        self.decoder3 = DecoderBlock6(filters[2], filters[1], m=3)
        # self.att3=Attention_Block(filters[1])

        self.decoder2 = DecoderBlock6(filters[1], filters[0], m=2)
        # self.att2 = Attention_Block(filters[0])

        self.decoder1 = DecoderBlock6(filters[0], filters[0], m=1)
        # self.att1 = Attention_Block(filters[0])

        # final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )
        self.classification = classification

        # self.final = nn.Sequential(
        #     nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.1),
        #     nn.Conv2d(32, n_classes, 3, padding=1),
        # )

        # self._initalize_weights()

    def _initalize_weights(self):
        init_set = {nn.Conv2d,nn.ConvTranspose2d,nn.Linear}
        for module in self.modules():
            if type(module) in init_set:
                nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, y):
        # encoder
        x_first, y_first, e_first = self.cross_block0(self.rgb_first(x), self.lidar_first(y))
        xe1, ye1, e1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first), e_first)
        xe2, ye2, e2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1), e1)
        xe3, ye3, e3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2), e2)

        ## center
        _, _, center = self.center(self.rgb_encoder4(xe3), self.lidar_encoder4(ye3), e3)
        # center=self.center(torch.cat([self.rgb_encoder4(xe3),self.lidar_encoder4(ye3)],dim=1))

        d4 = self.decoder4(center)
        # d4 = self.att4(d4 + e3)

        # d3 = self.decoder3(d4)
        # d3=self.att3(d3+e2)
        d3 = self.decoder3(d4 + e3)

        # d2 = self.decoder2(d3)
        # d2 = self.att2(d2 + e1)
        d2 = self.decoder2(d3 + e2)

        # d1 = self.decoder1(d2)
        # d1 = self.att1(d1)
        d1 = self.decoder1(d2 + e1)

        ## final classification
        # out = self.final(d1)
        out = self.final(d1)  ## new

        if self.classification == "Multi":
            return out
        elif self.classification == "Binary":
            return F.sigmoid(out)


if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    bands1 = 16  # gaofen
    bands2 = 3
    x = torch.randn(4, bands1, 256, 256, device=device)
    y = torch.randn(4, bands2, 256, 256, device=device)

    model = DE_CCFNet18(bands1=bands1, bands2=bands2, n_classes=20, is_pretrained=True).to(device)
    output = model(x, y)
    print("output", output.shape)
    # model = SKBlock(64,M=2,G=64)
    # model = cfm(64)
    # summary(model.to(device), x,y)


