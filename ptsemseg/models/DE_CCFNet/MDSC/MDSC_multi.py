import torch
from torch import nn
from torchvision import models
# from torchsummary import summary
from torchsummaryX import summary
import torch.nn.functional as F

from ptsemseg.models.DE_CCFNet.CMFs.CMF_re6 import CMF_re6, CMF_re6_2
# from ptsemseg.models.DE_CCFNet.decoder import DecoderBlock, DecoderBlock6
from ptsemseg.models.DE_CCFNet.utils import ConvBNReLU, DecoderBlock6


class DE_CCFNet_34_multi(nn.Module):
    def __init__(self,n_classes=2,is_pretrained=True):
        super(DE_CCFNet_34_multi, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet34
        reduction = [1, 2, 4, 8, 16]
        rgb_resnet = models.resnet34(pretrained=is_pretrained)
        lidar_resnet = models.resnet34(pretrained=is_pretrained)

        self.rgb_first = nn.Sequential(
            ConvBNReLU(4, filters[0] // 2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
        )

        self.rgb_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            rgb_resnet.layer1
        )

        self.rgb_encoder2 = rgb_resnet.layer2
        self.rgb_encoder3 = rgb_resnet.layer3
        self.rgb_encoder4 = rgb_resnet.layer4

        self.lidar_first = nn.Sequential(
            ConvBNReLU(2, filters[0] // 2, ks=3, stride=2, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
            ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.lidar_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            lidar_resnet.layer1
        )
        self.lidar_encoder2 = lidar_resnet.layer2
        self.lidar_encoder3 = lidar_resnet.layer3
        self.lidar_encoder4 = lidar_resnet.layer4

        # cross_block
        self.cross_block0 = CMF_re6(filters[0],filters[0],r=8)
        self.cross_block1 = CMF_re6(filters[0],filters[0],r=8)
        # self.cross_block0 = CMF_sp(filters[0],r=8)  #(*,64,256,256)
        # self.cross_block1 = CMF_sp(filters[0],r=8)  #(*,64,128,128)
        self.cross_block2 = CMF_re6(filters[1],filters[0],r=16)
        self.cross_block3 = CMF_re6(filters[2],filters[1],r=16)

        # Center
        # self.center = nn.Sequential(
        #     ConvBNReLU(filters[3]*2,filters[3]),
        #     TwofoldGCN(filters[3],filters[3],filters[3])
        # )
        self.center = CMF_re6(filters[3],filters[2], r=16)
        # self.center=CCAM_Module(filters[3])

        # decoder
        self.decoder4 = DecoderBlock6(filters[3], filters[2],m=4)
        # self.att4 = Attention_Block(filters[2])

        self.decoder3 = DecoderBlock6(filters[2], filters[1],m=3)
        # self.att3=Attention_Block(filters[1])

        self.decoder2 = DecoderBlock6(filters[1], filters[0],m=2)
        # self.att2 = Attention_Block(filters[0])

        self.decoder1 = DecoderBlock6(filters[0], filters[0],m=1)
        # self.att1 = Attention_Block(filters[0])

        # final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(32, n_classes - 1, 3, padding=1),
        )

        # self.final = nn.Sequential(
        #     nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.1),
        #     nn.Conv2d(32, n_classes - 1, 3, padding=1),
        # )

        # self._initalize_weights()

    def _initalize_weights(self):
        init_set={nn.Conv2d,nn.ConvTranspose2d,nn.Linear}
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
        xe1, ye1, e1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first),e_first)
        xe2, ye2, e2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1),e1)
        xe3, ye3, e3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2),e2)

        ## center
        _,_,center = self.center(self.rgb_encoder4(xe3), self.lidar_encoder4(ye3),e3)
        # center=self.center(torch.cat([self.rgb_encoder4(xe3),self.lidar_encoder4(ye3)],dim=1))

        d4 = self.decoder4(center)
        # d4 = self.att4(d4 + e3)

        # d3 = self.decoder3(d4)
        # d3=self.att3(d3+e2)
        d3 = self.decoder3(d4+e3)

        # d2 = self.decoder2(d3)
        # d2 = self.att2(d2 + e1)
        d2=self.decoder2(d3+e2)

        # d1 = self.decoder1(d2)
        # d1 = self.att1(d1)
        d1=self.decoder1(d2+e1)

        ## final classification
        # out = self.final(d1)
        out = self.final(d1)  ## new

        return F.sigmoid(out)


class DE_CCFNet_34_multi_2(nn.Module):
    def __init__(self,n_classes=2,is_pretrained=True):
        super(DE_CCFNet_34_multi_2, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet34
        reduction = [1, 2, 4, 8, 16]
        rgb_resnet = models.resnet34(pretrained=is_pretrained)
        lidar_resnet = models.resnet34(pretrained=is_pretrained)

        # self.rgb_first = nn.Sequential(
        #     ConvBNReLU(4, filters[0] // 2, ks=3, stride=2, padding=1),
        #     ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
        #     ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
        # )
        self.rgb_first = ConvBNReLU(4, filters[0], ks=7, stride=2, padding=3)

        self.rgb_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            rgb_resnet.layer1
        )

        self.rgb_encoder2 = rgb_resnet.layer2
        self.rgb_encoder3 = rgb_resnet.layer3
        self.rgb_encoder4 = rgb_resnet.layer4

        # self.lidar_first = nn.Sequential(
        #     ConvBNReLU(2, filters[0] // 2, ks=3, stride=2, padding=1),
        #     ConvBNReLU(filters[0] // 2, filters[0] // 2, ks=3, stride=1, padding=1),
        #     ConvBNReLU(filters[0] // 2, filters[0], ks=3, stride=1, padding=1),
        #     # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.lidar_first = ConvBNReLU(2, filters[0], ks=7, stride=2, padding=3)

        self.lidar_encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            lidar_resnet.layer1
        )
        self.lidar_encoder2 = lidar_resnet.layer2
        self.lidar_encoder3 = lidar_resnet.layer3
        self.lidar_encoder4 = lidar_resnet.layer4

        # cross_block
        self.cross_block0 = CMF_re6(filters[0],filters[0],r=8)
        self.cross_block1 = CMF_re6(filters[0],filters[0],r=8)
        # self.cross_block0 = CMF_sp(filters[0],r=8)  #(*,64,256,256)
        # self.cross_block1 = CMF_sp(filters[0],r=8)  #(*,64,128,128)
        self.cross_block2 = CMF_re6(filters[1],filters[0],r=16)
        self.cross_block3 = CMF_re6(filters[2],filters[1],r=16)

        # Center
        # self.center = nn.Sequential(
        #     ConvBNReLU(filters[3]*2,filters[3]),
        #     TwofoldGCN(filters[3],filters[3],filters[3])
        # )
        self.center = CMF_re6(filters[3],filters[2], r=16)
        # self.center=CCAM_Module(filters[3])

        # decoder
        self.decoder4 = DecoderBlock6(filters[3], filters[2],m=4)
        # self.att4 = Attention_Block(filters[2])

        self.decoder3 = DecoderBlock6(filters[2], filters[1],m=3)
        # self.att3=Attention_Block(filters[1])

        self.decoder2 = DecoderBlock6(filters[1], filters[0],m=2)
        # self.att2 = Attention_Block(filters[0])

        self.decoder1 = DecoderBlock6(filters[0], filters[0],m=1)
        # self.att1 = Attention_Block(filters[0])

        # final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(32, n_classes - 1, 3, padding=1),
        )

        # self.final = nn.Sequential(
        #     nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.1),
        #     nn.Conv2d(32, n_classes - 1, 3, padding=1),
        # )

        # self._initalize_weights()

    def _initalize_weights(self):
        init_set={nn.Conv2d,nn.ConvTranspose2d,nn.Linear}
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
        xe1, ye1, e1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first),e_first)
        xe2, ye2, e2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1),e1)
        xe3, ye3, e3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2),e2)

        ## center
        _,_,center = self.center(self.rgb_encoder4(xe3), self.lidar_encoder4(ye3),e3)
        # center=self.center(torch.cat([self.rgb_encoder4(xe3),self.lidar_encoder4(ye3)],dim=1))

        d4 = self.decoder4(center)
        # d4 = self.att4(d4 + e3)

        # d3 = self.decoder3(d4)
        # d3=self.att3(d3+e2)
        d3 = self.decoder3(d4+e3)

        # d2 = self.decoder2(d3)
        # d2 = self.att2(d2 + e1)
        d2=self.decoder2(d3+e2)

        # d1 = self.decoder1(d2)
        # d1 = self.att1(d1)
        d1=self.decoder1(d2+e1)

        ## final classification
        # out = self.final(d1)
        out = self.final(d1)  ## new

        return F.sigmoid(out)




if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # x = torch.randn(1, 64,256,256, device=device)
    # y = torch.randn(1,64,256,256,device=device)
    x = torch.randn(4, 4,512,512, device=device)
    y = torch.randn(4,2,512,512,device=device)
    model =DE_CCFNet_34_multi(n_classes=2,is_pretrained=True)
    # model=SKBlock(64,M=2,G=64)
    # model=cfm(64)
    summary(model.to(device), x,y)






