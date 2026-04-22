import torch
import torchvision
from torch import nn
import torch.nn.functional as F
# from torchsummaryX import summary


# from gaijin.attention import Attention
# from src.RDSC import decoder_block, decoder_blockc, decoder_blockcc
# from src.GFM import Gated_Fusion


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


class CMGFNet34(nn.Module):

    def __init__(self, bands1, bands2, n_classes, classification="Multi", pretrained="ResNet34_Weights.IMAGENET1K_V1"):
        super().__init__()

        self.num_classes = n_classes
        self.pretrained = pretrained

        # self.pre_conv = ConvBN(224, 3, kernel_size=3)

        # RGB Encoder Part

        self.resnet_features = torchvision.models.resnet34(weights="ResNet34_Weights.DEFAULT")

        self.enc_rgb1 =nn.Sequential(
            nn.Conv2d(bands1, 64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc_rgb2 = nn.Sequential(self.resnet_features.maxpool,
                                      self.resnet_features.layer1)

        self.enc_rgb3 = self.resnet_features.layer2
        self.enc_rgb4 = self.resnet_features.layer3
        self.enc_rgb5 = self.resnet_features.layer4

        # DSM Encoder Part
        self.encoder_depth = torchvision.models.resnet34(weights=self.pretrained)

        # avg = torch.mean(self.encoder_depth.conv1.weight.data, dim=1)
        # avg = avg.unsqueeze(1)
        # conv1d = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # conv1d.weight.data = avg
        # self.encoder_depth.conv1 = conv1d
        #
        # self.enc_dsm1 = nn.Sequential(self.encoder_depth.conv1,
        #                               self.encoder_depth.bn1,
        #                               self.encoder_depth.relu, )
        self.enc_dsm1=nn.Sequential(
            nn.Conv2d(bands2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc_dsm2 = nn.Sequential(self.encoder_depth.maxpool,
                                      self.encoder_depth.layer1)

        self.enc_dsm3 = self.encoder_depth.layer2
        self.enc_dsm4 = self.encoder_depth.layer3
        self.enc_dsm5 = self.encoder_depth.layer4

        self.pool = nn.MaxPool2d(2)

        self.gate5 = Gated_Fusion(16)
        self.gate4 = Gated_Fusion(16)
        self.gate3 = Gated_Fusion(16)
        self.gate2 = Gated_Fusion(16)
        self.gate1 = Gated_Fusion(16)

        self.gate_final = Gated_Fusion(16)

        self.dconv6_rgb = decoder_block(16, 16)
        self.dconv5_rgb = decoder_block(32 , 16)
        self.dconv4_rgb = decoder_block(16 + 16, 16)
        self.dconv3_rgb = decoder_block(16 + 16, 16)
        self.dconv2_rgb = decoder_block(16 + 16, 16)
        self.dconv1_rgb = decoder_block(16 + 16, 16)

        self.side6_rgb = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side5_rgb = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side4_rgb = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.side3_rgb = nn.Conv2d(128, 16, kernel_size=1, padding=0)
        self.side2_rgb = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.side1_rgb = nn.Conv2d(64, 16, kernel_size=1, padding=0)

        self.dconv6_cross = decoder_block(16, 16)
        self.dconv5_cross = decoder_block(48, 16)
        self.dconv4_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv3_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv2_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv1_cross = decoder_block(16 + 16 + 16, 16)

        self.side6_cross = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side5_cross = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side4_cross = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.side3_cross = nn.Conv2d(128, 16, kernel_size=1, padding=0)
        self.side2_cross = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.side1_cross = nn.Conv2d(64, 16, kernel_size=1, padding=0)

        self.Upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.Upsample1 = nn.Upsample(scale_factor=4,mode='bilinear')
        # self.final_fused = nn.Sequential(
        #     nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(self.num_classes),
        #     nn.ReLU(inplace=True),
        # )
        #
        #
        # self.final_rgb = nn.Sequential(
        #     nn.Conv2d(16, self.num_classes, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(self.num_classes),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.final_dsm = nn.Sequential(
        #     nn.Conv2d(16, self.num_classes, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(self.num_classes),
        #     nn.ReLU(inplace=True),
        # )
        self.final_fused = nn.Sequential(
            nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0),
            # nn.BatchNorm2d(self.num_classes),
            # nn.Sigmoid(),
        )

        self.final_rgb = nn.Sequential(
            nn.Conv2d(16, self.num_classes, kernel_size=1, padding=0),
            # nn.BatchNorm2d(self.num_classes),
            # nn.Sigmoid(),
        )

        self.final_dsm = nn.Sequential(
            nn.Conv2d(16, self.num_classes, kernel_size=1, padding=0),
            # nn.BatchNorm2d(self.num_classes),
            # nn.Sigmoid(),
        )

        self.classification = classification

    def forward(self, x_rgb, x_dsm):
        # dsm_encoder

        y1 = self.enc_dsm1(x_dsm)  # bs * 64 * W/2 * H/2
        y1_side = self.side1_cross(y1)
        x1 = self.enc_rgb1(x_rgb)  # bs * 64 * W/2 * H/2
        x1_side = self.side1_rgb(x1)

        ##########################################################
        y2 = self.enc_dsm2(y1)  # bs * 64 * W/4 * H/4
        y2_side = self.side2_cross(y2)
        x2 = self.enc_rgb2(x1)  # bs * 64 * W/4 * H/4
        x2_side = self.side2_rgb(x2)

        ##########################################################
        y3 = self.enc_dsm3(y2)  # bs * 128 * W/8 * H/8
        y3_side = self.side3_cross(y3)
        x3 = self.enc_rgb3(x2)  # bs * 128 * W/8 * H/8
        x3_side = self.side3_rgb(x3)

        ##########################################################
        y4 = self.enc_dsm4(y3)  # bs * 256 * W/16 * H/16
        y4_side = self.side4_cross(y4)
        x4 = self.enc_rgb4(x3)  # bs * 256 * W/16 * H/16
        x4_side = self.side4_rgb(x4)

        ##########################################################
        y5 = self.enc_dsm5(y4)
        y5_side = self.side5_cross(y5)
        x5 = self.enc_rgb5(x4)
        # bs * 512 * W/16 * H/16
        x5_side = self.side5_rgb(x5)
        ##########################################################
        y6 = self.pool(y5)
        y6_side = self.side6_cross(y6)
        out_dsm1 = self.dconv6_cross(y6_side)
        x6 = self.pool(x5)
        x6_side = self.side6_rgb(x6)
        out_rgb1 = self.dconv6_rgb(x6_side)

        ##########################################################
        x6_side2=self.Upsample(x6_side)

        x5_side1 = torch.add(x5_side,x6_side2)
        FG = torch.cat((x5_side1, out_rgb1), dim=1)

        out_rgb2 = self.dconv5_rgb(FG)

        y6_side2 = self.Upsample(y6_side)
        y5_side1 = torch.add(y5_side,y6_side2)

        FG_cross = self.gate5(x5_side1, y5_side1)

        FG_dsm = torch.cat((FG_cross, out_dsm1), dim=1)
        out_dsm2 = self.dconv5_cross(FG_dsm)

        ##########################################################
        x6_side3 = self.Upsample1(x6_side)
        x5_side2 = self.Upsample(x5_side)
        x4_side1 = torch.add(x4_side,x5_side2)
        x4_side1 = torch.add(x4_side1,x6_side3)
        FG = torch.cat((x4_side1, out_rgb2), dim=1)
        out_rgb3 = self.dconv4_rgb(FG)

        y6_side3 = self.Upsample1(y6_side)
        y5_side2 = self.Upsample(y5_side)
        y4_side1 = torch.add(y4_side,y5_side2)
        y4_side1 = torch.add(y4_side1,y6_side3)
        FG_cross = self.gate4(x4_side1, y4_side1)
        FG_dsm = torch.cat((FG_cross, out_dsm2), dim=1)
        out_dsm3 = self.dconv4_cross(FG_dsm)

        ##########################################################
        x5_side3 = self.Upsample1(x5_side)
        x4_side2 = self.Upsample(x4_side)
        x3_side1 = torch.add(x3_side,x4_side2)
        x3_side1 = torch.add(x3_side1,x5_side3)
        FG = torch.cat((x3_side1, out_rgb3), dim=1)
        out_rgb4 = self.dconv3_rgb(FG)

        y5_side3 = self.Upsample1(y5_side)
        y4_side2 = self.Upsample(y4_side)
        y3_side1 = torch.add(y3_side,y4_side2)
        y3_side1 = torch.add(y3_side1,y5_side3)
        FG_cross = self.gate3(x3_side1, y3_side1)
        FG_dsm = torch.cat((FG_cross, out_dsm3), dim=1)
        out_dsm4 = self.dconv3_cross(FG_dsm)

        ##########################################################
        x4_side3 = self.Upsample1(x4_side)
        x3_side2 = self.Upsample(x3_side)
        x2_side1 = torch.add(x2_side,x3_side2)
        x2_side1 = torch.add(x2_side1,x4_side3)
        FG = torch.cat((x2_side1, out_rgb4), dim=1)
        out_rgb5 = self.dconv2_rgb(FG)

        y4_side3 = self.Upsample1(y4_side)
        y3_side2 = self.Upsample(y3_side)
        y2_side1 = torch.add(y2_side, y3_side2)
        y2_side1 = torch.add(y2_side1,y4_side3)
        FG_cross = self.gate2(x2_side1, y2_side1)
        FG_dsm = torch.cat((FG_cross, out_dsm4), dim=1)
        out_dsm5 = self.dconv2_cross(FG_dsm)

        ##########################################################
        x3_side3 = self.Upsample1(x3_side)
        x2_side2 = self.Upsample(x2_side)
        x1_side1 = torch.add(x1_side,x2_side2)
        x1_side1 = torch.add(x1_side1,x3_side3)
        FG = torch.cat((x1_side1, out_rgb5), dim=1)
        out_rgb6 = self.dconv1_rgb(FG)

        y3_side3 = self.Upsample1(y3_side)
        y2_side2 = self.Upsample(y2_side)
        y1_side1 = torch.add(y1_side, y2_side2)
        y1_side1 = torch.add(y1_side1,y3_side3)
        FG_cross = self.gate1(x1_side1, y1_side1)
        FG_dsm = torch.cat((FG_cross, out_dsm5), dim=1)
        out_dsm6 = self.dconv1_cross(FG_dsm)

        ##########################################################
        final_fused = self.gate_final(out_rgb6, out_dsm6)
        final_fused = self.final_fused(final_fused)

        final_rgb = self.final_rgb(out_rgb6)
        final_dsm = self.final_dsm(out_dsm6)

        if self.classification == "Multi":
            return final_fused, final_rgb, final_dsm
        elif self.classification == "Binary":
            final_fused = F.sigmoid(final_fused)
            final_rgb = F.sigmoid(final_rgb)
            final_dsm = F.sigmoid(final_dsm)
            return final_fused, final_rgb, final_dsm

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights



class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class Attention_block(nn.Module):
    def __init__(self, F_c, F_de,  reduction=16, concat=True):
        super(Attention_block,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(F_c, F_c//reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 =  nn.Conv2d(F_c//reduction, F_de, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()


        self.spatial_se = nn.Sequential(nn.Conv2d(F_de, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self,f, x):
        f=self.avg_pool(f)
        f = self.fc1(f)
        f = self.relu(f)
        f = self.fc2(f)
        chn_se = self.sigmoid(f)
        chn_se = chn_se * x

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se

        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class decoder_block(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(decoder_block, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

        self.decode = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.BatchNorm2d(input_channels),
            depthwise_separable_conv(input_channels, input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            depthwise_separable_conv(input_channels, output_channels),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)

        out = self.decode(x)

        out += residual

        return out


class Gated_Fusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return torch.cat([FG, PG], dim=1)


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # x = torch.randn(4, 3, 512, 512, device=device)
    # y = torch.randn(4, 2, 512, 512, device=device)
    bands1 = 3 
    bands2 = 1
    classes = 2
    x = torch.randn(4, bands1, 128, 128, device=device)
    y = torch.randn(4, bands2, 128, 128, device=device)
    model = CMGFNet34(bands1=bands1, bands2=bands2, n_classes=classes, pretrained="ResNet34_Weights.IMAGENET1K_V1")
    # model=CRFN_CMF(n_classes=2,is_pretrained=True)
    # model = CRFN_HFM(n_classes=2, is_pretrained=True)
    # model=rectification(64,128,128)
    # model=crossFusionMoudle1(64,512,512,reduction=8)
    model = model.to(device)
    final_fused, final_rgb, final_dsm = model(x, y)
    print("model", final_fused.shape, final_rgb.shape, final_dsm.shape)
    # summary(model,x,y)