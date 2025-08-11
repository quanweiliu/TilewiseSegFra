import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet34_Weights


__all__ = ["ResNet50", "ResNet34"]


def ResNet50():
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return resnet50


def ResNet34():
    resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    return resnet34


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True, ):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm, )
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm, )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        # X:{tensor:10, 512, 16, 16} --> {tensor:10, 512, 32, 32}
        if skip is not None:
            # skip[0, 1, 2] [0]{tensor:10, 512, 32, 32}, [1]{tensor:10, 256, 64, 64}, [2]{tensor:10, 64, 128, 128}
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_skip = 3
        self.conv_more = Conv2dReLU(1024, 512, kernel_size=3, padding=1, use_batchnorm=True, )
        in_channels = [512, 256, 128, 64]
        out_channels = [256, 128, 64, 16]
        skip_channels = [512, 256, 64, 0]

        if self.n_skip != 0:
            for i in range(3 - self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # # hidden_states:{tensor:10, 256, 768}
        # # features[0, 1, 2] [0]{tensor:10, 512, 32, 32}, [1]{tensor:10, 256, 64, 64}, [2]{tensor:10, 64, 128, 128}
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # # X:{tensor:10, 512, 16, 16}
        # x = x.contiguous().view(B, hidden, h, w)
        x = hidden_states
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# multiscale gated fusion
class MGF(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(MGF, self).__init__()


        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # multiscale attention
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            fusion = w_d1 * input_rgb + w_d2 * input_dsm

        return fusion


class SPALayer(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SPALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = nn.parameter.Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7, 7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7, 7]).unsqueeze(dim=1)],
            dim=1)
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return x * y


# cross-modal interaction
class CMI(nn.Module):
    def __init__(self, in_dim, num_groups=8):
        super(CMI, self).__init__()
        self.num_groups = num_groups
        self.group_channels = in_dim // num_groups

        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rgb, input_dsm):
        batch_size, C_rgb, H, W = input_rgb.size()
        group_channels = C_rgb // self.num_groups

        # RGB branch
        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        value_rgb = self.value_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        # DSM branch
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        value_dsm = self.value_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_rgb_to_dsm = torch.matmul(query_rgb, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb_to_dsm = self.softmax(attn_rgb_to_dsm)

        out_dsm = torch.matmul(attn_rgb_to_dsm, value_dsm.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)

        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]

        attn_dsm_to_rgb = torch.matmul(query_dsm, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm_to_rgb = self.softmax(attn_dsm_to_rgb)

        out_rgb = torch.matmul(attn_dsm_to_rgb, value_rgb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        # Apply residual connection
        out_rgb = self.gamma_rgb * out_rgb + input_rgb
        out_dsm = self.gamma_dsm * out_dsm + input_dsm

        return out_rgb, out_dsm


# cross-modal multiscale extraction
class CMME(nn.Module):
    def __init__(self, dim, in_dim):
        super(CMME, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        # Cross semantic Guide Model
        self.num_groups = 8
        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        # Cross semantic Guide Model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        x = self.down_conv(fusion)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)

        energy2 = torch.bmm(proj_query3, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        energy3 = torch.bmm(proj_query4, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        energy4 = torch.bmm(proj_query2, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
        # Cross semantic Guide Model
        batch_size, C_rgb, H, W = input_dsm.size()
        group_channels = C_rgb // self.num_groups
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        value = self.value_conv(fusion).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_dsm = torch.matmul(query_dsm, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm = self.softmax(attn_dsm)
        out_dsm = torch.matmul(attn_dsm, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)
        attn_rgb = torch.matmul(query_rgb, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb = self.softmax(attn_rgb)
        out_rgb = torch.matmul(attn_rgb, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        fusion = out_dsm * self.gamma_dsm + fusion + out_rgb * self.gamma_rgb
        # Cross semantic Guide Model
        return fusion


class MGFNet_Wu50(nn.Module):
    def __init__(self, bands1, bands2, num_classes=6, classification="Multi"):
        super(MGFNet_Wu50, self).__init__()
        resnet50 = ResNet50()
        self.dsm_conv0 = nn.Conv2d(bands2, 3, kernel_size=1)
        self.dsm_conv1 = copy.deepcopy(resnet50.conv1)
        self.dsm_bn1 = copy.deepcopy(resnet50.bn1)

        self.rgb_conv0 = nn.Conv2d(bands1, 3, kernel_size=1)
        self.rgb_conv1 = resnet50.conv1
        self.rgb_bn1 = resnet50.bn1

        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.rgb_layer1 = resnet50.layer1
        self.dsm_layer1 = copy.deepcopy(resnet50.layer1)

        self.rgb_layer2 = resnet50.layer2
        self.dsm_layer2 = copy.deepcopy(resnet50.layer2)

        self.rgb_layer3 = resnet50.layer3
        self.dsm_layer3 = copy.deepcopy(resnet50.layer3)

        
        self.csam_layer2 = CMI(in_dim=512)
        self.csam_layer3 = CMI(in_dim=1024)

        
        self.se_layer0 = MGF(64, 64)
        self.se_layer1 = MGF(256, 256)
        self.se_layer2 = MGF(512, 512)
        self.se_layer3 = MGF(1024, 1024)

        
        self.PAME = CMME(dim=1024, in_dim=1024)
     
        self.decoderCup = DecoderCup()
      
        self.segmentationHead = SegmentationHead(16, num_classes)
        
        self.classification = classification
        
    def forward(self, input_rgb, input_dsm):
        SE = True
        features = []
        input_rgb = self.rgb_conv0(input_rgb)
        input_dsm = self.dsm_conv0(input_dsm)  

        input_rgb0 = self.relu(self.rgb_bn1(self.rgb_conv1(input_rgb)))
        input_dsm0 = self.relu(self.dsm_bn1(self.dsm_conv1(input_dsm)))
        if SE:
            fusion0 = self.se_layer0(input_rgb0, input_dsm0)
        else:
            fusion0 = input_rgb0 + input_dsm0
        features.append(fusion0)
        input_rgb = self.maxpool(fusion0)
        input_dsm = self.maxpool(input_dsm0)

        input_rgb1 = self.rgb_layer1(input_rgb)
        input_dsm1 = self.dsm_layer1(input_dsm)
        if SE:
            fusion1 = self.se_layer1(input_rgb1, input_dsm1)
        else:
            fusion1 = input_rgb1 + input_dsm1
        features.append(fusion1)

        input_rgb2 = self.rgb_layer2(fusion1)
        input_dsm2 = self.dsm_layer2(input_dsm1)
        input_rgb2, input_dsm2 = self.csam_layer2(input_rgb2, input_dsm2)
        if SE:
            fusion2 = self.se_layer2(input_rgb2, input_dsm2)
        else:
            fusion2 = input_rgb2 + input_dsm2
        features.append(fusion2)

        # deep RGB
        input_rgb3 = self.rgb_layer3(fusion2)
        input_dsm3 = self.dsm_layer3(input_dsm2)
        input_rgb3, input_dsm3 = self.csam_layer3(input_rgb3, input_dsm3)
        if SE:
            fusion3 = self.se_layer3(input_rgb3, input_dsm3)
        else:
            fusion3 = input_rgb3 + input_dsm3
        fusion3 = self.PAME(fusion3, input_rgb3, input_dsm3)
        # print("fusion3:", fusion3.shape)
        x = self.decoderCup(fusion3, features[::-1])
        logits = self.segmentationHead(x)

        if self.classification == "Multi":
            return logits
        elif self.classification == "Binary":
            return F.sigmoid(logits)

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    bands1 = 64  # gaofen
    bands2 = 3  # lidar

    x = torch.randn(2, bands1, 512, 512, device=device)
    y = torch.randn(2, bands2, 512, 512, device=device)

    model = MGFNet_Wu50(bands1, bands2, num_classes=1, classification="Multi").to(device)
    print("output:", model(x, y).shape)