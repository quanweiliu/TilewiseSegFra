import os
import torch
from torch import nn
from torch.nn import functional as F
from thop import profile, clever_format

from .mix_transformer import OverlapPatchEmbed, mit_b0
from .convnext import convnext_tiny, LayerNorm
from .MLPDecoder import DecoderHead
# from mix_transformer import OverlapPatchEmbed, mit_b0
# from convnext import convnext_tiny, LayerNorm
# from MLPDecoder import DecoderHead


def load_pretrain2(net, pretrain_name):
    dir_path = os.getcwd()
    pretrain_path = os.path.join(dir_path, '/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains', pretrain_name)
    # print("Pretrain_path:", pretrain_path)
    net_dict = net.state_dict()

    pretrain_dict = torch.load(pretrain_path, weights_only=True)

    dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
    net_dict.update(dict)
    net.load_state_dict(net_dict)
    return net


model1 = convnext_tiny(pretrained=True, drop_path_rate=0.3)
ft1 = model1.stages
stem = model1.downsample_layers
stem1 = [stem[0], stem[1], stem[2], stem[3]]
layers1 = [
    ft1[0],
    ft1[1],
    ft1[2],
    ft1[3]]

model2 = mit_b0()
model2 = load_pretrain2(model2, pretrain_name='mit_b0.pth')
stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]


def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, N, groups, channels_per_group)

    # Transpose operation is not valid for 5D tensor, so we need to use permute
    x = x.permute(0, 1, 3, 2).contiguous()

    # flatten
    x = x.view(batchsize, N, -1)

    return x


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2):
        super(Cross_Atten_Lite_split, self).__init__()
        self.midc1 = torch.tensor(inc1 // 4)
        self.midc2 = torch.tensor(inc2 // 4)

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout = nn.Dropout(0.2)
        self._init_weight()

    def forward(self, x, x1, x2):
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)

        x1 = self.bn_x1(x1)
        x2 = self.bn_x2(x2)

        kq1 = self.kq1(x1.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq2 = self.kq2(x2.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = torch.split(kq, self.midc2, dim=2)

        v = self.v_conv(x.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        v1, v2 = torch.split(v, self.midc1, dim=2)

        mat = torch.matmul(q1, k1.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v1 = torch.matmul(mat, v1)

        mat = torch.matmul(q2, k2.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v2 = torch.matmul(mat, v2)

        v = torch.cat([v1, v2], dim=2).view(batch_size, h, w, -1)
        v = self.out_conv(v)
        v = v.permute(0, 3, 1, 2)
        v = self.bn_last(v)
        v = v + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.inc = torch.tensor(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self._init_weight()

    def forward(self, x):

        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)

        y_spatial = self.fc_spatial(y_avg).view(b, c, 1, 1)
        y_channel = self.fc_channel(y_avg).view(b, c, 1, 1)
        y_channel = y_channel.sigmoid()

        map = (x * (y_spatial)).sum(dim=1) / self.inc
        map = (map / self.inc).sigmoid().unsqueeze(dim=1)
        return map * x * y_channel

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        channel = inc_rgb + inc_depth2

        self.fus_atten = SpatialAttention_max(in_channels=channel)
        self.conv1 = nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inc_depth2)

        self.cross_atten = Cross_Atten_Lite_split(inc_depth2, inc_rgb)

    def forward(self, depth_out, rgb_out):
        fus_s = torch.cat([depth_out, rgb_out], dim=1)
        fus_s = self.fus_atten(fus_s)
        fus_s = self.conv1(fus_s)
        fus_s = self.bn(fus_s)

        fus_s = self.cross_atten(fus_s, depth_out, rgb_out)

        return fus_s

# stem = nn.Sequential(
#     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
# )

class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num, bands1=3, bands2=1):
        super(down_sample_block, self).__init__()
        self.block_num = block_num

        if block_num != 0:
            self.depth_stem = stem2[block_num]
            self.rgb_stem = stem1[block_num]
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = stem1[0]
            # self.rgb_stem = nn.Sequential(
            #                 nn.Conv2d(193, 96, kernel_size=4, stride=4),
            #                 LayerNorm(96, eps=1e-6, data_format="channels_first")
            #             )
        self.rgb_layer = layers1[block_num]            # conv
        self.depth_layer = layers2[block_num]          # transformer

        self.depth_norm = norm2[block_num]

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb)

    def forward(self, image, depth):
        B = image.shape[0]
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        depth_out, H, W = self.depth_stem(depth)

        for i, blk in enumerate(self.depth_layer):
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # SCC_Ablation
        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, bands1, bands2, n_classes, classification="Multi", **param_dict):
        super(B0_T, self).__init__()

        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], 
                                               inc_rgb=channel_list2[0], 
                                               block_num=0, 
                                               bands1=bands1, 
                                               bands2=bands2,)

        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1],
                                               inc_rgb=channel_list2[1], 
                                               block_num=1)

        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2],
                                               inc_rgb=channel_list2[2], 
                                               block_num=2)

        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3],
                                               inc_rgb=channel_list2[3], 
                                               block_num=3)

        self.Decoder = DecoderHead(in_channels=self.channel, 
                                   num_classes=n_classes, 
                                   dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)
        
        self.classification = classification

    def forward(self, image, depth):
        # print("image", image.shape, "depth", depth.shape)
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        # print("rgb_out", rgb_out.shape, "depth_out1", depth_out1.shape)

        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)
        # print("rgb_out", rgb_out.shape, "depth_out2", depth_out2.shape)


        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        # print("rgb_out", rgb_out.shape, "depth_out3", depth_out3.shape)

        _, depth_out = self.down_sample_4(rgb_out, depth_out3)
        # print("rgb_out", rgb_out.shape, "depth_out", depth_out.shape)


        rgb_out = self.Decoder([
            depth_out1,
            depth_out2,
            depth_out3,
            depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        
        if self.classification == "Multi":
            return rgb_out
        elif self.classification == "Binary":
            return torch.sigmoid(rgb_out)


if __name__ == '__main__':

    # 只修改了两处，第一处是 import LayerNorm 层，第二处是将 rgb_stem 替换为     
    #  self.rgb_stem = nn.Sequential(
                            # nn.Conv2d(193, 96, kernel_size=4, stride=4),
                            # LayerNorm(96, eps=1e-6, data_format="channels_first")

    bands1 = 3
    bands2 = 1
    image = torch.randn(4, bands1, 512, 512)
    depth = torch.randn(4, bands2, 512, 512)
    # image = torch.randn(8, 193, 128, 128)
    # depth = torch.randn(8, 3, 128, 128)
    model = B0_T(bands1, bands2, n_classes=20, classification="Multi")
    out = model(image, depth)
    print(out.shape)
    # print("Done")


    ####################### FLOPs and Params #########################################
    # input=(torch.randn(2, 193, 128, 128), torch.randn(2, 3, 128, 128))
    # flops, params = profile(model, inputs=input)
    # flops, params = clever_format([flops, params])
    # print('# Model FLOPs: {}'.format(flops))
    # print('# Model Params: {}'.format(params))
################################# train ###################################################