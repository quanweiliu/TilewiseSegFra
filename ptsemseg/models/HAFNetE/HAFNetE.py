import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.encoders.efficientnet import EfficientNetEncoder, _get_pretrained_settings

import time
import torch.nn.functional as F

# from torchsummaryX import summary
# import torchinfo
# from torchsummary import summary
# from ptsemseg.models.AFF_fusion.fusion import Muti_SP
# from fusion import Muti_SP
# from ptsemseg.models.CMF import CMF3
# from ptsemseg.models.SAGate import SAGate


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if in_channels < r:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels * r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels * r, in_channels, bias=False),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // r, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, rgb, dsm, cross_modal=None):
        # Single tensor for squeeze excitation ops
        if cross_modal is not None:
            x = torch.cat((rgb, dsm, cross_modal), dim=1)
        else:
            x = torch.cat((rgb, dsm), dim=1)
        batch, channels, _, _ = x.shape
        theta = self.squeeze(x).view(batch, channels)
        # print("theta", theta.shape)


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


class EfficientHAFNetCMEncoder(EfficientNetEncoder, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        super(EfficientHAFNetCMEncoder, self).__init__(stage_idxs=stage_idxs,
                                                       out_channels=out_channels,
                                                       model_name=model_name,
                                                       depth=5)
        self.se_blocks = self.get_se_blocks_list(out_channels)

    def forward(self, rgb_features, dsm_features):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate
        cross_modal_features = []
        # Dummy feature
        cross_modal_features.append(torch.ones(1, 1, 1, 1))   # test feature
        # First fusion has only rgb and dsm features
        x = self.se_blocks[0](rgb_features[1], dsm_features[1])
        cross_modal_features.append(x)

        for i in range(2, self._depth + 1):
            # i + 1 : Identity stage in rgb and dsm features is skipped
            for module in stages[i]:
                drop_connect = drop_connect_rate * block_number / len(self._blocks)
                block_number += 1.
                x = module(x, drop_connect)
            cross_modal_features.append(x)
            x = self.se_blocks[i - 1](rgb_features[i], dsm_features[i], cross_modal_features[-1])

        return cross_modal_features

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, strict=False, **kwargs)

    @staticmethod  # 声明为静态方法，可以使用类直接及进行调用，不需要self参数，其使用方法和直接调用函数一样(一般来说，调用某个类的方法，需要先生成一个实例)
    def get_se_blocks_list(out_channels):
        # stage_img_size=[(512,512),(256,256),(128,128),(64,64),(32,32),(16,16)]
        se_blocks = []
        # Remove Identity stage channel,
        # (not used to choose the number of input channels in SE fusion block)
        out_channels = list(out_channels)
        out_channels.pop(0)    # 移除列表中第一个元素，并且返回该元素的值
        out_channels = tuple(out_channels)
        for idx, c in enumerate(out_channels):
            if idx == 0:
                se_blocks.append(SEBlock(in_channels = c*2))

            else:
                se_blocks.append(SEBlock(in_channels = c*3))

        return nn.ModuleList(se_blocks)


class EfficientHAFNet(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0', encoder_weights='imagenet', n_classes=2):
        # encoder_name='efficientnet-b0', encoder_weights='imagenet'
        super(EfficientHAFNet, self).__init__()

        self.rgb_stream = smp.Unet(    # use encoder of efficientnet-b0
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = 193,
            classes = n_classes,
        )

        self.dsm_stream = smp.Unet(      # use encoder of efficientnet-b0
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
        )

        self.cross_modal_stream = smp.Unet(   #
            encoder_name='efficienthafnet-cm-b0'   # use encoder of efficienthafnet-cm-b0  下面刚定义好的encoder
        )
        # self.decision_level_fusion_block = MCE_MSP(in_channels=3,c=3,input_size=(512,512))
        self.decision_level_fusion_block = SEBlock(3)

    def forward(self, rgb, dsm):
        # print("rgb", rgb.shape)
        # print("dsm", dsm.shape)
        rgb_features = self.rgb_stream.encoder(rgb)
        rgb_decoder_out = self.rgb_stream.decoder(*rgb_features)
        rgb_pred = self.rgb_stream.segmentation_head(rgb_decoder_out)
        # for f in rgb_features:
        #     print("f.shape", f.shape)
        # print("rgb_pred", rgb_pred.shape)
        # print("rgb_decoder_out", rgb_decoder_out.shape)
        # print("rgb_pred", rgb_pred.shape)

        dsm_features = self.dsm_stream.encoder(dsm)
        dsm_decoder_out = self.dsm_stream.decoder(*dsm_features)
        dsm_pred = self.dsm_stream.segmentation_head(dsm_decoder_out)
        # print("dsm_features", rgb_pred.shape)
        # print("dsm_decoder_out", dsm_decoder_out.shape)
        # print("dsm_pred", dsm_pred.shape)

        cross_modal_features = self.cross_modal_stream.encoder(rgb_features, dsm_features)
        cross_modal_decoder_out = self.cross_modal_stream.decoder(*cross_modal_features)
        cross_modal_pred = self.cross_modal_stream.segmentation_head(cross_modal_decoder_out)

        x = self.decision_level_fusion_block(rgb_pred, dsm_pred, cross_modal_pred)

        # feats = {
        #     "rgb_features": rgb_features,
        #     "dsm_features": dsm_features,
        #     "cross_modal_features": cross_modal_features
        # }
        # preds = {
        #     "rgb_pred": rgb_pred,
        #     "dsm_pred": dsm_pred,
        #     "cross_modal_pred": cross_modal_pred
        # }

        # # return x, feats, preds

        return F.sigmoid(x)


smp.encoders.encoders["efficienthafnet-cm-b0"] = {
    "encoder": EfficientHAFNetCMEncoder,                                  # encoder(class)
    "pretrained_settings": _get_pretrained_settings("efficientnet-b0"),   # weight
    "params": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (3, 5, 9, 16),
        "model_name": "efficientnet-b0",
    },
}
smp.encoders.encoders["efficienthafnet-cm-b2"] = {
    "encoder": EfficientHAFNetCMEncoder,
    "pretrained_settings": _get_pretrained_settings("efficientnet-b2"),
    "params": {
        "out_channels": (3, 32, 24, 48, 120, 352),
        "stage_idxs": (5, 8, 16, 23),
        "model_name": "efficientnet-b2",
    },
}
smp.encoders.encoders["efficienthafnet-cm-b4"] = {
    "encoder": EfficientHAFNetCMEncoder,
    "pretrained_settings": _get_pretrained_settings("efficientnet-b4"),
    "params": {
        "out_channels": (3, 48, 32, 56, 160, 448),
        "stage_idxs": (6, 10, 22, 32),
        "model_name": "efficientnet-b4",
    },
}


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # rgb = torch.randn(1, 4, 512, 512, device = device)
    # dsm = torch.randn(1, 2, 512, 512, device = device)
    # rgb = torch.randn(1, 193, 128, 128, device = device)
    # dsm = torch.randn(1, 3, 128, 128, device = device)
    rgb = torch.randn(1, 193, 480, 640, device = device)
    dsm = torch.randn(1, 3, 480, 640, device = device)
    efficient_hafnet = EfficientHAFNet(encoder_name = 'efficientnet-b0', \
                                       encoder_weights = 'imagenet', \
                                       n_classes = 1)

    model = efficient_hafnet.to(device)
    output = model(rgb, dsm)

    print("output", output.shape)
    # rgb_parameters = [param for name, param in efficient_hafnet.named_parameters()]
    # summary(model, rgb, dsm)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # rgb = torch.randn(1, 64, 256, 256, device=device)
    # lidar = torch.randn(1, 64, 256, 256, device=device)
    # model=MCE_MSP(in_channels=64*2,c=2,input_size=(256,256))
    # summary(model.to(device),rgb,lidar)


