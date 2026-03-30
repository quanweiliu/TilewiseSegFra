# @copyright ziqi-jin

import torch.nn as nn
import torch
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
from .segment_anything_ori.modeling.mask_decoder import MaskDecoder
from typing import List, Tuple
from torch.nn import functional as F
from .mask_decoder_heads import SemSegHead
from .mask_decoder_neck import MaskDecoderNeck


class BaseMaskDecoderAdapter(MaskDecoder):
    '''
      multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    '''

    # is fix and load params
    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseMaskDecoderAdapter, self).__init__(transformer_dim=ori_sam.mask_decoder.transformer_dim,
                                                     transformer=ori_sam.mask_decoder.transformer)
        self.sam_mask_decoder = ori_sam.mask_decoder
        if fix:
            fix_params(self.sam_mask_decoder)  # move to runner to implement

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(image_embeddings=image_embeddings,
                                                                   image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                   sparse_prompt_embeddings=sparse_embeddings,
                                                                   dense_prompt_embeddings=dense_embeddings,
                                                                   multimask_output=multimask_output, )
        return low_res_masks, iou_predictions


class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False, class_num=20):
        super(SemMaskDecoderAdapter, self).__init__(ori_sam, fix)
        self.decoder_neck = MaskDecoderNeck(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                            transformer=self.sam_mask_decoder.transformer,
                                            num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs)
        self.decoder_head = SemSegHead(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                       num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs,
                                       iou_head_depth=self.sam_mask_decoder.iou_head_depth,
                                       iou_head_hidden_dim=self.sam_mask_decoder.iou_head_hidden_dim,
                                       class_num=class_num)
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck)
        self.pair_params(self.decoder_head)

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True,
                scale=1):
        # 获取原始的 image_pe
        image_pe = prompt_adapter.sam_prompt_encoder.get_dense_pe()
        pe_encoder = prompt_adapter.sam_prompt_encoder
        
        # 如果输入大小不是 1024，需要调整 image_pe 和 dense_embeddings
        # image_embedding_size 在初始化时被设置为固定值（64 for 1024 input）
        current_h, current_w = image_embeddings.shape[2], image_embeddings.shape[3]
        expected_h, expected_w = pe_encoder.image_embedding_size
        
        if current_h != expected_h or current_w != expected_w:
            image_pe = self._resize_image_pe(image_pe, current_h, current_w)
            dense_embeddings = self._resize_dense_embeddings(dense_embeddings, current_h, current_w)
        
        src, iou_token_out, mask_tokens_out, src_shape = self.decoder_neck(image_embeddings=image_embeddings,
                                                                           image_pe=image_pe,
                                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                                           dense_prompt_embeddings=dense_embeddings,
                                                                           multimask_output=multimask_output, )
        masks, iou_pred = self.decoder_head(src, iou_token_out, mask_tokens_out, src_shape, mask_scale=scale)
        return masks, iou_pred
    
    def _resize_image_pe(self, image_pe, tgt_h, tgt_w):
        """
        调整 image_pe（位置编码）的大小
        
        Args:
            image_pe: 形状 (1, C, H_src, W_src)，其中 C=embed_dim
            tgt_h: 目标高度
            tgt_w: 目标宽度
        
        Returns:
            调整后的 image_pe，形状 (1, C, tgt_h, tgt_w)
        """
        # image_pe 已经是 (B, C, H, W) 格式，直接插值
        image_pe = F.interpolate(
            image_pe,
            size=(tgt_h, tgt_w),
            mode='bilinear',
            align_corners=False
        )
        
        return image_pe
    
    def _resize_dense_embeddings(self, dense_embeddings, tgt_h, tgt_w):
        """
        调整 dense_embeddings 的大小以匹配目标高度和宽度
        
        Args:
            dense_embeddings: 形状 (B, C, H_src, W_src)
            tgt_h: 目标高度
            tgt_w: 目标宽度
        
        Returns:
            调整后的 dense_embeddings，形状 (B, C, tgt_h, tgt_w)
        """
        B, C, H_src, W_src = dense_embeddings.shape
        
        # 如果大小已经匹配，直接返回
        if H_src == tgt_h and W_src == tgt_w:
            return dense_embeddings
        
        # dense_embeddings 已经是 (B, C, H, W) 格式，直接插值
        dense_embeddings = F.interpolate(
            dense_embeddings,
            size=(tgt_h, tgt_w),
            mode='bilinear',
            align_corners=False
        )
        
        return dense_embeddings

    def pair_params(self, target_model: nn.Module):
        src_dict = self.sam_mask_decoder.state_dict()
        for name, value in target_model.named_parameters():
            if name in src_dict.keys():
                value.data.copy_(src_dict[name].data)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
