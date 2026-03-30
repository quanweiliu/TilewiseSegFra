# copyright ziqi-jin
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter


def interpolate_pos_embed(pos_embed, src_size, tgt_size):
    """
    插值调整位置嵌入的大小
    
    Args:
        pos_embed (torch.Tensor): 原始位置嵌入，形状 (1, H, W, C) 或 (1, L, C)
        src_size (int): 原始图像大小（如 1024）
        tgt_size (int): 目标图像大小（如 256 或 512）
    
    Returns:
        torch.Tensor: 调整后的位置嵌入
    """
    if pos_embed.ndim == 4:  # (1, H, W, C) 格式
        src_h = src_w = src_size // 16  # 默认 patch_size=16
        tgt_h = tgt_w = tgt_size // 16
        
        # 从 (1, H, W, C) 转换为 (1, C, H, W) 便于插值
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        
        # 使用双线性插值调整大小
        pos_embed = F.interpolate(
            pos_embed,
            size=(tgt_h, tgt_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 转换回 (1, H, W, C)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
    
    return pos_embed


def interpolate_rel_pos(rel_pos, src_size, tgt_size):
    """
    插值调整相对位置嵌入的大小（用于 Transformer 中的相对位置）
    
    Args:
        rel_pos (torch.Tensor): 相对位置嵌入，形状 (2*src_h-1, C) 例如 (127, 64)
        src_size (int): 原始大小（如 1024）
        tgt_size (int): 目标大小（如 256 或 512）
    
    Returns:
        torch.Tensor: 调整后的相对位置嵌入，形状 (2*tgt_h-1, C)
    """
    src_h = src_size // 16
    tgt_h = tgt_size // 16
    
    # 相对位置的标准大小是 2*h-1
    src_len = 2 * src_h - 1
    tgt_len = 2 * tgt_h - 1
    
    if rel_pos.shape[0] != tgt_len:
        # rel_pos 形状: (src_len, C) 例如 (127, 64)
        # 转换为 (1, C, src_len) 以便在位置维度进行插值
        rel_pos = rel_pos.permute(1, 0).unsqueeze(0)  # (1, C, src_len)
        
        # 使用线性插值调整位置维度
        rel_pos = F.interpolate(
            rel_pos,
            size=tgt_len,
            mode='linear',
            align_corners=False
        )  # (1, C, tgt_len)
        
        # 转换回 (tgt_len, C)
        rel_pos = rel_pos.squeeze(0).permute(1, 0)  # (tgt_len, C)
    
    return rel_pos


class BaseExtendSam(nn.Module):

    '''
    基本只是包了一层，外加冻结控制。对SAM没有任何改动，输入输出也完全一样。
    主要是为了后续的语义分割 SAM 做准备，方便在不改动原版 SAM 的基础上进行修改。
     - 语义分割 SAM 的改动主要集中在 mask decoder 上，其他部分基本保持不变。
     - 通过调整位置嵌入来适应不同输入大小的图像编码器也是在这个基础上实现的，保持了原版 SAM 的灵活性和兼容性。
     - 这样设计的好处是可以最大程度地复用原版 SAM 的权重和结构，同时又能针对语义分割的需求进行定制化修改，而不会影响原版 SAM 的其他功能。
    '''

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)

    def forward(self, img):
        x = self.img_adapter(img)
        points = None
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


class SemanticSam(BaseExtendSam):
    """
    语义分割 SAM
    支持任意输入大小，通过调整位置嵌入（pos_embed）来适应

    相比原版 SAM 
    - 把原来的 BaseMaskDecoderAdapter，中的 mask decoder 换成了 SemMaskDecoderAdapter。（neck + head）
    - 把原来的 mask 输出头改成了面向语义分割的 head，输出类别数由 class_num 参数控制。
     - 其他部分（图像编码器、提示编码器）基本保持不变，输入输出接口也保持不变，方便后续的集成和使用。
     - 通过调整位置嵌入来支持不同输入大小的图像编码器，使得模型更加灵活，可以适应不同分辨率的输入图像，
     - 而不需要修改原版 SAM 的结构或权重。这种设计使得语义分割 SAM 能够在保持原版 SAM 的优势和性能的同时，针对语义分割任务进行定制化修改。
    """
    
    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, 
                 class_num=20, model_type='vit_b', input_size=1024):
        """
        Args:
            ckpt_path: SAM 模型权重文件路径
            fix_img_en: 是否冻结 image encoder
            fix_prompt_en: 是否冻结 prompt encoder
            fix_mask_de: 是否冻结 mask decoder
            class_num: 语义分割的类别数
            model_type: SAM 模型大小 ('vit_b', 'vit_l', 'vit_h')
            input_size: 输入图像大小（默认 1024），支持 256, 512, 768, 1024 等
        """
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)
        
        # 调整图像编码器以支持不同的输入大小
        self.input_size = input_size
        self.default_input_size = 1024
        
        if input_size != self.default_input_size:
            self._adjust_pos_embed(input_size)
    
    def forward(self, img):
        """
        前向传播，处理不同输入大小
        """
        # print(f"输入图像形状: {img.shape}")  # 调试信息
        
        x = self.img_adapter(img)
        # print(f"图像编码器输出形状: {x.shape}")  # 调试信息
        points = None
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        # print(f"稀疏嵌入形状: {sparse_embeddings.shape}")  # 调试信息
        # print(f"密集嵌入形状: {dense_embeddings.shape}")  # 调试信息

        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # print(f"低分辨率掩码形状: {low_res_masks.shape}")  # 调试信息
        # print(f"IOU预测形状: {iou_predictions.shape}")  # 调试信息

        upsampled_logits = F.interpolate(
            low_res_masks, 
            size=(256, 256),  # 目标输出大小
            mode='bilinear', 
            align_corners=False
        )

        return upsampled_logits
        # return upsampled_logits, iou_predictions
    
    def _adjust_pos_embed(self, input_size):
        """
        调整 image_encoder 中的位置嵌入和相对位置嵌入，支持不同的输入大小
        """
        encoder = self.img_adapter.sam_img_encoder  # ✅ 正确的属性名
        
        print(f"调整位置嵌入: {self.default_input_size}×{self.default_input_size} → {input_size}×{input_size}")
        
        # 调整绝对位置嵌入 (pos_embed)
        if encoder.pos_embed is not None:
            encoder.pos_embed.data = interpolate_pos_embed(
                encoder.pos_embed.data,
                self.default_input_size,
                input_size
            ).to(encoder.pos_embed.device)
        
        # 调整相对位置嵌入 (rel_pos_h, rel_pos_w)
        # 遍历所有 Transformer 块，调整其相对位置嵌入
        for blk in encoder.blocks:
            if hasattr(blk.attn, 'rel_pos_h') and blk.attn.rel_pos_h is not None:
                blk.attn.rel_pos_h.data = interpolate_rel_pos(
                    blk.attn.rel_pos_h.data,
                    self.default_input_size,
                    input_size
                ).to(blk.attn.rel_pos_h.device)
            
            if hasattr(blk.attn, 'rel_pos_w') and blk.attn.rel_pos_w is not None:
                blk.attn.rel_pos_w.data = interpolate_rel_pos(
                    blk.attn.rel_pos_w.data,
                    self.default_input_size,
                    input_size
                ).to(blk.attn.rel_pos_w.device)
        
        print(f"✅ 位置嵌入调整完成！")
