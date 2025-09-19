import torch
import torch.nn.functional as F


def tta_scale_inference(model, gaofen, lidar, scales=[0.9, 0.96, 1.0, 1.06, 1.1]):
    """
    Test-Time Augmentation (TTA) with multi-scale + horizontal flip
    Args:
        model: segmentation model
        gaofen: Tensor (B, C, H, W)
        lidar: Tensor (B, C, H, W)
        scales: list of scale ratios
    Returns:
        outputs: averaged prediction (B, C, H, W)
    """
    B, C, H, W = gaofen.shape
    preds_all = []

    for scale in scales:
        # 1. resize to target scale
        new_H, new_W = int(H * scale), int(W * scale)
        gaofen_resized = F.interpolate(gaofen, size=(new_H, new_W), mode='bilinear', align_corners=True)
        lidar_resized = F.interpolate(lidar, size=(new_H, new_W), mode='bilinear', align_corners=True)

        # 2. inference on original
        # print("gaofen_resized", gaofen_resized.shape, "lidar_resized", lidar_resized.shape)
        pred = model(gaofen_resized, lidar_resized)
        # resize back to original resolution
        pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
        preds_all.append(pred)

        # 3. inference on flipped
        gaofen_flip = torch.flip(gaofen_resized, dims=[3])
        lidar_flip = torch.flip(lidar_resized, dims=[3])
        pred_flip = model(gaofen_flip, lidar_flip)
        pred_flip = torch.flip(pred_flip, dims=[3])  # flip back
        pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=True)
        preds_all.append(pred_flip)

    # 4. average predictions
    outputs = torch.stack(preds_all, dim=0).mean(dim=0)  # (B, C, H, W)
    return outputs


def tta_rotate_inference(model, gaofen, lidar):
    # 第一组：原图 + 旋转90°
    gaofen_90 = torch.rot90(gaofen, k=1, dims=[2, 3])
    lidar_90 = torch.rot90(lidar, k=1, dims=[2, 3])
    # print("gaofen_90 shape: ", gaofen_90.shape) # B, C, H, W

    # 拼接 batch：原图 和 旋转图
    gaofen_batch = torch.cat([gaofen, gaofen_90], dim=0)
    lidar_batch = torch.cat([lidar, lidar_90], dim=0)
    # print("gaofen_batch shape: ", gaofen_batch.shape) # 2B, C, H, W

    # 第二组：左右翻转
    gaofen_flip = torch.flip(gaofen_batch, dims=[3])  # 翻转 W
    lidar_flip = torch.flip(lidar_batch, dims=[3])

    # 模型推理
    pred_a = model(gaofen_batch, lidar_batch)         # 原图 + 旋转图
    pred_b = model(gaofen_flip, lidar_flip)           # 翻转后预测
    pred_b = torch.flip(pred_b, dims=[3])             # 翻转回来

    # 融合两个方向的预测（上面只是把镜像图复原了，旋转图还没有复原）
    pred = (pred_a + pred_b) / 2                      # shape: (2B, C, H, W)

    # 拆分原图和旋转图的结果（复原旋转图）
    B = gaofen.shape[0]
    pred1 = pred[:B]                                  # 原图预测
    pred2 = torch.rot90(pred[B:], k=-1, dims=[2, 3])  # 旋转回原角度
    outputs = (pred1 + pred2) / 2                        # 最终融合 (B, C, H, W)

    return outputs