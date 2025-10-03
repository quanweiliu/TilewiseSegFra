import copy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch import Tensor, einsum


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def cross_entropy2d(input, target, ignore_index=-1):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        # target = target.unsequeeze(1)
        # target = F.upsample(target, size=(h, w), mode="nearest")
        # target = target.sequeeze(1)
        input = F.interpolate(input, size=(ht, wt), mode="biliner", align_corners=True)
    elif h < ht and w < wt:  # upsample images
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    loss = F.cross_entropy(input, target, ignore_index=ignore_index)
    return loss, loss, loss

class dice_bce_loss1(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss1, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weight = (torch.from_numpy(np.array(weight))).type(torch.Tensor)
        # weight = weight.to(device)
        # self.weight = weight
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):

        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

class my_dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(my_dice_bce_loss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weight = (torch.from_numpy(np.array(weight))).type(torch.Tensor)
        # weight = weight.to(device)
        # self.weight = weight
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, out,refine_out, y_true):

        a = self.bce_loss(out, y_true)
        b = self.soft_dice_loss(y_true, refine_out)
        return 4*a + b

def MSE(input, target, size_average=True):
    target.float()
    input= input.squeeze(1)
    return F.mse_loss(input, target, size_average=True)

# def dice_loss(y_pred, y_true):
#     smooth = 1.0
#
#     y_pred = y_pred.view(-1)
#     # print(y_pred.type())
#     y_true = y_true.view(-1)
#     # print(y_true.type())
#     y_true = y_true.float()
#     # print(y_true.type())
#
#     i = torch.sum(y_true)
#     j = torch.sum(y_pred)
#     intersection = torch.sum(y_true * y_pred)
#     score = (2. * intersection + smooth) / (i + j + smooth)
#     soft_dice_coeff = score.mean()
#
#     soft_dice_loss = 1 - soft_dice_coeff
#
#     return soft_dice_loss

def dice_loss(input, target):
    # input = F.sigmoid(input)

    n, c, h, w = input.size()
    # nt,ht, wt = target.size()
    nt,nc, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)

    smooth = 0.00001
    # smooth=1.0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # W_eight= (torch.from_numpy(np.array(W_eight))).type(torch.Tensor)
    # W_eight = W_eight.to(device)
    # print(input, target)
    y_pred = input.view(-1)
    #print(y_pred.type())
    y_true = target.view(-1).float()
    #print(y_true.type())
    # print(y_pred)
    # print(y_true)

    # print("y_true", y_true.min(), "y_true", y_true.max())  # 查看 `y_true` 的数值范围
    i = torch.sum(y_true)
    j = torch.sum(y_pred)
    intersection = torch.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (i + j + smooth)
    soft_dice_coeff = score.mean()

    soft_dice_loss = 1 - soft_dice_coeff

    #a = bce_loss(input = y_pred, target = y_true, weight = W_eight, size_average =S_ize_average)
    #b = soft_dice_loss

    return soft_dice_loss

def bce_loss(input, target):

    # input = F.sigmoid(input)
    _, _, h, w = input.size()
    # _, ht, wt = target.size()
    _, _, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)
        target = target.long()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = target.contiguous().view(-1, 1)
    input = input.contiguous().view(-1,1)
    target = target.float()
    # loss = F.binary_cross_entropy_with_logits(input, target, class_weight, size_average)
    loss = F.binary_cross_entropy(input, target)
    return loss


def bce_loss2(input, target, device_num, weight=None):

    # input = F.sigmoid(input)
    _, _, h, w = input.size()
    # _, ht, wt = target.size()
    _, _, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)
        target = target.long()

    device = torch.device("cuda:"+str(device_num) if torch.cuda.is_available() else "cpu")
    weight = (torch.from_numpy(np.array(weight))).type(torch.Tensor)
    weight = weight.to(device)
    weight = weight.view(-1, 1)

    target = target.contiguous().view(-1, 1)
    input = input.contiguous().view(-1,1)
    class_weight = torch.gather(weight, 0, target.long())
    # target = target.float()
    # loss = F.binary_cross_entropy_with_logits(input, target, class_weight, size_average)
    loss = F.binary_cross_entropy(input, target, class_weight)
    return loss

def focal_loss(input, target, device, gamma=2, alpha=None, size_average=True):
    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")

    input = input.contiguous().view(-1, 1)
    target = target.contiguous().view(-1, 1)

    pro = torch.cat((1-input, input), 1)
    #pro_0 = torch.cat((input, 1-input), 1)

    #target_float = target.float()
    #select_1 = (pro.gather(1,target))*(target_float) + 1e-9
    #select_0 = (pro.gather(1,1-target))*(target_float) + 1e-9

    select_init = torch.FloatTensor(len(pro), 2).zero_().to(device)
    # select_init = torch.FloatTensor(len(pro), 2).zero_()
    select = select_init.scatter(1, target.long(), 1.)

    if alpha is not None:
        weight = torch.tensor([[alpha], [1.0-alpha]])
        if weight.type() != input.data.type():
            weight = weight.type_as(input.data)
    weight = weight.to(device)
    # weight = weight

    weight = weight.view(-1,1)
    class_weight = torch.gather(weight, 0, target.long())


    pro_data = (pro*select).sum(1).view(-1, 1)
    pro_data = torch.clamp(pro_data, 1e-7,1-1e-7)
    batchloss = -class_weight*((1-pro_data)**gamma)*pro_data.log()

    # if alpha is not None:
    #     alpha = torch.tensor(alpha)
    #     if alpha.type() != input.data.type():
    #         alpha = alpha.type_as(input.data)
    # alpha = alpha.to(device)

    # pos_part = -1*(1-input)**gamma*(select_1.data.log())
    # p_sum = pos_part.sum()
    # neg_part = -1*input**gamma*(select_0.data.log())
    # n_sum = neg_part.sum()
    #
    # loss = alpha*pos_part + (1-alpha)*neg_part
    # p1_sum = (alpha*pos_part).sum()
    # n1_sum = ((1-alpha)*neg_part).sum()

    if size_average == True:
        loss = batchloss.mean()
    else:
        loss = batchloss

    return loss


def focal_dice_loss(input, target, device, gamma = 2, alpha = 0.25, size_average=True):
    f_loss = focal_loss(input, target,device, gamma, alpha, size_average)
    d_loss = dice_loss(input, target)
    loss = f_loss + d_loss
    return loss


def focal_bce_loss(input, target, device, gamma = 2, alpha = 0.25, size_average=True):
    f_loss = focal_loss(input, target, device, gamma, alpha, size_average)
    b_loss = bce_loss(input, target, size_average=size_average)
    loss = f_loss + b_loss
    return loss, b_loss, f_loss


def multiclass_dice_loss(input, target, classes=None, epsilon=1e-6):
    """
    input: [N, C, H, W] - softmax probabilities
    target: [N, H, W] - integer labels from 0 to C-1
    """

    if classes is None:
        classes = input.shape[1]

    input_soft = F.softmax(input, dim=1)  # Convert logits to probabilities
    target_one_hot = F.one_hot(target, num_classes=classes).permute(0, 3, 1, 2).float()  # [N, C, H, W]

    dims = (0, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    union = torch.sum(input_soft + target_one_hot, dims)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss = 1. - dice.mean()

    return dice_loss


def multiclass_ce_dice_loss(input, target, a, b, classes):
    """
    input: [N, C, H, W] - raw logits
    target: [N, H, W]   - integer labels
    """
    ce_loss = F.cross_entropy(input, target)  # CE Loss
    dice = multiclass_dice_loss(input, target, classes)
    loss = a * ce_loss + b * dice
    return loss, ce_loss, dice


def dice_bce_loss_re(input, target, a, b, size_average=True):
    bceloss = bce_loss(input, target)
    diceloss = dice_loss(input, target)
    # alpha = 0.3
    # loss = (1-alpha)*bceloss + alpha*diceloss
    loss = a * bceloss + b * diceloss
    return loss, bceloss, diceloss


def dice_bce_loss_re2(input, target, device, weight, a, b, size_average=True):
    bceloss = bce_loss2(input, target, device_num=device, weight=weight, size_average=size_average)
    diceloss = dice_loss(input, target)
    # alpha = 0.3
    # loss = (1-alpha)*bceloss + alpha*diceloss
    loss = a * bceloss + b * diceloss
    return loss, bceloss, diceloss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def dice_bce_loss_re3(input, target, a, b):
    bceloss = nn.BCEWithLogitsLoss()(input,target)
    diceloss = DiceLoss()(input, target)
    loss = a*bceloss + b*diceloss
    return loss


def new_dice_bce_loss(input1, input2, input3, target, size_average=True):
    bceloss1 = bce_loss(input1, target, size_average = size_average)
    bceloss2= bce_loss(input2, target, size_average = size_average)
    bceloss3= bce_loss(input3, target, size_average = size_average)
    diceloss1 = dice_loss(input1, target)
    diceloss2 = dice_loss(input2, target)
    diceloss3 = dice_loss(input3, target)
    a=0.3
    bceloss = a*bceloss1 + a*bceloss2+(1-a*2)*bceloss3
    diceloss=a*diceloss1+a*diceloss2+(1-a*2)*diceloss3
    b=0.75
    loss=b*bceloss+(1-b)*diceloss

    return loss, bceloss, diceloss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=[0.93, 1.07]):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

##  待温度的 KD_loss
def KD_loss(p, q, Temp=3):# p是老师的预测(经过softmax)，q是学生的预测
    pt = F.softmax(p / Temp, dim=1)
    ps = F.log_softmax(q / Temp, dim=1)
    return nn.KLDivLoss(reduction='mean')(ps, pt) * (Temp ** 2)


def KL_loss(inputs, target, Temp=3):   # inputs 是学生，target是老师
    KL_loss=[]
    for input in inputs:
        loss=KD_loss(target,input,Temp)
        KL_loss.append(loss)
    return sum(KL_loss)/len(KL_loss)


def Label_loss(inputs, target, a, b):
    l_loss = []
    for l_pred in inputs:
        loss = dice_bce_loss_re3(l_pred, target, a, b)
        l_loss.append(loss)
    return sum(l_loss) / len(l_loss)

# def multi_loss(input1, input2, input3, target, device, size_average=True):
#     loss1=dice_loss(input1, target)
#     loss2=bce_loss(input2, target, size_average = size_average)
#     loss3=bce_loss(input3, target, size_average = size_average)
#     loss = (loss1 + loss2 + loss3) / 3
#     return loss, loss1, loss2

def multiclass_multi_loss(input, target, a, b, classes, aux_weight=1):
    out_feats = input[0]                     # [B, C, H, W]
    aux_feats = input[1:]                    # list of [B, C, h', w']
    aux_loss = []

    _, _, h, w = out_feats.shape

    for aux in aux_feats:
        # Upsample aux prediction to main output size
        aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
        ce = F.cross_entropy(aux, target)
        dice = multiclass_dice_loss(aux, target, classes)
        aux_loss.append(a * ce + b * dice)

    # Main loss
    loss1 = a * F.cross_entropy(out_feats, target) + \
            b * multiclass_dice_loss(out_feats, target, classes)
    loss2 = sum(aux_loss)
    loss = loss1 + aux_weight * loss2

    return loss, loss1, loss2


def multi_loss(input, target, a, b, aux_weight=1):
    out_feats= input[0]
    aux_feats, aux_loss = input[1:], []
    _, _, h, w = target.size()

    for aux in aux_feats:
        # b, _, h, w = aux.size()
        aux_target = F.interpolate(aux.float().unsqueeze(1), size=(1, h, w)).squeeze(1)
        # print(aux_target.shape, target.shape)

        bceloss = bce_loss(aux_target, target)
        diceloss = dice_loss(aux_target, target)
        aux_loss.append(a*bceloss + b*diceloss)
        # aux_loss.append(bceloss)
    
    loss1 = a*bce_loss(out_feats, target) + b*dice_loss(out_feats, target)
    loss2 = sum(aux_loss)
    loss = loss1 + aux_weight * loss2
    return loss, loss1, loss2

def multi_loss2(input, target, device, a, b, size_average=True, alpha=0.5):
    target=target.squeeze(1)
    input0 = input[0]
    input1 = input[1]
    input2 = input[2]

    # 计算 loss from labels
    la_loss = Label_loss(input1, class2one_hot(target,2).float(), a, b)

    # 计算 loss from distllation
    d_teacher = input2[-1]
    d_student = input2[0 : -1]
    kd_loss = KL_loss(d_student, d_teacher)

    # 计算最后标签loss
    final_loss = dice_bce_loss_re3(input0, class2one_hot(target, 2).float(), a, b)

    # loss=alpha*(la_loss+final_loss)+(1-alpha)*kd_loss
    # loss=alpha*(la_loss)+(1-alpha)*kd_loss
    loss = final_loss
    return loss, la_loss, kd_loss

def multi_loss3(input, target, device, a, b, size_average=True, aux_weight=1):
    aux_loss = []
    for aux in input:
        # b, _, h, w = aux.size()
        bceloss = bce_loss(aux, target)
        diceloss = dice_loss(aux, target)
        aux_loss.append(a*bceloss + b*diceloss)
    loss = sum(aux_loss)
    return loss, loss, loss
