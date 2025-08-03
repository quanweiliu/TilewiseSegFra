
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    torch.cuda.synchronize(img.device)
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        torch.cuda.synchronize(img.device)
        ret = torch.min(p1, p2)
        del p1, p2
        return ret
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        torch.cuda.synchronize(img.device)
        ret = torch.min(torch.min(p1, p2), p3)
        del p1, p2, p3
        return ret


def soft_dilate(img):
    torch.cuda.synchronize(img.device)
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
        del img1, delta
    return skel


def soft_cldice(y_true, y_pred, iter=2, smooth=1.0):
    skel_pred = soft_skel(y_pred, iter)

    # pred_show=np.array(skel_pred.cpu().squeeze().squeeze())
    # pred_show=pred_show*255
    # pred_show = pred_show.astype(np.uint8)
    # cv2.imshow("image", pred_show)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    tprec = (torch.sum(torch.mul(skel_pred, y_true)) + smooth) / (torch.sum(skel_pred) + smooth)
    skel_true = soft_skel(y_true, iter)
    tsens = (torch.sum(torch.mul(skel_true, y_pred)) + smooth) / (torch.sum(skel_true) + smooth)
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
    soft_cldice_loss=1-cl_dice.mean()

    return soft_cldice_loss


def soft_dice(y_true, y_pred):
    smooth = 0.0001  # 防止0除
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(torch.mul(y_pred, y_true))

    dice_score=(2. * intersection + smooth) / (torch.sum(y_pred) + torch.sum(y_true) + smooth)
    soft_dice_loss = 1 - dice_score.mean()

    return soft_dice_loss

def bce_loss(input, target):
    # input = F.sigmoid(input)
    _, _, h, w = input.size()
    # _, ht, wt = target.size()
    _, _, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsequeeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)
        target = target.long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = target.contiguous().view(-1, 1)
    input = input.contiguous().view(-1, 1)
    target = target.float()
    # loss = F.binary_cross_entropy_with_logits(input, target, class_weight)
    loss = F.binary_cross_entropy(input, target)
    return loss


def bce_loss2(input, target,device_num, weight =None,size_average = True):

    # input = F.sigmoid(input)
    _, _, h, w = input.size()
    # _, ht, wt = target.size()
    _,_, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsequeeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
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
    loss = F.binary_cross_entropy(input, target,class_weight)
    return loss


def dice_cldice_loss(input, target, a, b, device, size_average = True):
    soft_dice_loss = soft_dice(target, input)
    soft_cldice_loss=soft_cldice(target,input,iter=20, smooth=1)
    loss=a*soft_dice_loss+b*soft_cldice_loss
    return loss,soft_dice_loss,soft_cldice_loss


def bce_cldice_loss(input, target, a, b, c, device, weight=None, size_average = True):
    bceloss=bce_loss2(input, target,device_num=device,weight=weight,size_average=size_average)
    soft_cldice_loss=soft_cldice(target,input,iter=20, smooth=1)
    soft_dice_loss = soft_dice(target, input)
    loss = a * bceloss + b * soft_cldice_loss+c*soft_dice(target, input)
    return loss,bceloss,soft_cldice_loss


if __name__ == "__main__":
    device = 'cuda'
    # img=tifffile.imread(r'G:\Erie_crop\Erie_v1\multi_data\train\mask\buffalo_0_1.tif')
    img = cv2.imread(r'G:\Erie_crop\Erie_v1\mask_new\buffalo_7_5.tif', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    # img1 = np.where(img > 120, 0.2, 0.8)
    # img2 = np.where(img > 120, 0.8, 0.2)
    # img = np.stack((img1, img2))
    img = np.where(img > 120, 1,0)
    img = torch.from_numpy(img).to(device)
    pred = torch.unsqueeze(img, dim=0).unsqueeze(dim=0).float()

    # img=tifffile.imread(r'G:\Erie_crop\Erie_v1\multi_data\train\mask\buffalo_0_1.tif')
    img = cv2.imread(r'G:\Erie_crop\Erie_v1\mask_new\buffalo_7_5.tif', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    img = np.where(img > 120, 1, 0)
    img = torch.from_numpy(img).to(device)
    target = torch.unsqueeze(img, dim=0).unsqueeze(dim=0).float()


    loss = soft_cldice(target, pred,iter=20)

    print(f'loss={loss.item()}, gredient={loss}')
