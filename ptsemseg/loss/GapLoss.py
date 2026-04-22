import numpy as np
import tifffile
from skimage.morphology import skeletonize
import cv2
import torch
import torch.nn as nn
from torch.nn.modules.loss import *
import torch.nn.functional as F

'''
References:

Yuan, W.; Xu, W. GapLoss: A Loss Function for Semantic Segmentation of Roads in Remote Sensing Images. Remote Sens. 2022, 14, 2422. 
https://doi.org/10.3390/rs14102422
'''


class GapLoss(nn.Module):
    def __init__(self, K=60):
        super(GapLoss, self).__init__()
        self.K = K

    def forward(self, pred, target):
        # Input is processed by softmax function to acquire cross-entropy map L
        criterion = CrossEntropyLoss(reduction='none')
        L = criterion(pred, target)

        # Input is binarized to acquire image A
        A = torch.argmax(pred, dim=1)

        # Skeleton image B is obtained from A
        A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)

        # Generate endpoint map C
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).double()

        # Generate weight map W
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.double).to(pred.device)
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
        temp = torch.where(N == 0, 1, 0)
        W = N + temp

        loss = torch.mean(W * L)
        return loss


class BinaryGapLoss(nn.Module):
    def __init__(self, K=60):
        super(BinaryGapLoss, self).__init__()
        self.K = K
    
    def forward(self, pred, target):
        # Input is processed by softmax function to acquire cross-entropy map L
        # criterion = CrossEntropyLoss(reduction='none')
        # criterion=F.binary_cross_entropy(reduction=None)
        # L = criterion(pred, target)
        L = F.binary_cross_entropy(pred, target, reduction="none")

        # Input is binarized to acquire image A
        # A = copy.deepcopy(pred)
        # A[A >= 0.5] = 1
        # A[A < 0.5] = 0
        pred_np=pred.detach().cpu().numpy()
        A_np=pred_np
        A_np[A_np >= 0.5] = 1
        A_np[A_np < 0.5] = 0


        # Skeleton image B is obtained from A
        # A_np = A.detach().cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).float()
        # B = torch.unsqueeze(B, dim=1)

        # Generate endpoint map C
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to(pred.device)
        kernel[0][0][1][1] = 0
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).float()

        # Generate weight map W
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.float).to(pred.device)
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
        temp = torch.where(N == 0, 1, 0)
        W = N + temp

        loss = torch.mean(W * L)
        return loss


def gaploss(input, target, K=60, device=None):
    criterion = GapLoss(K)
    # alpha = 0.3
    # loss = (1-alpha)*bceloss + alpha*diceloss
    loss = criterion(input, target)
    return loss, loss, loss


class GapLoss_demonstration(nn.Module):
    def __init__(self, K=60):
        super(GapLoss_demonstration, self).__init__()
        self.K = K

    def forward(self, pred, target):
        # Input is processed by softmax function to acquire cross-entropy map L
        # criterion = CrossEntropyLoss(reduction='none')
        # criterion = F.binary_cross_entropy(reduction=None)
        L = F.binary_cross_entropy(pred, target,reduction="none")

        # Input is binarized to acquire image A
        # A = torch.argmax(pred, dim=1)
        pred_np=pred.detach().cpu().numpy()
        A_np=pred_np
        A_np[A_np >= 0.5] = 1
        A_np[A_np < 0.5] = 0
        A_show = torch.squeeze(pred, dim=0)
        A_show = A_show.cpu().numpy() * 255

        # Skeleton image B is obtained from A
        # A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)
        B_show = B.squeeze()
        B_show = B_show.cpu().numpy() * 255

        # Generate endpoint map C
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).double()

        # Generate weight map W
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.double).to(pred.device)
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
        N_show = N.squeeze()
        N_show = N_show.cpu().numpy()
        N_show = np.where(N_show > 255, 255, N_show)
        temp = torch.where(N == 0, 1, 0)
        W = N + temp
        W_show = np.where(N_show > 0, N_show, B_show)
        white = np.ones((512, 10), dtype=np.uint8) * 255
        show = np.hstack((A_show, white, B_show, white, W_show))
        show = show.astype(np.uint8)
        cv2.imshow("image", show)
        cv2.waitKey()
        cv2.destroyAllWindows()

        loss = torch.mean(W * L)
        return loss


# for debug
if __name__ == "__main__":
    device = 'cuda'
    # img=tifffile.imread(r'G:\Erie_crop\Erie_v1\multi_data\train\mask\buffalo_0_1.tif')
    img = cv2.imread(r'G:\Erie_crop\Erie_v1\mask_new\buffalo_18_3.tif', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    # img1 = np.where(img > 120, 0.2, 0.8)
    # img2 = np.where(img > 120, 0.8, 0.2)
    # img = np.stack((img1, img2))
    img = np.where(img > 120, 0.8,0.2)
    img = torch.from_numpy(img).to(device)
    pred = torch.unsqueeze(img, dim=0).float()

    # img=tifffile.imread(r'G:\Erie_crop\Erie_v1\multi_data\train\mask\buffalo_0_1.tif')
    img = cv2.imread(r'G:\Erie_crop\Erie_v1\mask_new\buffalo_18_3.tif', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    img = np.where(img > 120, 1, 0)
    img = torch.from_numpy(img).to(device)
    target = torch.unsqueeze(img, dim=0).float()

    criterion = GapLoss_demonstration()
    loss = criterion(pred, target)

    print(f'loss={loss.item()}, gredient={loss}')


