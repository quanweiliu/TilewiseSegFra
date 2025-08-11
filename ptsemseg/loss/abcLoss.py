import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss

# from soft_ce import SoftCrossEntropyLoss
# from joint_loss import JointLoss
# from dice import DiceLoss


class ABCLoss(nn.Module):

    def __init__(self, ignore_index=255, weight=0.4):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        self.weight = weight

    def forward(self, logits, labels):
        # print("self.training", self.training)  # True
        if self.training and len(logits) == 3:
            logit_main,  h2, h3  = logits
            h2 = F.interpolate(h2, size=labels.shape[1:], mode='bilinear', align_corners=True)
            h3 = F.interpolate(h3, size=labels.shape[1:], mode='bilinear', align_corners=True)
            loss_aux = self.aux_loss(h2, labels) + self.aux_loss(h3, labels)
            # loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
            loss = self.main_loss(logit_main, labels) + self.weight * loss_aux
        else:
            loss = self.main_loss(logits, labels)

        return loss

def abc_loss(input, target):
    criterion = ABCLoss()
    loss = criterion(input, target)
    return loss, loss, loss

if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = ABCLoss()
    loss = model(logits, targets)

    print(loss)