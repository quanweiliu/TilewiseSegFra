import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1,
                 gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
        #     return [base_lr for base_lr in self.base_lrs]
        # else:
        #     factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        #     return [base_lr * factor for base_lr in self.base_lrs]
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        return [base_lr * factor for base_lr in self.base_lrs]



if __name__=="__main__":
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(1, 10)

        def forward(self, x):
            return self.fc(x)

        # class

    lrs = []
    model = Net()
    LR = 0.001
    epochs = 800
    warm_up = 10
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, 100, 0.8)
    # scheduler = PolynomialLR(optimizer, 20800, 1, 0.3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=2,  # T_0就是初始restart的epoch数目
    #     T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
    #     eta_min=1e-5  # 最低学习率
    # )

    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        # scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, color='r')
    plt.text(0, lrs[0], str(lrs[0]))
    plt.text(epochs, lrs[-1], str(lrs[-1]))
    plt.show()
    plt.savefig('warmup_cosine_lr.png')

