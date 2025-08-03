import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]



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
    epochs = 150
    warm_up = 10
    optimizer = Adam(model.parameters(), lr=LR)
    cosine_lr = WarmupCosineLR(optimizer, 1e-5, LR, warm_up, epochs, 0)
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
        cosine_lr.step()
        # scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, color='r')
    plt.text(0, lrs[0], str(lrs[0]))
    plt.text(epochs, lrs[-1], str(lrs[-1]))
    plt.show()

