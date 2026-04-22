import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup_epochs, total_epochs, start_ratio=0.1, last_epoch=-1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warmup_epochs: 前多少个 epoch 进行 warmup
            - total_epochs: 总训练 epoch 数
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
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        epoch = self.last_epoch  # 当前 epoch 计数（从 0 开始）

        # ---------- Warmup 阶段 ----------
        if epoch < self.warmup_epochs:
            # 线性 warmup
            warmup_progress = epoch / float(self.warmup_epochs)
            scale = self.start_ratio + (1 - self.start_ratio) * warmup_progress
            lr = self.lr_min + (self.lr_max - self.lr_min) * scale

        # ---------- Cosine 阶段 ----------
        else:
            cosine_progress = (epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(np.pi * cosine_progress))

        # print(lr)
        return [lr for _ in self.base_lrs]


class WarmUp(_LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup_epoch, warmup_mode='linear', start_ratio=0.1, last_epoch=-1):
        """
        Description:
            - get warmup scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warmup_epoch: 前多少个 epoch 进行 warmup
            - total_epochs: 总训练 epoch 数
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
        self.warmup_epoch = warmup_epoch
        self.warmup_mode = warmup_mode
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        epoch = self.last_epoch  # 当前 epoch 计数（从 0 开始）

        # ---------- Warmup 阶段 ----------
        # print(epoch, self.warmup_epoch, self.warmup_mode)
        if epoch < self.warmup_epoch:
            # 线性 warmup
            warmup_progress = epoch / float(self.warmup_epoch)
            
            if self.warmup_mode == 'linear':
                scale = self.start_ratio + (1 - self.start_ratio) * warmup_progress
            elif self.warmup_mode == 'exp':
                scale = self.start_ratio + (1 - self.start_ratio) * (1 - np.exp(-5 * warmup_progress))
            else:
                raise ValueError(f"Unsupported warmup mode: {self.warmup_mode}")
            lr = self.lr_min + (self.lr_max - self.lr_min) * scale
        
        else:
            lr = self.lr_max
        return [lr for _ in self.base_lrs]


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
    # cosine_lr = WarmupCosineLR(optimizer, 1e-5, LR, warm_up, epochs, 0)
    cosine_lr = WarmUp(optimizer, 1e-5, LR, warm_up)
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
    plt.savefig('warmup_cosine_lr.png')
