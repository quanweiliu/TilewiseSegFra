import os
import torch
import matplotlib.pyplot as plt


# myFmt = DateFormatter("%M:%S")
def plot_training_results(df_tr, df_ts, model_name, savefig_path=None):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df_tr['epoch'].values, df_tr['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('valLoss', color='tab:blue')
    ax2.plot(df_ts['epoch'].values, df_ts['valLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # ax3 = ax1.twinx()  
    # ax3.set_ylabel('trainingTime(sec)', color='tab:orange', labelpad=-32)
    # # ax3.yaxis.set_major_formatter(myFmt)
    # ax3.tick_params(axis="y",direction="in", pad=-23)
    # ax3.plot(df['epoch'].values, df['duration_train'].dt.total_seconds(), color='tab:orange')
    # ax3.tick_params(axis='y', labelcolor='tab:orange')

    # ax4 = ax1.twinx()  
    # ax4.set_ylabel('Kappa', color='tab:orange', labelpad=-232)
    # ax4.tick_params(axis="y",direction="in", pad=-203)
    # ax4.plot(df_ts['epoch'].values, df_ts['Kappa'].values, color='tab:orange')
    # ax4.tick_params(axis='y', labelcolor='tab:orange')

    ax5 = ax1.twinx()  
    ax5.set_ylabel('mIOU', color='tab:green', labelpad=-80)
    ax5.tick_params(axis="y",direction="in", pad=-80)
    ax5.plot(df_ts['epoch'].values, df_ts['mIOU'].values, color='tab:green')
    ax5.tick_params(axis='y', labelcolor='tab:green')

    plt.suptitle(f'{model_name} Training, Validation Curves')
    if savefig_path:
        plt.savefig(os.path.join(savefig_path, "metric"), bbox_inches='tight')
    else:
        plt.show()

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=4,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)