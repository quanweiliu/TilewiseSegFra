import copy
import logging 
from ptsemseg.schedulers2.schedulers import *
from ptsemseg.schedulers2.warmuplr import WarmupCosineLR, WarmUp

# from schedulers import *
# from warmuplr import WarmupCosineLR, WarmUp

logger = logging.getLogger('ptsemseg')

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'step_lr': StepLR,
                 'multi_step': MultiStepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR,
                 'warm_up': WarmUp,
                 'warmupcosine_lr':WarmupCosineLR,
                 }

def get_warmup(optimizer, scheduler_dict):
    if scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)

    scheduler_dict_=copy.deepcopy(scheduler_dict['training']['warm_up'])

    s_type = scheduler_dict_['name']
    scheduler_dict_.pop('name')

    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            scheduler_dict))

    return key2scheduler[s_type](optimizer, **scheduler_dict_)


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)

    scheduler_dict_=copy.deepcopy(scheduler_dict['training']['scheduler'])

    s_type = scheduler_dict_['name']
    scheduler_dict_.pop('name')

    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            scheduler_dict))

    return key2scheduler[s_type](optimizer, **scheduler_dict_)

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

    optimizer = Adam(model.parameters(), lr=LR)
    # cosine_lr = WarmupCosineLR(optimizer, 1e-5, LR, warm_up, epochs, 0)
    # cosine_lr = WarmUp(optimizer, 1e-5, LR, warm_up)
    scheduler = get_scheduler(optimizer, {'training': {'scheduler': {'name': 'step_lr', 'step_size': 100, 'gamma': 0.8}}})

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