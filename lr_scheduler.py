from typing import Optional
from torch.optim.optimizer import Optimizer

class LrScheduler:

    def __init__(self, optimizer: Optimizer, max_iter, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 10, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * (self.iter_num / self.max_iter)) ** (-self.decay_rate)
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

