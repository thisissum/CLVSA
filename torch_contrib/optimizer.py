import torch
from torch import nn
from torch.optim import Optimizer
from itertools import chain
from torch.optim import Adam
import math
from collections import defaultdict


class Lookahead(Optimizer):
    """pytorch implementation of 'k step forward, 1 step back'
    args:
        base_optimizer: torch.optim, the base optimizer
        k: int, step of forward
        alpha: float, ratio of interpolation
    """

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        super(Lookahead, self).__init__()
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha

        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = base_optimizer.state

        for group in self.param_groups:
            group['counter'] = 0

    def update(self, group):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.zeros_like(fast.data)
                param_state['slow_param'].copy_(fast.data)
            slow = param_state['slow_param']
            slow += self.alpha * (fast.data - slow)
            fast.data.copy_(slow)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self.update(group)
            group['counter'] += 1
            if group['counter'] == self.k:
                group['counter'] = 0
        return loss


class RAdam(Optimizer):
    """Implementation of 'paper On the Variance of the Adaptive Learning Rate and Beyond'
    args:
        params: come from model.parameters()
        lr: learning rate, default 0.001
        beta1: first momentum estimate, default 0.9
        beta2: second momentum estimate, default 0.999
        epsilon: small num to avoid divided by zero, default 1e-8
        decay: l1 norm of params, default 0.0
    """
    def __init__(self, params, 
                       lr=0.001, 
                       beta1=0.9, 
                       beta2=0.999, 
                       epsilon=1e-8, 
                       decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        epsilon=epsilon, weight_decay=decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:

                if param.grad is None:
                    continue

                grad = param.grad.data.float()
                param_data = param.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients')

                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['mt'] = torch.zeros_like(param_data)
                    state['vt'] = torch.zeros_like(param_data)
                else:
                    state['mt'] = state['mt'].type_as(param_data)
                    state['vt'] = state['vt'].type_as(param_data)

                mt, vt = state['mt'], state['vt']
                beta1, beta2 = group['beta1'], group['beta2']

                vt.mul_(beta2).addcmul_(1-beta2, grad, grad)
                mt.mul_(beta1).add_(1-beta1, grad)

                state['step'] += 1
                buffer = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffer[0]:
                    rho, rt = buffer[1], buffer[2]
                else:
                    buffer[0] = state['step']
                    limit_rho_inf = 2 / (1 - beta2) - 1
                    rho = limit_rho_inf - 2 * \
                        state['step'] * beta2**state['step'] / \
                        (1 - beta2**state['step'])
                    buffer[1] = rho

                    if rho >= 5:
                        rt = math.sqrt((1-beta2**state['step']) * (rho-4) / (limit_rho_inf-4) * (
                            rho-2) / rho * limit_rho_inf / (limit_rho_inf-2)) / (1-beta1**state['step'])
                    else:
                        rt = 1.0 / (1 - beta1**state['step'])
                    buffer [2] = rt
                
                if group['weight_decay'] != 0:
                    param_data.add(-group['weight_decay'] * group['lr'], param_data)
                
                if rho >= 5:
                    denom = vt.sqrt().add_(group['epsilon'])
                    param_data.addcdiv_(-rt * group['lr'], mt, denom)
                else:
                    param_data.add_(-rt * group['lr'], mt)
                param.data.copy_(param_data)

        return loss

'''class TryModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(TryModel,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output

torch.manual_seed(1)

x = torch.randn(64,100)
y = torch.randn(64,5)
model = TryModel(100,5)
criterion = nn.MSELoss()
opt = RAdam(model.parameters())
losses_ahead = []
for i in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses_ahead.append(float(loss.data))
    opt.zero_grad()
    loss.backward()
    opt.step()


model = TryModel(100,5)
criterion = nn.MSELoss()
opt = Adam(model.parameters())
losses_adam = []
for i in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses_adam.append(float(loss.data))
    opt.zero_grad()
    loss.backward()
    opt.step()'''
