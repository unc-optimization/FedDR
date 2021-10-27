import torch
from torch.optim.optimizer import Optimizer, required
from asyncfeddr.utils.serialization import ravel_model_params, unravel_model_params
from asyncfeddr.utils.messaging import MessageCode, MessageListener, send_message

class PerturbedSGD(Optimizer):
    """Perturbed SGD optimizer"""

    def __init__(self, params, lr=0.01, mu=0.0):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, mu=mu)
        self.idx = 0

        super(PerturbedSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['v_star'] = torch.zeros_like(p)
                
    def update_v_star(self, v_star):

        current_index = 0

        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                size = p.data.size()

                state = self.state[p]
                state['v_star'] = (v_star[current_index:current_index+numel].view(size)).clone()

                current_index += numel

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

 
        # internal sgd update
        for group in self.param_groups:
            #get the lr
            lr = group['lr']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data + mu*(p.data - state['v_star'])
                p.data.add_(d_p, alpha=-lr)
        
        self.idx += 1
        return loss
