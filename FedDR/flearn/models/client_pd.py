import numpy as np
import copy

class Client_PD(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, train_model=None, options=None):
        self.model = train_model
        self.id = id # integer
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])

        if options is not None:
            self.eta = options.get('eta', 1.0)
            self.alpha = options.get('alpha', 0.9)
        else:
            self.eta = 1.0
            self.alpha = 0.9

        self.first_update = True

        self.x_0 = self.get_params()
        self.lbd_i = [np.zeros_like(x) for x in self.get_params()]
        
    def set_params(self, model_params):
        '''initialize model parameters'''
        if self.first_update:
            self.model.set_params(model_params)
            self.first_update = False

        for i in range(len(model_params)):
            self.x_0[i] = model_params[i]

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def solve_inner_dr(self, num_epochs=1, batch_size=10, debug=False):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size

        self.model.optimizer.set_params(self.x_0, self.model)
        self.model.optimizer.update_lbd(self.lbd_i, self.model)

        # x_i update
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size

        # lbd update
        local_params = self.model.get_params()
        for i in range(len(self.lbd_i)):
            self.lbd_i[i] += (local_params[i] - self.x_0[i])/self.eta
            self.x_0[i] = local_params[i] + self.eta*self.lbd_i[i]

        comp += self.model.size*2

        if debug:
            print('aaa',self.lbd_i[0], local_params[0][0][0], self.x_0[0][0][0])
        
        return (self.num_samples, self.x_0), (bytes_w, comp, bytes_r)

    # def solve_iters(self, num_iters=1, batch_size=10):
    #     '''Solves local optimization problem

    #     Return:
    #         1: num_samples: number of samples used in training
    #         1: soln: local optimization solution
    #         2: bytes read: number of bytes received
    #         2: comp: number of FLOPs executed in training process
    #         2: bytes_write: number of bytes transmitted
    #     '''

    #     bytes_w = self.model.size
    #     soln, comp = self.model.solve_iters(self.train_data, num_iters, batch_size)
    #     bytes_r = self.model.size
    #     return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
