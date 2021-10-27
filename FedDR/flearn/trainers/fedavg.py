import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import History

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        '''initialize local models, build comp. graph'''

        # init local solver
        tf.reset_default_graph()
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # init base class
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging algorithm'''
        print('Training using FedAvg with {} workers ---'.format(self.clients_per_round))
        history = History(['ComRound','TrainLoss','GradNorm','TrainAcc','TestAcc','GradDiff','NumBytes'])

        tqdm.write(' ---------------------------------------------------------------------------------------')
        tqdm.write('| Com. Round | Train Loss | Grad Norm | Train Acc. | Test Acc. | Grad Diff. | Num Bytes |')
        tqdm.write(' ---------------------------------------------------------------------------------------')

        total_bytes_r = 0
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                model_len = process_grad(self.latest_model).size
                global_grads = np.zeros(model_len)
                client_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grad * num)
                global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

                grad_norm = np.sqrt(np.sum(np.square(global_grads)))

                difference = 0
                for idx in range(len(self.clients)):
                    difference += np.sum(np.square(global_grads - local_grads[idx]))
                difference = difference * 1.0 / len(self.clients)

                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
                test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
                train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
                tqdm.write('| {:10d} | {:10.5f} | {:9.2e} | {:10.5f} | {:9.5f} | {:10.2e} | {:9.2e} |'.format(i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r))
                history.update([i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r])

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  # buffer for receiving client solutions

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # update num bytes r
                total_bytes_r += stats[2]

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        # compute stats
        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)
        client_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            num, client_grad = c.get_grads(model_len)
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads = np.add(global_grads, client_grad * num)
        global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

        grad_norm = np.sqrt(np.sum(np.square(global_grads)))

        difference = 0
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference = difference * 1.0 / len(self.clients)

        stats = self.test() # have set the latest model for all clients
        stats_train = self.train_error_and_loss()

        train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
        test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
        train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
        tqdm.write('| {:10d} | {:10.5f} | {:9.2e} | {:10.5f} | {:9.5f} | {:10.2e} | {:9.2e} |'.format(i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r))
        history.update([i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r])
        
        return history.get_history(dataframe=True)