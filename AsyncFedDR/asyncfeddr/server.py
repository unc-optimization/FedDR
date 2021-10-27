import os
import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import datetime

from sklearn.metrics import accuracy_score

from asyncfeddr.utils.messaging import MessageCode, MessageListener, send_message
from asyncfeddr.utils.serialization import ravel_model_params, unravel_model_params
from asyncfeddr.utils.models import SimpleNetMNIST, SimpleNetFEMNIST
from asyncfeddr.utils.file_io import *
from asyncfeddr.utils.dataset import partition_dataset

class ParameterServer(MessageListener):
    """ParameterServer"""
    def __init__(self, model):
        print(("Creating ParameterServer"))
        self.model = model
        self.last_time = datetime.datetime.now()
        self.message_cnt = [0,0,0]
        self.criterion = nn.CrossEntropyLoss()

        super().__init__(model)

    def evaluate(self, train=True):

        model = self.model
        if train:
            data_loader_list = self.train_loader_list
            data_size_list = self.train_size_list
        else:
            data_loader_list = self.test_loader_list
            data_size_list = self.test_size_list

        total_loss, total_acc = 0, 0
        model.eval()

        total_weight = 0.0

        for dataloader, size in zip(data_loader_list, data_size_list):

            tmp_loss, tmp_acc = 0, 0
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs, labels = data

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(predicted, labels)

                tmp_loss += loss.item()
                tmp_acc += accuracy

            num_batch = len(dataloader)

            total_weight += size

            total_loss += size*tmp_loss/num_batch
            total_acc += size*tmp_acc/num_batch

        model.train()

        return total_loss/total_weight, total_acc/total_weight

    def setup(self, args, train_loader_list, test_loader_list, num_workers):
        self.train_loader_list = train_loader_list
        self.test_loader_list = test_loader_list

        # compute sample size of each worker
        self.train_size_list, self.test_size_list = [], []
        for train_loader, test_loader in zip(self.train_loader_list, self.test_loader_list):
            self.train_size_list.append(len(train_loader.dataset))
            self.test_size_list.append(len(test_loader.dataset))

        self.num_workers = num_workers

        self.latest_model = ravel_model_params(self.model)
        self.model_size = self.latest_model.numel()

        self.float_to_byte = 4
        self.model_size_in_bytes = self.model_size*self.float_to_byte

        self.update_flag = torch.tensor([False for i in range(self.num_workers)])
        self.param_list = [self.latest_model.clone() for i in range(num_workers)]
        self.param_list = torch.stack(self.param_list)

        self.comm_round = 0
        self.log_interval = args.log_interval
        self.update_mode = args.update_mode
        self.args = args
        if args.worker_per_round == 0:
            self.worker_per_round = num_workers
        else:
            self.worker_per_round = min(args.worker_per_round, num_workers)

        self.terminate_flag = torch.tensor([False for i in range(self.num_workers)])


    def receive(self, sender, message_code, parameter):
        if message_code == MessageCode.ParameterUpdate:
            self.message_cnt[0] += 1
            self.update_flag[sender-1] = True
            self.param_list[sender-1] = parameter.clone()

    def check_worker(self, active_workers=None):
        if active_workers is None:
            return False not in self.update_flag
        else:
            return False not in self.update_flag[active_workers]

    def reset_flag(self):
        self.update_flag = torch.tensor([False for i in range(self.num_workers)])

    def sample_workers(self, worker_per_round=None):
        if worker_per_round is not None:
            return torch.randperm(self.num_workers)[:worker_per_round]
        else:
            return torch.arange(self.num_workers)

    def aggregation(self, param_list):
        new_model = torch.zeros(self.model_size)
        total_weight = 0.0
        for i in range(self.num_workers):
            sample_size = self.train_size_list[i]
            total_weight += sample_size
            new_model += sample_size*param_list[i,:]

        new_model /= total_weight

        return new_model

    def run(self, worker_per_round = None):
        self.running = True

        update = True

        print("Started Running!")

        print(' -------------------------------------------------------------------------')
        print('| TimeStamp | Train Loss | Train Acc. | Test Loss | Test Acc. | Num Bytes |')
        print(' -------------------------------------------------------------------------')

        # evaluate performance stats
        train_loss, train_acc = self.evaluate(train=True)

        # evaluate performance stats
        test_loss, test_acc = self.evaluate(train=False)

        total_bytes_r = 0

        print('| {:9.2f} | {:10.5f} | {:10.5f} | {:9.2e} | {:9.5f} | {:9.2e} |'.format(0, train_loss, train_acc, test_loss, test_acc, total_bytes_r))

        log_folder = './log'
        if self.update_mode == 'synchronous':
            file_name = 'feddr_' + self.args.dataset + '.csv'
        else:
            file_name = 'asyncfeddr_' + self.args.dataset + '.csv'

        log_file_path = os.path.join(log_folder,file_name)

        str_to_write = "TimeStamp,TrainLoss,TrainAcc,TestLoss,TestAcc,NumBytes"
        write_file(log_file_path, str_to_write, mode='w')

        str_to_write = '{},{},{},{},{},{}'.format(0, train_loss, train_acc, test_loss, test_acc, total_bytes_r)
        write_file(log_file_path, str_to_write, mode='a')

        start_time = datetime.datetime.now()
        last_time = start_time

        # send model to all users in asynchronous mode
        if self.update_mode != 'synchronous':
            for worker in range(self.num_workers):
                send_message(MessageCode.ParameterUpdate, self.latest_model, dst=worker+1)    

        # run till receive terminate signal
        while self.running:
            # synchronous mode
            if self.update_mode == 'synchronous':
                # generate a set of random workers
                if update:
                    active_workers = self.sample_workers(worker_per_round)

                    # send model to worker
                    for worker in active_workers:
                        send_message(MessageCode.ParameterUpdate, self.latest_model, dst=worker+1)    

                    update = False

                # receive local model
                _ = dist.recv(tensor=self.m_parameter)
                self.receive(int(self.m_parameter[0].item()),
                             MessageCode(self.m_parameter[1].item()),
                             self.m_parameter[2:])

                total_bytes_r += self.model_size_in_bytes

                # check if all workers has send their local model
                if self.check_worker():
                    # aggregate models
                    self.latest_model = self.aggregation(self.param_list)
                    unravel_model_params(self.model, self.latest_model)

                    update = True
                    self.reset_flag()

                    current_time = datetime.datetime.now()
                    if (current_time - start_time).total_seconds() >= self.args.runtime:
                        for worker in range(self.num_workers):
                            send_message(MessageCode.Terminate, self.latest_model, dst=worker+1) 
                        break

                    duration = current_time - last_time
                    if duration.total_seconds() >= 30:
                        last_time = current_time

                        # evaluate performance stats
                        train_loss, train_acc = self.evaluate(train=True)

                        # evaluate performance stats
                        test_loss, test_acc = self.evaluate(train=False)

                        time_stamp = (current_time-start_time).total_seconds()

                        print('| {:9.2f} | {:10.5f} | {:10.5f} | {:9.2e} | {:9.5f} | {:9.2e} |'.format((current_time-start_time).total_seconds(), train_loss, train_acc, test_loss, test_acc, total_bytes_r))
                        str_to_write = '{},{},{},{},{},{}'.format(time_stamp, train_loss, train_acc, test_loss, test_acc, total_bytes_r)
                        write_file(log_file_path, str_to_write, mode='a')

            # asynchronous mode
            else:
                _ = dist.recv(tensor=self.m_parameter)
                sender = int(self.m_parameter[0].item())
                self.receive(sender,
                             MessageCode(self.m_parameter[1].item()),
                             self.m_parameter[2:])

                total_bytes_r += self.model_size_in_bytes

                self.latest_model = self.aggregation(self.param_list)
                unravel_model_params(self.model, self.latest_model)

                current_time = datetime.datetime.now()
                if (current_time - start_time).total_seconds() >= self.args.runtime:
                    send_message(MessageCode.Terminate, self.latest_model, dst=sender) 
                    self.terminate_flag[sender-1] = True   
                else:
                    send_message(MessageCode.ParameterUpdate, self.latest_model, dst=sender)    
                
                if False not in self.terminate_flag:
                    break

                duration = current_time - last_time
                # only print when receive updates from slowest worker
                if sender == self.num_workers and duration.total_seconds() >= 60:
                    last_time = current_time

                    # evaluate performance stats
                    train_loss, train_acc = self.evaluate(train=True)

                    # evaluate performance stats
                    test_loss, test_acc = self.evaluate(train=False)

                    time_stamp = (current_time-start_time).total_seconds()

                    print('| {:9.2f} | {:10.5f} | {:10.5f} | {:9.2e} | {:9.5f} | {:9.2e} |'.format((current_time-start_time).total_seconds(), train_loss, train_acc, test_loss, test_acc, total_bytes_r))
                    # history.update([time_stamp, train_loss, train_acc, test_loss, test_acc, total_bytes_r])
                    str_to_write = '{},{},{},{},{},{}'.format(time_stamp, train_loss, train_acc, test_loss, test_acc, total_bytes_r)
                    write_file(log_file_path, str_to_write, mode='a')


        # calculate stats for once last time
        current_time = datetime.datetime.now()


        # evaluate performance stats
        train_loss, train_acc = self.evaluate(train=True)

        # evaluate performance stats
        test_loss, test_acc = self.evaluate(train=False)

        time_stamp = (current_time-start_time).total_seconds()

        print('| {:9.2f} | {:10.5f} | {:10.5f} | {:9.2e} | {:9.5f} | {:9.2e} |'.format((current_time-start_time).total_seconds(), train_loss, train_acc, test_loss, test_acc, total_bytes_r))
        str_to_write = '{},{},{},{},{},{}'.format(time_stamp, train_loss, train_acc, test_loss, test_acc, total_bytes_r)
        write_file(log_file_path, str_to_write, mode='a')

        print('Server finished training')


def server_main(args):

    torch.manual_seed(args.seed)
    if args.dataset == 'MNIST':
        model = SimpleNetMNIST()
    elif args.dataset == 'FEMNIST':
        model = SimpleNetFEMNIST()

    train_loader_list, test_loader_list = partition_dataset(args)

    server = ParameterServer(model=model)
    server.setup(args, train_loader_list, test_loader_list, args.world_size-1)

    server.run()