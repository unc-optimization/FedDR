import os
import argparse
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
from random import Random

import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from asyncfeddr.server import *
from asyncfeddr.worker import *



def proc_run(rank, size, init_file_name, args):

    mode = args.update_mode
    cwd = os.getcwd()
    if mode == 'synchronous':
        file_path ='file://' + cwd +  '/hostname/'+ init_file_name
    else:
        file_path ='file://' + cwd +  '/hostname/'+ init_file_name

    dist.init_process_group('gloo',init_method=file_path, rank=rank, world_size=size, timeout=datetime.timedelta(0, args.runtime))

    args.rank = rank

    if args.rank == 0:
        server_main(args)
    else:
        worker_main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--rank', type=int, metavar='N', help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=11, metavar='N', help='size of the world')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--update-mode', type=str, default='synchronous', metavar='N', help='update mode: synchronous or asynchronous')
    parser.add_argument('--seed', type=int, default='1234', metavar='N', help='random seed for reproducibility')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='N', help='alpha parameter for FedDR')
    parser.add_argument('--eta', type=float, default=1.0, metavar='N', help='eta parameter for FedDR')
    parser.add_argument('--runtime', type=float, default=3600.0, metavar='N', help='total runtime in seconds')
    parser.add_argument('--worker-per-round', type=int, default=0, metavar='N', help='number of workers used for update')
    parser.add_argument('--worker-max-delay', type=float, default=20, metavar='N', help='max delay added after each update')
    args = parser.parse_args()

    mode = args.update_mode
    cwd = os.getcwd()
    if mode == 'synchronous':
        file_name ='feddr_' + args.dataset
    else:
        file_name ='asyncfeddr_' + args.dataset
    file_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # launch multiprocessing nodes
    size = args.world_size
    processes = []
    for rank in range(size):
        p = Process(target=proc_run, args=(rank, size, file_name, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # remove hostname files
    os.remove('hostname/'+ file_name)



