import os
import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
import torch.distributed as dist

import datetime
import pandas as pd

from asyncfeddr.utils.models import SimpleNetMNIST, SimpleNetFEMNIST
from asyncfeddr.utils.serialization import ravel_model_params, unravel_model_params
from asyncfeddr.utils.messaging import MessageCode, send_message

import torch.optim as optim
from asyncfeddr.optim.perturbed_sgd import PerturbedSGD
import time
import torchvision.models as models

from asyncfeddr.utils.dataset import partition_dataset


def extract_model(sender, message_code, parameter):
    if message_code == MessageCode.ParameterUpdate:
        return parameter, False
    elif message_code == MessageCode.Terminate:
        return parameter, True
    else:
        raise ValueError('undefined message code')

def worker_main(args):
    
    trainloader, testloader = partition_dataset(args)


    torch.manual_seed(args.seed)
    if args.dataset == 'MNIST':
        model = SimpleNetMNIST()
    elif args.dataset == 'FEMNIST':
        model = SimpleNetFEMNIST()

    optimizer = PerturbedSGD(model.parameters(), lr=args.lr, mu=1.0/args.eta)
    
    alpha = args.alpha
    
    # train
    model.train()

    # model size
    model_size = ravel_model_params(model).numel()

    # communication buffer
    m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)

    # FedDR local variables
    y_i = torch.zeros(model_size)
    x_hat = torch.zeros(model_size)
    x_i = ravel_model_params(model)

    criterion = nn.CrossEntropyLoss()

    while True:

        _ = dist.recv(tensor=m_parameter)


        latest_model, terminate = extract_model(  int(m_parameter[0].item()),
                                    MessageCode(m_parameter[1].item()),
                                    m_parameter[2:])

        if terminate:
            break

        # start local update
        start_time = datetime.datetime.now()

        # update y_i
        y_i = y_i + alpha*(latest_model - x_i)

        # update x_i
        optimizer.update_v_star(y_i)

        # loop over the dataset multiple times
        for epoch in range(args.epochs):

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # update x_i
        x_i = ravel_model_params(model)

        # update x_hat
        x_hat = 2*x_i - y_i

        end_time = datetime.datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # add a delay
        if args.worker_max_delay > 0:
            time.sleep(args.worker_max_delay*(args.rank-1)/args.world_size)

        # sending parameters to server
        send_message(MessageCode.ParameterUpdate, x_hat)

    # finish training
    print('Rank {:2} Finished Training'.format(args.rank))