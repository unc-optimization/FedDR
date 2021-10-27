import numpy as np
import argparse
import importlib
import random
import os, time
import tensorflow as tf
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddr', 'fedpd']
DATASETS = ['FEMNIST', 'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']
REG_TYPE = ['none','l1_norm','l2_norm_squared','linf_norm']

MODEL_PARAMS = {
    'FEMNIST.ann': (26,),  # num_classes
    'synthetic.ann': (10, ) # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--eta',
                        help='constant for feddr;',
                        type=float,
                        default=1.0)
    parser.add_argument('--alpha',
                        help='constant for feddr;',
                        type=float,
                        default=0.9)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)
    parser.add_argument('--reg_type',
                        help='type of regularizer',
                        type=str,
                        choices=REG_TYPE,
                        default='none')
    parser.add_argument('--reg_coeff',
                        help='regularization parameter',
                        type=float,
                        default=0.01)
    parser.add_argument('--exp_id',
                        help='experiment ID',
                        type=str,
                        default='')
    parser.add_argument('--log_suffix',
                        help='string to append to file name',
                        type=str,
                        default='')

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    users, groups, train_data, test_data = dataset

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    start = time.time()
    history = t.train()
    end= time.time()
    print('Total Training Time: {:.2f} s'.format(end - start))

    alg_name = options['optimizer']
    if len(options['log_suffix']) > 0:
        name_list = [ alg_name,options['dataset'],options['log_suffix']]
    else:
        name_list = [ alg_name,options['dataset']]
        
    file_name = '_'.join(name_list)
    log_folder = 'logs'
    if options['exp_id'] is None or len(options['exp_id']) < 1:
        exp_id = 'test_' + options['dataset']
        log_folder = os.path.join(log_folder,exp_id)
    else:
        log_folder = os.path.join(log_folder,options['exp_id'])

    save_df(history, log_folder, alg_name, file_name)

def save_df(df, log_folder, alg_name, file_name):
    

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
        
    df.to_csv(os.path.join(log_folder, file_name +'.csv'), index=False) 
    
if __name__ == '__main__':
    main()
