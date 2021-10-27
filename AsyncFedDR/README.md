# Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization

## Introduction

This package is the implementation of AsyncFedDR algorithm, an asynchronous variant of FedDR.

This package is built upon [distbelief](https://github.com/ucla-labx/distbelief).


## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you find it helpful, please consider citing the following publication:

* Q. Tran-Dinh, N. H. Pham, D. T. Phan, and L. M. Nguyen. **[FedDR -- Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization](https://arxiv.org/abs/2103.03452)**. <em>Conference on Neural Information Processing Systems</em>, 2021.

Feel free to send feedback and questions about the package to our maintainer Nhan H. Pham at <nhanph@live.unc.edu>.

Implementation of *asyncFedDR* an asynchronous variant of Federated Douglas Rachford algorithm.

This package is built upon [distbelief](https://github.com/ucla-labx/distbelief).


## Installation/Development instructions

Run `pip install .` or `pip install --user .`

## Install dependencies

```
pip install -r requirements.txt  
```

## Dataset preparation

For MNIST dataset, `torchvision` will download the dataset automatically. For FEMNIST dataset, follow the instruction [here](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) to generate the train and test folder using command
```
./preprocess.sh -s niid --sf 0.05 -k 50 -t sample
```

Create a folder name `FEMNIST` under `data` and put the `train` and `test` folder in there. In particular, the folder in `data` should be
```
.
+-- FEMNIST
|   +-- train
|   +-- test
```

## Running the example

First create `log` folder if it has not been created.
```
mkdir log
```

Then we can view all options:

```
python main.py -h
```

Examples:

1. Train MNIST with 15 users in synchronous mode for 1 hour (3600 seconds):

```
python main.py --world-size 16 --dataset MNIST --update-mode synchronous --eta 50.0 --runtime 3600
```

2. Train FEMNIST with 15 users in asynchronous mode for 1 hour (3600 seconds):

```
python main.py --world-size 16 --dataset FEMNIST --update-mode asynchronous --eta 50.0 --alpha 1.0 --runtime 3600
```

*Notes*: in case the threads are not automatically terminate, you can run
```
sh kill_process.sh
```
to terminate any running process.