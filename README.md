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

## How to run

There are two subfolders:
- `FedDR` provides examples to compare synchronous FedDR with FedAvg, FedProx, and FedPD.
- `AsyncFedDR` provides examples to compare between synchronous and asynchronous variants of FedDR.

Please see the `README.md` file in corresponding folder to get instructions of how to run these experimnets.