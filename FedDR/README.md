# Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization

## Introduction

This package is the implementation of FedDR algorithm and its variants along with other federated learning algorithms including FedAvg, FedProx, and FedPD.

This package is built upon [FedProx](https://github.com/litian96/FedProx).


## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you find it helpful, please consider citing the following publication:

* Q. Tran-Dinh, N. H. Pham, D. T. Phan, and L. M. Nguyen. **[FedDR -- Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization](https://arxiv.org/abs/2103.03452)**. <em>Conference on Neural Information Processing Systems</em>, 2021.

Feel free to send feedback and questions about the package to our maintainer Nhan H. Pham at <nhanph@live.unc.edu>.

## Preparation

### Dataset preparation

Checkout the README.md file in corresponding dataset name in `data` folder.

### Create virtual environment

Prerequisite: conda installation following [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```
conda env create -f environment.yml
```

## Run experiments

The scripts to reproduce the papers' results are in `run_script` folder.

For example, to compare **FedDR** with **FedAvg**, **FedProx**, and **FedPD** on the `FEMNIST` dataset, run

```
sh run_FEMNIST.sh
```

The results will be stored under `logs` folder.

## Plot results

See corresponding Jupyter notebooks under `notebooks` folder.