# FedMD_practice
An implementation of FedMD: Heterogenous Federated Learning via Model Distillation

Based on Tensorflow 2.0 and Python 3.7

## Introduction

### `pretrain.py`

Pretrain models with public data and their own private data

### `ac_train.py` & `FedMD.py`

The origin implementation of FedMD training

### `random_train.py` & `Fed_random.py`

Select stochastic batches of models to calculate consensus

## How to run

Run the pre_train for instance

```shell
python pretrain.py -conf conf/pre_train.json
```

