# FedMD_Imp
Try to improve the performance of FedMD

Based on Tensorflow 2.0 and Python 3.7

## Introduction

#### conf(Directory)

Use various json files to specify the type of training to be performed.

#### dataset(Directory)

Save Emnist dataset.

#### pretrain.py

Pretrain models with public data and their own private data.

#### model_train.py

According to the corresponding json file to run federal learning.

#### FedMD.py

Includes various federal learning models, such as FedMD and its variants.

- ##### FedMD

  The origin implementation of FedMD training.

- ##### FedMD_random

  Select stochastic batches of models to calculate consensus.

- ##### FedMD_simu

  Use cosine similarity function to calculate consensus.

- ##### FedMD_own

  Each client only learns on its private dataset.

## How to run

Run the pre_train for instance

```shell
python pre_train.py -conf conf/pre_train.json
```

