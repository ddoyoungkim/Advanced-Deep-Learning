# General trajectory representationusing GNN and self-supervised learning

## About

## Installation
Requirements
  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Pytorch >= 1.2.0
  
## Usage
  - Download all codes (*\*.py*) and put them in the same folder
  - Download PORTO dataset from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
  - Open terminal in the same folder
  - You can build region map and road network using `preprocessing.py` and `GraphRegion.py`
  - You can make k-hop sub-graphs using `create_trainval_edit.py`
  - You can see trajectory self-supervised tasks in `transformation.py`
  - Run `"python train.py"` to train and validate the model
  - Run `"python finetune.py"` to finetune the pretrained model on downstream tasks.

## Hyperparameters:
- Please check the hyperparameters for training in `config.py`
- Please check the hyperparameters for finetuning in `finetune_config.py`
