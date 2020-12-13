# GTR:General trajectory representation using GNN and self-supervised learning

## About
Source code and datasets of the paper [GTR:General trajectory representation using GNN and self-supervised learning]

Since the preprocessed datasets are too large to uploade on github, we will give you a intruction to preprocess the datasets from the scratch.

## Installation
Requirements
  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Pytorch >= 1.2.0
  
## Usage
  - Source Code:
    - Download all codes (*\*.py*) and put them in the same folder
  - Datasets:
    - Download PORTO dataset from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
    - Open terminal in the same folder
    - You can build region map and road network using `preprocessing.py` and `GraphRegion.py`
    - You can make k-hop sub-graphs using `create_trainval_edit.py`
  - Train:
    - You can see trajectory self-supervised tasks in `transformation.py`
    - You can add customized self-supervised tasks in `transformation.py` if you try other tasks
    - Run `"python train.py"` to train and validate the model
    - Run `"python finetune.py"` to finetune the pretrained model on downstream tasks.
    
## Hyperparameters:
  - Please check the hyperparameters for training in `config.py`
  - Please check the hyperparameters for finetuning in `finetune_config.py`
