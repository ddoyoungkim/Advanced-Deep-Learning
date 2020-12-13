# GTR:General trajectory representation using GNN and self-supervised learning

## About
The paper is for KAIST AI602 term paper.

Source code and datasets of the paper [GTR:General trajectory representation using GNN and self-supervised learning](https://openreview.net/forum?id=fJ98BQQKQQ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dkaist.ac.kr%2FKAIST%2FFall2020%2FAI602%2FAuthors%23your-submissions))

Since the preprocessed datasets are too large to uploade on github, we will give you a intruction to preprocess the datasets from the scratch.

## Overview
Trajectory includes various aspects of information which are transiting distribution, speed, etc. The diverse aspects are often incompatible, simultaneously making two trajectories similar from one aspect and dissimilar from another aspect. It is supported by the fact that trajectory similarities are defined by various perspectives and each trajectory application uses different similarity measure.
However, there is no general measure which can handle multiple aspects at the same time, and be widely used for multiple applications. In this paper, we propose General Trajectory Representation(GTR) that contains multidimensional information of trajectory. Even though there has been attempts to learn trajectory representation([t2vec](https://ieeexplore.ieee.org/document/8509283), [NeutraJ](https://ieeexplore.ieee.org/document/8731427)), their representations are not designed to be used for various applications. GTR generates a general representation such that it contains diverse or even incompatible concepts of trajectory, thereby accommodating multiple downstream applications by virtue of its learnt domain knowledge. The proposed model features two novel modules specialized in trajectory data: (1) spatial-temporal(ST) encoder that extracts the ST features from the given trajectories; and (2) projection heads corresponding to ST self-supervised tasks that define what information the representation has to include.

## Data Sets
|  City  | # trajectory  | Link         |
| :----: | :-----------: | :----------: |
| Porto  | 1.7 Milllion  | [link](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data) |

At this point, we use *Porto* dataset only, but other trajectory datasets will be updated soon.

## Installation
Requirements
  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Pytorch >= 1.2.0
  
## Usage
  - Source Code:
    - Download all codes (*\*.py*) and put them in the same folder
  - Download Dataset and Preprocess Dataset
    - Download *PORTO* dataset from [link](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)
    - Put dataset in `/data/*PORTO*`
    - Open terminal in the root folder
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
