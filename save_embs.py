import pathlib
import sys
import math
from collections import defaultdict, OrderedDict, Counter
import os
import timeit
import traceback
import pickle

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_sum
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, RandomSampler, BatchSampler
from torch_geometric.data import DataLoader as GraphDataloader
from torch_geometric.utils import subgraph

import data_utils as utils
from GraphRegion import GraphRegion
from preprocessing import SpatialRegion
from constants import Constants
from dataloader import TrajDataset, BucketSamplerLessOverhead, BucketSampler, collate_fn
# from dataloader_nowaiting import TrajDataset, BucketSamplerLessOverhead, collate_fn
from config import Config, AverageMeter
# from model import TrajectoryEncoder, graphregion
# from model import weights_init_classifier, DestinationProjHead, AugProjHead, MaskedProjHead, PermProjHead
# from model import compute_destination_loss, compute_aug_loss, compute_mask_loss, compute_perm_loss
from model_rnnbased import TrajectoryEncoder, graphregion
from model_rnnbased import weights_init_classifier, DestinationProjHead, AugProjHead, MapembProjHead, MaskedProjHead, PermProjHead
from model_rnnbased import compute_destination_loss, compute_aug_loss, compute_mask_loss, compute_perm_loss


from transformation import Reversed, Masked, Augmented, Destination, Normal
# Permuted

import argparse


######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='save embs')


parser.add_argument('--model_path', type=str, help='')
parser.add_argument('--dbname', type=str, help='')

parser.add_argument('--dm9', action='store_true',default=False,  help = 'dm9 directory')
parser.add_argument('--task', type=str, help='')
parser.add_argument('--cuda', default=1, type=int, help='hop size')


global opts
opts = parser.parse_args()

######################################################################


data_dir = pathlib.PosixPath("data/")
dset_name = "porto"

train_fname = "data/porto/merged_train_edgeattr.h5"
val_fname = "data/porto/merged_val_edgeattr.h5"


def save_embeddings(model_path, dbname, task, cuda, ):
    
    config = Config(graphregion.vocab_size)
    config.CUDA = cuda
    CUDA = config.CUDA

    use_gpu = torch.cuda.is_available()
    config.device = torch.device('cuda:{}'.format(CUDA)) if use_gpu else torch.device('cpu')
    config.dtype = torch.float32

    # init model
    # main encoder
    if "traj_encoder" in locals():
        del traj_encoder
    

    traj_encoder = TrajectoryEncoder(config)
        

    # mapemb + aug
    config.path_state =  model_path
    if opts.dm9:
        savedmodels_dict = torch.load(os.path.join("models/dm9", config.path_state),
                                  map_location=torch.device('cuda:{}'.format(CUDA))) 
    else : 
        savedmodels_dict = torch.load(os.path.join("models", config.path_state),
                                  map_location=torch.device('cuda:{}'.format(CUDA))) 
        
    traj_encoder.load_state_dict(savedmodels_dict['TrajectoryEncoder'],)
    traj_encoder.to(config.device, config.dtype)
    
    
    db_paths = [_ for _ in os.listdir("data/porto/") if (dbname in _) and  ('emb' not in _)]
    db_paths = sorted(db_paths, key = lambda x: int(x.split('_')[-2]))
    print("db_paths: ", db_paths)
    
    with torch.no_grad():
        for db_path in db_paths:
            db = torch.load("data/porto/{}".format(db_path))
            db_emb = []

            for i, pair in enumerate(db,1):
                batch = Batch.from_data_list(pair)
                _,_, traj_emb = traj_encoder(batch)
                db_emb.append(traj_emb[:,-1,:]) # (pair:2,hid:100)
                if i % 1000 == 0:
                    print("processed {} data".format(i))

            torch.save(db_emb, "data/porto/{}".format('emb_'+task+db_path))
            
            
if __name__ == '__main__':
    save_embeddings(model_path = opts.model_path,
                dbname = opts.dbname,
                task = opts.task,
                cuda = opts.cuda,
               )
#     save_embeddings(model_path = "traj_rnnbased_perm_num_hid_layer_2_e3.pt",
#                 dbname = 'nei_div3',
#                 task = 'perm',
#                 cuda = 0,
#                )

