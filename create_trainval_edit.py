import h5py
import pandas as pd
import pathlib

import ast
import sys
import traceback
import numpy as np
import timeit

import torch
import multiprocessing
import ctypes
from itertools import repeat 
import data_utils as utils

from GraphRegion import GraphRegion
from preprocessing import SpatialRegion
from collections import defaultdict
import argparse

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='create subgraphs')


parser.add_argument('--train', action='store_true',default=False,  help = 'train src')
parser.add_argument('--val', action='store_true',default=False,  help = 'val src')

parser.add_argument('--name', type=str, help='')
parser.add_argument('--k_hop', default=1, type=int, help='hop size')
parser.add_argument('--processors', default=20, type=int, help='num of processors')

global opts
opts = parser.parse_args()

######################################################################



def str2seq(string):
    string = string.strip().replace("UNK", "0").split()
    vocabs = list(map(int, string))
    return np.array(vocabs)


data_dir = pathlib.PosixPath("data/")
dset_name = "porto"

spatialregion_fname = "preprocessed_entire_porto.pkl"
adj_fname = "preprocessed_entire_porto_sparseadjmatrix.pt"

graphregion = GraphRegion(dataset_name="porto",
                          minlon=-8.735152, minlat=40.953673,
                          maxlon=-8.156309, maxlat=41.307945,
                          xstep=100, ystep=100,)
graphregion.load_graphregion_info(data_dir,
                                  spatialregion_fname,
                                  adj_fname
                                 )

trips_path = "preprocessed_entire_porto.h5"
trips_vocab_path = ["train_unique.trg", "valid_unique.trg"]

file_path = data_dir/"porto"/trips_path

with open(data_dir/dset_name/trips_vocab_path[0], "r") as f:
    train = f.readlines()

with open(data_dir/dset_name/trips_vocab_path[1], "r") as f:
    val = f.readlines()

src = train if opts.train else val

k_paths = ["entire_porto_sparseadj1hop.pt", "entire_porto_sparseadj1_2hop.pt", 
           "entire_porto_sparseadj1_3hop.pt", "entire_porto_sparseadj1_4hop.pt"]
k_fname = k_paths[opts.k_hop-1]
k_adjmat =torch.load(data_dir/graphregion.dataset_name/"region_info"/k_fname)
k_adjmat= k_adjmat.to_dense()

shared_array_base = multiprocessing.Array(ctypes.c_int16,
                                        k_adjmat.size(0)*k_adjmat.size(1))
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(k_adjmat.size(0),  k_adjmat.size(1))
shared_array[:] = k_adjmat.numpy()[:]
# print('shared_array dtype: ', shared_array.dtype)

def vocab2offset_normalized(vocab):
    cell_id = graphregion.vocab2hotcell[vocab]
    yoffset = cell_id // graphregion.numx
    xoffset = cell_id % graphregion.numx
    
    #normalize
    yoffset = yoffset/graphregion.numy
    xoffset = xoffset/graphregion.numx
    return (xoffset, yoffset)

graphregion.vocab2offset_normalized = {vocab:vocab2offset_normalized(vocab)
                                       for vocab in range(graphregion.vocab_start,graphregion.vocab_size)}
            

def create_train_val(src, processors, path, shared_array=shared_array):
    
    pool = multiprocessing.Pool(processes=processors)
    batch_n = processors
    batch_size = len(src)//batch_n
    batch_number = 0
    print("Start creating subgraphs")
    print("Total batch: ", batch_n)
    print("Batch size: ", batch_size)
    
    for i in range(len(src)):
        if (i!=0) and (i%batch_size==0):
            if batch_number == batch_n-1:
                print("Distributing ", (batch_size*batch_number, len(src))) 
                pool.apply_async(create_train_val_batch, (batch_size*batch_number, None, path))
            else : 
                print("Distributing ", (batch_size*batch_number, i)) 
                pool.apply_async(create_train_val_batch, (batch_size*batch_number, i, path))
            batch_number += 1
            
    pool.close()
    pool.join()
    

def create_train_val_batch(s,e, path, shared_array=shared_array):# d_all_nodes, d_traj_nodes
    """
    create sub adjacency matrix centered on each trajectory

    @param f : hd.5 io
    @param src : train or val
    @param num : index

    ex) create_train_val(f, src, num)
    """
#     print(s,e,"\n")
    if e is None:
        e = len(src)
    
    adj_size = graphregion.vocab_size-graphregion.vocab_start
    
    # path is set as a global var
    # path = data_dir/graphregion.dataset_name/"train_val"/A1/
    # s_e.h5
    print("Creating {} file \n".format(str(path/"{}_{}_{}_{}.h5".format("train" if opts.train else "val",
                                                                        opts.name,
                                                                     s,e))))
    with h5py.File(path/"{}_{}_{}_{}.h5".format("train" if opts.train else "val",opts.name,s,e),"w") as f:
        for num in range(s,e):
            seq = src[num]
            
            try:
                trip = str2seq(seq) # UNK -> 0
                #(n_of_unks) : # UNK -> -4
                
                
                # vocabs starting from graphregion.vocab_start ################
                # UNK -> -4
                # consecutive unique traj_nodes
#                 traj_nodes = trip[trip!=0] # filter out UNK
                traj_nodes = np.array([trip[i] for i in range(len(trip)-1) if trip[i] != trip[i+1]] + [trip[-1]])
                if (len(traj_nodes) == 1) or (len(traj_nodes[traj_nodes!=0]) == 0) : 
                    print('traj of which len = 1 or full of UNK filtered out')
                    f["{}/edge_index".format(num)] = [-1]
                    f["{}/all_nodes".format(num)] = [-1]
                    f["{}/traj_nodes".format(num)] = [-1]
                    f["{}/edge_attr".format(num)] = [-1]
                    f["{}/traj_index".format(num)] = [-1]
                    print('done\n')
                    continue
#                 traj_nodes = np.array([traj_nodes[i] for i in range(len(traj_nodes)-1) if traj_nodes[i] != traj_nodes[i+1]] + [traj_nodes[-1]])
                
                # compute conn_nodes with the trajectory
                trip_unique = np.unique(traj_nodes)
                trip_unique -= graphregion.vocab_start
                trip_unique = trip_unique[trip_unique>=0] #filter out UNK
                ###############################################################                                     # compute conn_nodes with the trajectory
                conn_nodes = shared_array[:, trip_unique] # (#vocabs, len(trip))
                conn_nodes = np.sum(conn_nodes, axis=1) #(#vocabs,)
                conn_nodes[trip_unique] = 1
                conn_nodes = np.arange(adj_size)[conn_nodes != 0]

                # compute edge_index
                sub_adj = shared_array[conn_nodes,:][:,conn_nodes]
                sub_adj = np.transpose(np.stack(sub_adj.nonzero(), axis=1)) # (2,E)
#                 print('sub_adj dtype: ', sub_adj.dtype)
#                 print('sub_adj max: ', sub_adj.max())
                # memory manage
                sub_adj = sub_adj.astype(np.int32)
                conn_nodes = (conn_nodes+graphregion.vocab_start)
                conn_nodes = np.append(conn_nodes, 0).astype(np.int32) # add UNK
                trip = trip.astype(np.int32)
                
#                 print('sub_adj max: ', sub_adj.max())
                # compute edge_attr
                # consecutive unique traj_nodes : 
                conn_nodes2idx = {node:i for i,node in enumerate(conn_nodes)}
                edge_index2idx = {str(sub_adj[:,edge_i]):edge_i for edge_i in range(sub_adj.shape[1])}
                # vocab to idx
                conn_nodes_idx = [conn_nodes2idx[node] for node in conn_nodes]
                traj_nodes_idx = [conn_nodes2idx[node] for node in traj_nodes] 
                # traj_point_movement to edge_idx
                traj_idx_ = np.array(list(zip(traj_nodes_idx[:-1], traj_nodes_idx[1:])))
                traj_idx = np.array([edge_index2idx.get(str(edge),-1) for edge in traj_idx_])

                unk_traj_index = [i for i,idx in enumerate(traj_idx) if idx== -1]
                if len(unk_traj_index) > 0 :
                    print("inserting unk movement")
                    unk_movement = np.unique(traj_idx_[unk_traj_index], axis=0, return_index=True)[1]
                    unk_movement = np.array([traj_idx_[unk_traj_index][ind] for ind in sorted(unk_movement)]).transpose() # (2, unk_move)
                    # add to adj_matrix
                    sub_adj=np.concatenate((sub_adj,unk_movement), axis=1) # (2, len+unk_move)
#                     print('added unk_movement: ', unk_movement)
#                     print('conn nodes size: ', len(conn_nodes_idx))
                    
                    for col_i in range(unk_movement.shape[1]): #(2, unk_move)
                        
                        edge_index2idx[str(unk_movement[:,col_i])] = len(edge_index2idx)
                        
                    traj_idx[unk_traj_index] = [edge_index2idx[str(unk_edge)] for unk_edge in traj_idx_[unk_traj_index]]
#                     print(traj_idx_[unk_traj_index], '->', traj_idx[unk_traj_index])
#                     print("sub adjmatrix size: ", sub_adj.shape)
                    print("len of : unk_movement",len([i for i,idx in enumerate(traj_idx) if idx== -1]))
#                     print("done \n")
                
#                 print('sub_adj max: ', sub_adj.max())
                # edge attr by order
#                 edge_attr = np.zeros((sub_adj.shape[1],1),)
#                 edge_attr[traj_idx] = np.expand_dims(np.arange(1, len(traj_idx)+1), axis=1)
#                 edge_attr = edge_attr.astype(np.float32)
#                 edge_attr = defaultdict(tuple)
#                 for i, idx in enumerate(traj_idx):
#                     edge_attr[idx]+=(i,)
#                 edge_attr = np.array(edge_attr.items())
#                 print(num, "sub_adj: ", sub_adj.shape, "edge_attr: ", edge_attr.shape)
#                 print("trip_consecutive_unique.shape: ", len(traj_nodes), "max(edge_attr): ", np.max(edge_attr))
#                 print("#outofconns: ", np.setdiff1d(trip, conn_nodes))

                trip_index = [conn_nodes2idx[node] for node in trip]
                trip_index = np.array(trip_index).astype(np.int32)
#                 print(num, "trip: ", trip, trip.shape, "trip_index: ", trip_index, trip_index.shape)
                # int32 float32 int32 int32
                f["{}/edge_index".format(num)] = sub_adj # (2,E)
                f["{}/edge_attr".format(num)] = traj_idx.astype(np.int32)
                # to be data.x
                f["{}/all_nodes".format(num)] = conn_nodes #including trip vocabs : np.int16 : vocab 기준으로 회복
                f["{}/traj_nodes".format(num)] = trip # UNK->0 : 
                f["{}/traj_index".format(num)] = trip_index # UNK->0 : 
                
#                 print('sub_adj:',sub_adj.astype(np.uint8),
#                       'edge_attr:',
#                      edge_attr.astype(np.int16),
#                      'conn_nodes:', conn_nodes,
#                      'trip:', trip)
                
                if (num-s) % 3000 == 2999 :
                    print("Batch({} ~ {}) processing {}/{}({:.2f}%)".format(s,e,(num-s+1),(e-s),
                                                                        (num-s+1)/(e-s)))

            except Exception as e: 
                print(e)
                print("trip ", trip)
                print("traj_nodes_idx ", traj_nodes_idx)
                print("error")
                traceback.print_exc()
                f["{}/error".format(num)] = str(e)
                f["{}/edge_index".format(num)] = None
                f["{}/all_nodes".format(num)] = None
                f["{}/traj_nodes".format(num)] = None
                f["{}/edge_attr".format(num)] = None

# in my folder
# subgraphs_dirs = ["A1","A2","A3","A4"]
# path = data_dir/graphregion.dataset_name/"train_val"/"A1"

subgraph_dir = pathlib.PosixPath("/data/dykim")
path = subgraph_dir/graphregion.dataset_name/"A_subgraphs"

create_train_val(src, processors=opts.processors,
                 path=path, 
                 shared_array=shared_array)

