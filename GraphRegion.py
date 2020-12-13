import h5py
import pandas as pd
import pathlib
import ast
import sys
from math import sin, ceil, floor
import numpy as np
from collections import defaultdict, Counter
from sklearn.neighbors import KDTree
import pickle

import torch
from torch_sparse import SparseTensor

import data_utils as utils
from preprocessing import SpatialRegion

class GraphRegion(SpatialRegion):
    def __init__(self, dataset_name, minlon, minlat, maxlon, maxlat, 
                 xstep, ystep, 
                 minfreq=50, maxvocab_size=50000, knn_k=5, 
                 vocab_start=4,
                 load_spatialregion=False
                ):
        # init numx, numy, and so on
        super(GraphRegion, self).__init__(dataset_name,
                                          minlon, minlat, maxlon, maxlat, 
                                          xstep, ystep, 
                                          minfreq=50, maxvocab_size=50000, 
                                          knn_k=5,vocab_start=4,)
            
    def save_graphregion_info(self, data_dir, path):
        pass
        
    def load_spatialregion_info(self, data_dir, fname):
        """
        load the following region info from .pkl file: 'cellcount', 'hotcell', 
        'hotcell2vocab', 'vocab2hotcell','vocab_size', 'hotcell_kdtree', 'built'
        7 in total

        load_spatialregion_info(data_dir, "portomap.pkl")
        """
        file_path = data_dir/self.dataset_name/"region_info"/fname
        info = pickle.load(open(file_path, "rb"))
        
        self.cellcount= info["cellcount"]
        self.hotcell= info["hotcell"]
        self.hotcell2vocab= info["hotcell2vocab"]
        self.vocab2hotcell= info["vocab2hotcell"]
        self.vocab_size= info["vocab_size"]
        self.hotcell_kdtree= info["hotcell_kdtree"]
        self.built= True
        
    def load_graphregion_info(self, data_dir, spatialregion_fname, adj_fname):
        """
        load the adj matrix
        """
        self.load_spatialregion_info(data_dir, spatialregion_fname)
        
        file_path = data_dir/self.dataset_name/"region_info"/adj_fname
        adjmatrix = torch.load(file_path)
#          for pkl file
        #         adjmatrix = pickle.load(open(file_path, "r"))
        self.adjmatrix = adjmatrix
        
    def make_adjmatrix(self, data_dir, fname, zerolen_tripids=None):
        """
        @fname :: trips f : "preprocessed_entire_porto.h5"
        ex) f["trips/{}".format(num)] where the num is bet 1~all including zerolen_trips
        
        make_adjmatrix(data_dir, "preprocessed_entire_porto.h5",)
        """
        file_path = data_dir/self.dataset_name/fname
        seq_vocabs = open(file_path, "r")
        
        # nodes ~ vocabs : self.vocab_start ~ self.vocab_size
        nodes = list(range(self.vocab_size))
        adj_matrix = np.zeros((len(nodes), len(nodes)),) # (nodes,nodes)
#         print(adj_matrix.shape)
        
        with h5py.File(file_path, 'r') as f:
            total_len = len(f["trips"].keys())
            for num in range(total_len):
                
                if num % 3000 == 2999 : 
                    print("Scanned {} trips out of {}".format(num+1, total_len))
                    
                if num+1 in set(zerolen_tripids): continue
                    
                trip = f["trips/"+str(num+1)][()] # nd.array (2,traj_len)
                
                if trip.shape[1] ==1 : continue # bc there cannot be any connention
                try:
                    trip = np.array(self.trip2seq(trip)) # seq of vocabs : (len_trip)
                    trip_pre, trip_curr = trip[:-1], trip[1:]
                    conn = np.array([[pre, curr] for pre, curr in zip(trip_pre, trip_curr)])
                    conn = np.unique(conn, axis=0) #(#conn, 2)
                    adj_matrix[conn[:,0],conn[:,1]] = 1
                        
                except:
                    conn = conn[np.all(conn!='UNK', axis=1)].astype(int) # (#conn, 2)
                    adj_matrix[conn[:,0],conn[:,1]] = 1
                
        adj_matrix = adj_matrix[self.vocab_start:,self.vocab_start:]
        adj_matrix[np.arange(adj_matrix.shape[0]),np.arange(adj_matrix.shape[0])] = 0
        self.adj_matrix = SparseTensor.from_dense(torch.from_numpy(adj_matrix))
        
        return adj_matrix
    
    @classmethod
    def save_adjmatrix(cls, adj_matrix, data_dir, ds_name, fname):
        """
        ds_name = "porto"
        fname = "preprocessed_entire_porto_sparseadjmatrix.pt"
        """
        path = data_dir/ds_name/"region_info"/fname
        torch.save(adj_matrix, path)


    
    