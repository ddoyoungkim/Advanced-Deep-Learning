import h5py
import pandas as pd
import pathlib


import sys
import numpy as np
import timeit
import random

import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import subgraph
from torch.utils.data import ConcatDataset, Sampler, RandomSampler, BatchSampler


from itertools import repeat 
import data_utils as utils

from torch_geometric.data import Data, Batch
from torch_scatter import scatter_sum

from collections import defaultdict
from GraphRegion import GraphRegion
from preprocessing import SpatialRegion

from constants import Constants

from collections import defaultdict, OrderedDict
import os

data_dir = pathlib.PosixPath("data/")
dset_name = "porto"

subgraph_dir = pathlib.PosixPath("/data/dykim")
subgraph_dir = subgraph_dir/dset_name/"A_subgraphs"
fnames = os.listdir(subgraph_dir)
trains = [h5 for h5 in fnames if h5.startswith("train_traj")]
vals = [h5 for h5 in fnames if h5.startswith("val_traj")]

# [(fname, start_num), ...]
trains = sorted([(f, int(f.split("_")[-2])) for f in trains], key= lambda x: x[1])
vals = sorted([(f, int(f.split("_")[-2])) for f in vals], key= lambda x: x[1])

# with h5py.File("data/porto/merged_train_index.h5", "w") as f :
#     for fname, start in trains:
#         f['/link_{}'.format(start)] = h5py.ExternalLink(str(subgraph_dir/fname),'/')

# with h5py.File("data/porto/merged_val_index.h5", "w") as f :
#     for fname, start in vals:
#         f['/link_{}'.format(start)] = h5py.ExternalLink(str(subgraph_dir/fname),'/')

merged_trains = h5py.File("data/porto/merged_train_index.h5", "r")
merged_vals = h5py.File("data/porto/merged_val_index.h5", "r")
components = ['edge_index', 'all_nodes', 'traj_nodes', 
              'edge_attr','traj_index']

class TrajDataForPermMasked(Data):
    def __init__(self, x=None, edge_index=None,
                 edge_attribute=None, edge_attribute_len=None,
                 tm_index=None, tm_len=None, traj_vocabs=None,
                 traj_len=None, y=None):
        super(TrajDataForPermMasked, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attribute = edge_attribute
        self.edge_attribute_len = edge_attribute_len
        self.tm_index = tm_index
        self.tm_len = tm_len
        self.traj_vocabs = traj_vocabs
        self.traj_len = traj_len
        self.y = y

    def __inc__(self, key, value):
        if key=='edge_attribute':
            return self.edge_index.size(1)
        if key=='edge_attribute_len':
            return 0
        if key=='tm_index':
            return self.num_nodes
        if key=='tm_len':
            return 0
        if key=='traj_vocabs':
            return 0
        if key=='traj_len':
            return 0

        else :
            return super(TrajDataForPermMasked,self).__inc__(key,value)

class TrajDataForAug(TrajDataForPermMasked):
    def __init__(self, x=None, edge_index=None,
                 edge_attribute=None, edge_attribute_len=None,
                 tm_index=None, tm_len=None, traj_vocabs=None,
                 traj_len=None, y=None):
        super(TrajDataForAug, self).__init__(x=x,
                                                           edge_index=edge_index,
                                                           edge_attribute=edge_attribute,
                                                           edge_attribute_len=edge_attribute_len,
                                                           tm_index=tm_index, tm_len=tm_len, 
                                                           traj_vocabs=traj_vocabs,
                                                           traj_len=traj_len, y=y)
    def __inc__(self, key,value):
        if key == 'y':
            return 0
        else : 
            return super(TrajDataForAug,self).__inc__(key,value)        

class TrajDataForDestination(TrajDataForPermMasked):
    def __init__(self, x=None, edge_index=None,
                 edge_attribute=None, edge_attribute_len=None,
                 tm_index=None, tm_len=None, traj_vocabs=None,
                 traj_len=None, y=None):
        super(TrajDataForDestination, self).__init__(x=x,
                                                           edge_index=edge_index,
                                                           edge_attribute=edge_attribute,
                                                           edge_attribute_len=edge_attribute_len,
                                                           tm_index=tm_index, tm_len=tm_len, 
                                                           traj_vocabs=traj_vocabs,
                                                           traj_len=traj_len, y=y)
    def __inc__(self, key,value):
        if key == 'y':
            return self.num_nodes
        else : 
            return super(TrajDataForDestination,self).__inc__(key,value)
        
class TrajDataset(Dataset):
    def __init__(self, file_path="data/porto/merged_train.h5", 
                 n_samples=1133657, n_processors=36,transform=None,
                 split='train'
                ):
        """
        h5py.File("data/porto/merged_train.h5", "r")
        default)
        n_trains=1133657,n_vals=284997, 
        train_processors=36, val_processors=9,

        """
        samples_per_file = n_samples//n_processors 
        self.samples2filelink = {i:i//samples_per_file if i//samples_per_file != n_processors else (n_processors-1) 
                            for i in range(n_samples)}
        self.data = h5py.File(file_path, "r")
        
        links = list(self.data.keys())
        links = sorted([(link,int(link.split('_')[1])) for link in links], key=lambda x:x[1])
        self.links = links
        
        self.n_samples = n_samples
        self.n_processors = n_processors
        
        self.split=split
        self.transform = transform
        
    def __getitem__(self, index):
        """
        index is a trajectory number
        """
        link = self.links[self.samples2filelink[index]]
        edge_index = torch.from_numpy(self.data["{link:}/{num:}/{component:}".format(link=link[0],
                                                  num=index,
                                                  component=components[0])][()]).to(torch.long)
        all_nodes = self.data["{link:}/{num:}/{component:}".format(link=link[0],
                                                  num=index,
                                                  component=components[1])][()]
        traj_nodes = self.data["{link:}/{num:}/{component:}".format(link=link[0],
                                                  num=index,
                                                  component=components[2])][()]
        __edge_attr = torch.from_numpy(self.data["{link:}/{num:}/{component:}".format(link=link[0],
                                                  num=index,
                                                  component=components[3])][()]).to(torch.long)
        traj_index = torch.from_numpy(self.data["{link:}/{num:}/{component:}".format(link=link[0],
                                                  num=index,
                                                  component=components[4])][()]).to(torch.long)
        
        if all_nodes[0] == -1:
#             print("all_nodes -1", index)
            return None

        if self.transform is None :
            data = TrajDataForPermMasked(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
            return data
            
    
        if (len(__edge_attr) > 10 ) & (self.transform is not None) :
            transform_name = self.transform.__class__.__name__.lower()
            
            if isinstance(self.transform, tuple or list) : # multiple transforms on the same data
                # e.g. self.transform = (Permuted(), Masked(), Augmented(), Destination(),)
                trsf_names = list(map(lambda x: x.__class__.__name__.lower(),
                                      self.transform))
                data_list = []
                for i, trsf in enumerate(trsf_names):
                    if trsf == 'augmented':
                        data = TrajDataForAug(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                        data_list.append(self.transform[i](data))
                    elif trsf == 'destination':
                        data = TrajDataForDestination(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                        data_list.append(self.transform[i](data))
                    elif (trsf == 'reversed') or (trsf == 'permuted') or (trsf == 'normal') or (trsf == 'masked'):
                        
                        data = TrajDataForPermMasked(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                        data_list.append(self.transform[i](data))

                    else : 
                        raise ValueError("Not valid transformation! -- message from Doyoung") 
                                         
                assert len(data_list) == len(self.transform)
                return data_list # list
                                         
                                         
            transform_name = self.transform.__class__.__name__.lower()
            
            if ('destination' in transform_name): #'Destination'

                data = TrajDataForDestination(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                return self.transform(data)
            elif ('aug' in transform_name): #'augmentation'

                data = TrajDataForAug(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                return self.transform(data)
            else : #'perm' or 'normal' or 'mask'
                data = TrajDataForPermMasked(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1),
                        edge_index= edge_index.to(torch.long),
                        edge_attribute= torch.tensor(__edge_attr, dtype=torch.long),
                        edge_attribute_len= torch.tensor(len(__edge_attr), dtype=torch.long).unsqueeze(-1),
                        tm_index = None,
                        tm_len= None,
                        traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long),
                        traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
                )
                return self.transform(data)

        else : # self.transform is not None and length is not > 10
#             print("length < 10 ", index)
            return None
        
    def __len__(self):
        return self.n_samples
        
     
def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
    assert length_bucket_step > 1.0
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * length_bucket_step))
    return boundaries


def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket,
                    length_bucket_step,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
    max_length = max_length or batch_size # smaller one
    if max_length < min_length:
        raise ValueError("max_length must be greater or equal to min_length")
   
    boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                    length_bucket_step)
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    max_length *= length_multiplier
   
    batch_sizes = [
        max(1, batch_size // length) for length in boundaries + [max_length]
    ]
#     print(batch_sizes)
    max_batch_size = max(batch_sizes)
    highly_composite_numbers = [
        1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
        2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
        83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
        720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
        7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
        36756720, 43243200, 61261200, 73513440, 110270160
    ]
    window_size = max(
        [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
    
    divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
#     print(window_size, divisors)
    # composite number의 인수 중 bs보다 작은 최대값 : 최대한 많은 인수가 잇는 값이 window여야 한다.
    batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
#     print(batch_sizes)
    window_size *= shard_multiplier
    
    batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
    max_batches_per_window = window_size // min(batch_sizes)
    shuffle_queue_size = max_batches_per_window * 3
#     print(max_batches_per_window,shuffle_queue_size)

    ret = { "boundaries": boundaries,
            "batch_sizes": batch_sizes,
            "min_length": min_length,
            "max_length": (max_length if drop_long_sequences else 10**9),
            "shuffle_queue_size": shuffle_queue_size
          }
    return ret



class BucketSamplerLessOverhead(Sampler):
    def __init__(self, tmlen2trajidx, batch_size=6000, max_length=400,
                 min_length_bucket=20, drop_last=True):
    
        scheme = batching_scheme(
                batch_size=batch_size,
                max_length=max_length,
                min_length_bucket=min_length_bucket,
                drop_long_sequences=drop_last,
                length_bucket_step=1.1)
        scheme['boundaries'] += [scheme['max_length']]
        
        # keys wo None data
        valid_keys= sorted([int(key_len) for key_len in list(tmlen2trajidx.keys()) if key_len != 'None'],
                           key = lambda x: x)
        # make buckets
        buckets = [[] for _ in range(len(scheme['boundaries']))]
        for key_len in valid_keys:
            for bucket_i, bound in enumerate(scheme['boundaries']):
                if key_len <= bound:
                    buckets[bucket_i].append(key_len)
                    break

        # make buckets2idx           
        self.buckets2idx = [[] for _ in range(len(scheme['boundaries']))]
        for i, bucket in enumerate(buckets):
            newbucket = []
            for key_len in bucket:
                newbucket += tmlen2trajidx[str(key_len)]

            self.buckets2idx[i] = newbucket
        
        # input 길이 boundary
        self.boundaries = scheme['boundaries']
        # 각 input 길이 boundary에서의 batch size
        self.batch_sizes = scheme['batch_sizes'] 
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        
        # shuffle the idx
        for b_idx in range(len(self.buckets2idx)):
            if self.buckets2idx[b_idx]:
                random.shuffle(self.buckets2idx[b_idx])
            
        self.buckets_len = [len(b) for b in self.buckets2idx]
        
        assert len(self.buckets2idx) == len(self.batch_sizes) == len(self.buckets_len)
        
    def __iter__(self):
        
        # get traj_idx & del the traj_idx
        while True:
            if sum(self.buckets_len) == 0:
                print("The dataloader has run out!!")
                break
            # choose which bucket
            b_idx = torch.multinomial(input=torch.tensor(self.buckets_len,
                                                         dtype=torch.float32),
                                      num_samples=1,
                                      replacement=False)

            batch_size = self.batch_sizes[b_idx]
            
            if len(self.buckets2idx[b_idx]) >= batch_size :
                traj_idx = self.buckets2idx[b_idx][:batch_size]
                del self.buckets2idx[b_idx][:batch_size]
                self.buckets_len[b_idx] -= batch_size

            else : # self.buckets2idx[b_idx] <= batch_size
                traj_idx = self.buckets2idx[b_idx][:]
                del self.buckets2idx[b_idx][:]
                self.buckets_len[b_idx] = 0
            
            yield traj_idx

    def __len__(self):
        raise NotImplementedError("BucketSampler cannot know the total number of batches.")

class BucketSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        scheme = batching_scheme(
                batch_size=batch_size,
                max_length=500,
                min_length_bucket=4,
                length_bucket_step=1.1)
        # input 길이 boundary
        self.boundaries = scheme['boundaries']
        
        # 각 input 길이 boundary에서의 batch size
        self.batch_sizes = scheme['batch_sizes'] 
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        

    def __iter__(self):
        buckets = [[] for i in range(len(self.boundaries))]
        for idx in self.sampler:
            
            # sampler에서 dataset을 가져와서 data의 length를 잼
#             print(self.sampler.data_source[idx].tm_len)
            data = self.sampler.data_source[idx]
            if data :
                if isinstance(data,tuple):
                    length = data[0].tm_len.item()
                else : 
                    length = data.tm_len.item()
            else :
#                 print('bucketsampler ', idx)
                continue
                
#             if data is None :
#                 print("i am still here")
            for i, boundary in enumerate(self.boundaries):
                if length <= boundary:
                    buckets[i].append(idx)
                    if len(buckets[i]) == self.batch_sizes[i]:
#                         print(buckets[i])
                        yield buckets[i]
                        buckets[i] = []
                    break
        if not self.drop_last:
            for bucket in filter(len, buckets):
                yield bucket

    def __len__(self):
        raise NotImplementedError("BucketSampler cannot know the total number of batches.")
        
def collate_fn(samples):
#     print(samples)
    # filtering none
    samples = [sample for sample in samples if sample is not None]
    if samples : # nonempty : tuple or torch_batch
        
        if isinstance(samples[0], list): # list : multiple transform
            num_transforms = len(samples[0])
            flatten_list = [_  for sample in samples for _ in sample] # bs * num_transforms
            data_trsfs_dict = OrderedDict()
            for trsf_i in range(num_transforms):
                trsf_data = flatten_list[trsf_i::num_transforms] # list or list of tuples(aug, perm)
                trsf_data = [data for data in trsf_data if data is not None] # None filtered
                if trsf_data:
                    if isinstance(trsf_data[0], tuple): # aug, perm
                        
                        sample_list = [_  for sample in trsf_data for _ in sample]
                        data_trsfs_dict[trsf_i] = tuple([Batch.from_data_list(sample_list[pair_i::len(trsf_data[0])]) for pair_i in range(len(trsf_data[0]))])
#                         left, right = sample_list[::2], sample_list[1::2]
#                         data_trsfs_dict[trsf_i]=(Batch.from_data_list(left), Batch.from_data_list(right))
                    else : # dest, mask
                        data_trsfs_dict[trsf_i]=Batch.from_data_list(trsf_data)
                else : # transformed data is all none and filtered out
                    data_trsfs_dict[trsf_i] = None
                    
            return list(data_trsfs_dict.values())
        
        elif isinstance(samples[0], tuple): # tuple
            sample_list = [_  for sample in samples for _ in sample]
            left, right = sample_list[::2], sample_list[1::2]
            return Batch.from_data_list(left), Batch.from_data_list(right)
        else : # torch_batch
            #samples = [sample for sample in samples if sample is not None]
            return Batch.from_data_list(samples)
            
    else : #empty
        return None
        
     