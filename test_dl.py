import pickle
import random
import pathlib
import time
# from dataloader_nowaiting import TrajDataset, BucketSamplerLessOverhead, collate_fn
from dataloader import TrajDataset, BucketSamplerLessOverhead, BucketSampler, collate_fn
from transformation import Permuted, Masked, Augmented, Destination, Normal
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, RandomSampler, BatchSampler

data_dir = pathlib.PosixPath("data/")
dset_name = "porto"

train_fname = "data/porto/merged_train_edgeattr.h5"
val_fname = "data/porto/merged_val_edgeattr.h5"


tmlen2trajidx = pickle.load(open('data/porto/tmlen2trajidx.pkl','rb'))
val_tmlen2trajidx = pickle.load(open('data/porto/val_tmlen2trajidx.pkl','rb'))

def get_dataloader_fast(fname, tmlen2trajidx, n_samples=None,n_processors=None,num_workers=0, transform=None):
    """
    @param fname : train_fname or val_fname
    @param transform : Augmented(); Masked(); Permuted(); Destination()
    default)
        n_trains=1133657,n_vals=284997, 
        train_processors=36, val_processors=9,
    """
    
    dataloader = TrajDataset(file_path=fname, 
                             n_samples=n_samples, n_processors=n_processors,
                             transform=transform,
                             split='val') # split doesnt matter
    batch_sampler = BucketSamplerLessOverhead(tmlen2trajidx, # global var
                                              batch_size=6000, 
                                              max_length=400,
                                              min_length_bucket=20,
                                              drop_last=True)
    dataloader = iter(DataLoader(dataloader,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn, num_workers=num_workers, ))
    return dataloader


# def get_dataloader(fname, n_samples=None,n_processors=None, transform=None):
#     """
#     @param transform : Augmented(); Masked(); Permuted(); Destination()
#     default)
#         n_trains=1133657,n_vals=284997, 
#         train_processors=36, val_processors=9,
#     """
    
#     dataloader = TrajDataset(file_path=fname,
#                              n_samples=n_samples, n_processors=n_processors,
#                              transform=transform,
#                              split='val')
#     sampler = RandomSampler(dataloader)
#     bucketing = BucketSampler(sampler, 6000)
#     dataloader = iter(DataLoader(dataloader,
#                         batch_sampler=bucketing,
#                         collate_fn=collate_fn))
#     return dataloader


# multiDL_nw0 = get_dataloader_fast(train_fname,
#                     n_samples=1133657,n_processors=36,num_workers=0,
#                     transform=(Destination(), Augmented(),Masked(),Permuted()))
# multiDL_nw4 = get_dataloader_fast(train_fname,
#                     n_samples=1133657,n_processors=36,num_workers=4,
#                     transform=(Destination(), Augmented(),Masked(),Permuted()))

valmultiDL_nw4 = get_dataloader_fast(val_fname,val_tmlen2trajidx,
                    n_samples=284997,n_processors=9,num_workers=4,
                    transform=(Destination(), Augmented(),Masked(),Permuted()))

# start_time = time.time()
# for i,batch in enumerate(multiDL_nw0):
# #     print(batch)
#     if i == 50:
#         break
# print("multiDL_nw0 --- {} seconds ---".format(time.time() - start_time))


start_time = time.time()
for i,batch in enumerate(valmultiDL_nw4):
#     print(batch)
    if i == 50:
        break
print("multiDL_nw4 --- {} seconds ---".format(time.time() - start_time))

# dataloader_Normal = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Normal())

# start_time = time.time()
# for i,batch in enumerate(dataloader_Normal):
# #     print(batch)
#     if i == 50:
#         break
# print("DL -overhead --- {} seconds ---".format(time.time() - start_time))
