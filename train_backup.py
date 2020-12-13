import pathlib
import sys
import math
from collections import defaultdict, OrderedDict
import os
import timeit
import traceback

import argparse
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
from config import Config, AverageMeter
from model import TrajectoryEncoder, graphregion
from model import weights_init_classifier, DestinationProjHead, AugProjHead, MaskedProjHead, PermProjHead
from model import compute_destination_loss, compute_aug_loss, compute_mask_loss, compute_perm_loss
from transformation import Permuted, Masked, Augmented, Destination, Normal

import warnings


# # Autoreload Setting
# %load_ext autoreload
# %autoreload 2

data_dir = pathlib.PosixPath("data/")
dset_name = "porto"

train_fname = "data/porto/merged_train_edgeattr.h5"
val_fname = "data/porto/merged_val_edgeattr.h5"


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
def get_dataloader(fname, n_samples=None,n_processors=None, transform=None):
    """
    @param transform : Augmented(); Masked(); Permuted(); Destination()
    default)
        n_trains=1133657,n_vals=284997, 
        train_processors=36, val_processors=9,
    """
    
    dataloader = TrajDataset(file_path=fname, 
                             n_samples=n_samples, n_processors=n_processors,
                             transform=transform,
                             split='val')
    sampler = RandomSampler(dataloader)
    bucketing = BucketSampler(sampler, 12000)
    dataloader = iter(DataLoader(dataloader,
                        batch_sampler=bucketing,
                        collate_fn=collate_fn, num_workers=4, ))
    return dataloader

def get_dataloader_fast(fname, n_samples=None,n_processors=None, transform=None):
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
                        collate_fn=collate_fn, num_workers=4, ))
    return dataloader

def validation(val_dataloader_destination, val_dataloader_aug,
           val_dataloader_mask,val_dataloader_perm,
           traj_encoder, dest_proj, aug_proj, mask_proj,perm_proj,
           graphregion, config, criterion_ce, log_f,log_error, val_limits=None):
    
    traj_encoder.eval()
    dest_proj.eval()
    aug_proj.eval()
    mask_proj.eval()
    perm_proj.eval()
    
    avg_dest, avg_aug, avg_mask, avg_perm = 0.,0.,0.,0. 
    dest_cnt, aug_cnt, mask_cnt, perm_cnt = 0,0,0,0
    
    with torch.no_grad():
        val_runs = 0
        while True:
            val_runs += 1
            # reinit loss to zero every iter
            loss = 0.
            
            ####### Destination ###############################################################
            try : 
                val_batch_dest = next(val_dataloader_destination)
                val_batch_dest = val_batch_dest.to(config.device,config.dtype)
                loss_destination, out_tm, h_t, w_uh_t, negs, neg_term = compute_destination_loss(val_batch_dest, 
                                                            traj_encoder,
                                                            dest_proj,
                                                            graphregion,
                                                            config,)
        
                loss_destination = config.loss_dest_weight*loss_destination
                loss += loss_destination
            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
                loss_destination = None
                val_dataloader_destination = get_dataloader(val_fname,n_samples=config.n_vals,
                                                            n_processors=config.processors_vals,transform=Destination())
                pass

            ####################################################################################
            ####### Augmentation ###############################################################
            try:
                val_left_aug, val_right_aug = next(val_dataloader_aug) # left, right
                val_left_aug, val_right_aug = val_left_aug.to(config.device,config.dtype), val_right_aug.to(config.device,config.dtype)
                loss_augmentation, left, right, l_aug_input, r_aug_input, logits = compute_aug_loss(val_left_aug, val_right_aug, 
                                                                                                    traj_encoder, aug_proj,
                                                                                                    graphregion, config,
                                                                                                    criterion_ce)

                loss += loss_augmentation
            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
                loss_augmentation = None
                val_dataloader_aug = get_dataloader(val_fname,n_samples=config.n_vals,
                                                    n_processors=config.processors_vals,transform=Augmented())
                pass
            ####################################################################################
            ####### mask #######################################################################
            try:
                val_batch_mask = next(val_dataloader_mask)
                val_batch_mask = val_batch_mask.to(config.device,config.dtype)
                loss_mask, batch_queries, h_t, w_uh_t, negs, neg_term  = compute_mask_loss(val_batch_mask,
                                                                                           traj_encoder,
                                                                                           mask_proj, 
                                                                                           graphregion,
                                                                                           config,)

                loss_mask = config.loss_mask_weight*loss_mask
                loss += loss_mask
            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
                loss_mask = None
                val_dataloader_mask = get_dataloader(val_fname,n_samples=config.n_vals,
                                                     n_processors=config.processors_vals,transform=Masked())
                pass
            ####################################################################################
            ####### perm #######################################################################

            try:
                val_left_perm, val_right_perm = next(val_dataloader_perm) # left, right
                val_left_perm, val_right_perm = val_left_perm.to(config.device,config.dtype), val_right_perm.to(config.device,config.dtype)
                loss_perm, similarity_logits = compute_perm_loss(val_left_perm, val_right_perm,
                                                                 traj_encoder, perm_proj, 
                                                                 graphregion, config, criterion_ce)

                loss += loss_perm
                
            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
                loss_perm = None
                val_dataloader_perm = get_dataloader(val_fname,n_samples=config.n_vals,
                                                     n_processors=config.processors_vals,transform=Permuted())
                pass
            ####################################################################################

            if (loss_destination is not None):
#                 print("loss_destination: ", loss_destination.item())
                avg_dest += loss_destination.item()
                dest_cnt += 1
                
            if (loss_augmentation is not None):
#                 print("loss_augmentation: ", loss_augmentation.item())
                avg_aug += loss_augmentation.item()
                aug_cnt += 1
                
            if (loss_mask is not None):
#                 print("loss_mask: ", loss_mask.item())
                avg_mask += loss_mask.item()
                mask_cnt += 1
                
            if (loss_perm is not None):
#                 print("loss_perm: ", loss_perm.item())
                avg_perm += loss_perm.item()
                perm_cnt += 1
            
            # skipping the validation
            if val_limits:
                if val_runs > val_limits :
                    print("Reached the validation limits {}".format(val_limits))
                    break
                    
            #print()
        # averaging        
        avg_dest, avg_aug = avg_dest/dest_cnt, avg_aug/aug_cnt
        avg_mask, avg_perm = avg_mask/mask_cnt, avg_perm/perm_cnt
        
        return avg_dest, avg_aug, avg_mask, avg_perm
           
def train_one_epoch(dataloader_destination, dataloader_aug, dataloader_mask, dataloader_perm,
                    traj_encoder, dest_proj, aug_proj, mask_proj,perm_proj,
                    optimizer, scheduler,  criterion_ce, graphregion, config, log_f, log_error):
    
    traj_encoder.train()
    dest_proj.train()
    aug_proj.train()
    mask_proj.train()
    perm_proj.train()
    
    losses = AverageMeter()
    losses_dest = AverageMeter()
    losses_aug = AverageMeter()
    losses_mask = AverageMeter()
    losses_perm = AverageMeter()
    
    train_runs = 0
    sample_cnt = 0
    
    while True: 
        
        train_runs += 1
        sample_cnted = False
        # reinit loss to zero every iter
        loss = 0.
        

        ####### Destination ###############################################################
    #         loss_destination = None
        try : 
#             batch_dest = batch_dest_origin.clone()
            batch_dest = next(dataloader_destination)
            loss_destination, out_tm, h_t, w_uh_t, negs, neg_term = compute_destination_loss(batch_dest, 
                                                        traj_encoder,
                                                        dest_proj,
                                                        graphregion,
                                                        config,)
#             print("loss_destination", loss_destination)
            loss_destination = config.loss_dest_weight*loss_destination
            loss += loss_destination

        except Exception  as e:
            traceback.print_exc()
            log_error.write(traceback.format_exc())
#             print(e)
            loss_destination = None
            pass

        ####################################################################################
        ####### Augmentation ###############################################################
    #         loss_augmentation = None
        try:
#             left_aug, right_aug = left_aug_origin.clone(), right_aug_origin.clone() # left, right
            left_aug, right_aug = next(dataloader_aug) # left, right
            loss_augmentation, left, right, l_aug_input, r_aug_input, logits = compute_aug_loss(left_aug, right_aug, 
                                                                                                traj_encoder, aug_proj,
                                                                                                graphregion, config,
                                                                                                criterion_ce)

            loss += loss_augmentation
        except Exception as e:
            traceback.print_exc()
            log_error.write(traceback.format_exc())
#             print(e)
            loss_augmentation = None
            pass
        ####################################################################################
        ####### mask #######################################################################
    #         loss_mask = None
        try:
#             batch_mask = batch_mask_origin.clone()
            batch_mask = next(dataloader_mask)
            loss_mask, batch_queries, h_t, w_uh_t, negs, neg_term  = compute_mask_loss(batch_mask,
                                                                                       traj_encoder,
                                                                                       mask_proj, 
                                                                                       graphregion,
                                                                                       config,)
#             print('loss_mask', loss_mask)
            loss_mask = config.loss_mask_weight*loss_mask
            loss += loss_mask
        except Exception as e:
            traceback.print_exc()
            log_error.write(traceback.format_exc())
            loss_mask = None
            pass
        ####################################################################################
        ####### perm #######################################################################
    #         loss_perm = None
        try:
#             left_perm, right_perm = left_perm_origin.clone(), right_perm_origin.clone()
            left_perm, right_perm = next(dataloader_perm) # left, right
            loss_perm, similarity_logits = compute_perm_loss(left_perm, right_perm, traj_encoder, perm_proj, 
                                                             graphregion, config, criterion_ce)
            loss_perm = config.loss_perm_weight*loss_perm
            loss += loss_perm
        except Exception as e:
            traceback.print_exc()
            log_error.write(traceback.format_exc())
#             print(e)
            loss_perm = None
            pass
        ####################################################################################
        if (loss_destination is None) and (loss_augmentation is None) and (loss_mask is None) and (loss_perm is None):
            log_f.write("All dataloader ran out, finishing {}-th epoch's training. \n".format(config.epoch))
            print("All dataloader ran out, finishing {}-th epoch's training. \n".format(config.epoch))
            break
        
        losses.update(loss.item(), )
        if loss_destination is not None:
            losses_dest.update(loss_destination.item(), )
            sample_cnt += batch_dest.tm_len.size(0)
            sample_cnted = True
        if loss_augmentation is not None:
            losses_aug.update(loss_augmentation.item(), )
            if not sample_cnted:
                sample_cnt += left_aug.tm_len.size(0)
                sample_cnted = True
        if loss_mask is not None:
            losses_mask.update(loss_mask.item(), )
            if not sample_cnted:
                sample_cnt += batch_mask.tm_len.size(0)
                sample_cnted = True
        if loss_perm is not None:
            losses_perm.update(loss_perm.item(), )
            if not sample_cnted:
                sample_cnt += left_perm.tm_len.size(0)
                sample_cnted = True
            
            
        if train_runs % 100 == 0:
            log_f.write('Train Epoch:{} approx. [{}/{}] total_loss:{:.2f}({:.2f})\n'.format(config.epoch, 
                                                                            sample_cnt,
                                                                            config.n_trains,
                                                                            losses.val,
                                                                            losses.avg
                                                                    ))
            log_f.write('loss_destination:{:.2f}({:.2f}) \nloss_augmentation:{:.2f}({:.2f}) \nloss_mask:{:.2f}({:.2f}) \nloss_perm:{:.2f}({:.2f}) \n\n'.format( 
                losses_dest.val, losses_dest.avg, losses_aug.val, losses_aug.avg,
                losses_mask.val, losses_mask.avg, losses_perm.val, losses_perm.avg, ))
            print('Train Epoch:{} approx. [{}/{}] total_loss:{:.2f}({:.2f})'.format(config.epoch, 
                                                                            sample_cnt,
                                                                            config.n_trains,
                                                                            losses.val,
                                                                            losses.avg
                                                                    ))
            print('loss_destination:{:.2f}({:.2f}) \nloss_augmentation:{:.2f}({:.2f}) \nloss_mask:{:.2f}({:.2f}) \nloss_perm:{:.2f}({:.2f}) \n'.format( 
                losses_dest.val, losses_dest.avg, losses_aug.val, losses_aug.avg,
                losses_mask.val, losses_mask.avg, losses_perm.val, losses_perm.avg, ))          
            log_f.flush()
            log_error.flush()
            
        if train_runs % 4500 == 0: 
            log_f.write("At step 4500, save model {}.pt\n".format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs+1)))
            print("At step 4500, save model {}.pt\n".format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs+1)))
            ######
            models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                           mask_proj.__class__.__name__:mask_proj.state_dict(),
                           perm_proj.__class__.__name__:perm_proj.state_dict(),
                           aug_proj.__class__.__name__:aug_proj.state_dict(),
                           dest_proj.__class__.__name__:dest_proj.state_dict(),}
            
            torch.save(models_dict, 
                       os.path.join('models', config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs) +'.pt'))
            
        
            ######
            
            
        optimizer.zero_grad()

        loss.backward()

        # every iter
        optimizer.step()
        

def main():
    EPOCHS = 1000
    config = Config(graphregion.vocab_size)
    CUDA = config.CUDA
    
    use_gpu = torch.cuda.is_available()
    config.device = torch.device('cuda:{}'.format(CUDA)) if use_gpu else torch.device('cpu')
    config.dtype = torch.float32

    # init model
    # main encoder
    if "traj_encoder" in locals():
        del traj_encoder
    traj_encoder = TrajectoryEncoder(config)

    # proj layers
    dest_proj = DestinationProjHead(config)
    aug_proj = AugProjHead(config)
    mask_proj = MaskedProjHead(config)
    perm_proj = PermProjHead(config)

    traj_encoder.apply(weights_init_classifier).to(config.device, config.dtype)
    dest_proj.apply(weights_init_classifier).to(config.device, config.dtype)
    aug_proj.apply(weights_init_classifier).to(config.device, config.dtype)
    mask_proj.apply(weights_init_classifier).to(config.device, config.dtype)
    perm_proj.apply(weights_init_classifier).to(config.device, config.dtype)
    
    # create log_file
    log_f = open('log/{}_train.txt'.format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers)), 'w')
    log_error = open('log/{}_train_error.txt'.format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers)), 'w')
    print("Config {} \n".format(str(config.__dict__)))
    log_f.write("Config {} \n".format(str(config.__dict__)))
    log_f.flush()
    # loss_func
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.AdamW(list(traj_encoder.parameters()) +\
                                  list(dest_proj.parameters()) +\
                                  list(aug_proj.parameters()) +\
                                  list(mask_proj.parameters()) +\
                                  list(perm_proj.parameters()), 
                                  lr=.001 )
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[1, 2, 3,], 
                                                     gamma=0.1)

    val_best_dest, val_best_aug, val_best_mask, val_best_perm = float('inf'), float('inf'), float('inf'), float('inf')
    
    for epoch in range(EPOCHS):

        # init dataloader
        log_f.write("{} epoch: initializing dataloaders \n".format(epoch+1))
        print("{} epoch: initializing dataloaders \n".format(epoch+1))
        dataloader_destination = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Destination())
        dataloader_aug = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Augmented())
        dataloader_mask = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Masked())
        dataloader_perm = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Permuted())

        val_dataloader_destination = get_dataloader(val_fname,n_samples=284997,n_processors=9,transform=Destination())
        val_dataloader_aug = get_dataloader(val_fname,n_samples=284997,n_processors=9,transform=Augmented())
        val_dataloader_mask = get_dataloader(val_fname,n_samples=284997,n_processors=9,transform=Masked())
        val_dataloader_perm = get_dataloader(val_fname,n_samples=284997,n_processors=9,transform=Permuted())

        ## train_one_epoch model ################################################################
        log_f.write("{} epoch: start training \n\n".format(epoch+1))
        print("{} epoch: start training \n".format(epoch+1))
        config.epoch = epoch+1
        train_one_epoch(dataloader_destination, dataloader_aug, dataloader_mask, dataloader_perm,
                        traj_encoder, dest_proj, aug_proj, mask_proj,perm_proj,
                        optimizer, scheduler,  criterion_ce, graphregion, config, log_f, log_error)

        ## validate model #######################################################################
        # once an epoch finishes, validate model's performance
        log_f.write("{} epoch: start validating \n\n".format(epoch+1))
        print("{} epoch: start validating \n".format(epoch+1))
        models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                           mask_proj.__class__.__name__:mask_proj.state_dict(),
                           perm_proj.__class__.__name__:perm_proj.state_dict(),
                           aug_proj.__class__.__name__:aug_proj.state_dict(),
                           dest_proj.__class__.__name__:dest_proj.state_dict(),}
        # val_* : lower is better
        
        # do validation
        val_avg_dest, val_avg_aug, val_avg_mask, val_avg_perm = validation(val_dataloader_destination,
                                                                           val_dataloader_aug,
                                                                           val_dataloader_mask,
                                                                           val_dataloader_perm,
                                                                           traj_encoder, dest_proj,
                                                                           aug_proj, mask_proj,
                                                                           perm_proj, graphregion,
                                                                           config, criterion_ce, 
                                                                           log_f,log_error,
                                                                           val_limits=config.val_limits,
                                                                           )
        
        
        saved_modelname = ''
        if val_avg_dest < val_best_dest: # save
            saved_modelname += 'dest_'
            val_best_dest = val_avg_dest
        if val_avg_aug < val_best_aug: # save
            saved_modelname += 'aug_'
            val_best_aug = val_avg_aug
        if val_avg_mask < val_best_mask: # save
            saved_modelname += 'mask_'
            val_best_mask = val_avg_mask
        if val_avg_perm < val_best_perm: # save
            saved_modelname += 'perm_'
            val_best_perm = val_avg_perm
        
        if saved_modelname: # the model improved on at least one of four tasks.
            log_f.write("Save model {}.pt\n".format(saved_modelname))
            print("Save model {}.pt".format(saved_modelname))
            torch.save(models_dict, 
                       os.path.join('models', config.name+'_'+saved_modelname+'_num_hid_layer_'+str(config.num_hidden_layers)+'.pt'))

        # save model every config.save_epoch
        if (epoch+1) % config.save_epoch == 0:
            log_f.write("Save model {}.pt\n".format(config.name + '_e{}'.format(epoch+1)))
            print("Save model {}.pt".format(config.name + '_e{}'.format(epoch+1)))
            torch.save(models_dict, 
                       os.path.join('models', config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_e{}'.format(epoch+1) +'.pt'))
        log_f.flush()
        # every epoch
        scheduler.step()
    log_f.close()
    


######################################################################
# Options
######################################################################
#parser = argparse.ArgumentParser(description='Inspect dataloader')


#parser.add_argument('--dataloader_destination', action='store_true',default=False,  help = 'dataloader_destination')
#parser.add_argument('--dataloader_aug', action='store_true',default=False,  help = '')
#parser.add_argument('--dataloader_mask', action='store_true',default=False,  help = '')
#parser.add_argument('--dataloader_perm', action='store_true',default=False,  help = '')


#global opts
#opts = parser.parse_args()

#####################################################################
#dataloader_destination = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Destination())
#dataloader_aug = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Augmented())
#dataloader_mask = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Masked())
#dataloader_perm = get_dataloader(train_fname,n_samples=1133657,n_processors=36,transform=Permuted())

    
if __name__ == '__main__':
    tmlen2trajidx = pickle.load(open('data/porto/tmlen2trajidx.pkl','rb'))
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()


