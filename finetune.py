import pathlib
import sys
import math
from collections import defaultdict, OrderedDict
import os
import timeit
import traceback
import pickle

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
##################################################################
from finetune_config import Config, AverageMeter
# from model import TrajectoryEncoder, graphregion
# from model import weights_init_classifier, DestinationProjHead, AugProjHead, MapembProjHead, MaskedProjHead, PermProjHead
# from model import compute_destination_loss, compute_aug_loss, compute_mask_loss, compute_perm_loss

from model_rnnbased import TrajectoryEncoder, graphregion
from model_rnnbased import weights_init_classifier, DestinationProjHead, AugProjHead, MapembProjHead, MaskedProjHead, PermProjHead
from model_rnnbased import compute_destination_loss, compute_aug_loss, compute_mask_loss, compute_perm_loss


from transformation import Reversed, Masked, Augmented, Destination, Normal
# Permuted
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

def get_dataloader_fast(fname, tmlen2trajidx, n_samples=None,n_processors=None, batch_size =12000, num_workers=0,transform=None):
    """
    @param fname : train_fname or val_fname
    @param tmlen2trajidx : tmlen2trajidx or val_tmlen2trajidx
    @param transform : Augmented(); Masked(); Permuted(); Destination()
    default)
        n_trains=1133657,n_vals=284997, 
        train_processors=36, val_processors=9,
    """
    
    dataloader = TrajDataset(file_path=fname, 
                             n_samples=n_samples, n_processors=n_processors,
                             transform=transform,
                             split='val') # split doesnt matter
    batch_sampler = BucketSamplerLessOverhead(tmlen2trajidx, 
                                              batch_size=batch_size, 
                                              max_length=400,
                                              min_length_bucket=20,
                                              drop_last=True)
    dataloader = iter(DataLoader(dataloader,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn, num_workers=num_workers, ))
    return dataloader

def validation(val_dest_aug_mask_perm_dataloader,
           traj_encoder, dest_proj, position_proj,
           graphregion, config, criterion_ce, log_f,log_error, val_limits=None):
    
    
    traj_encoder.eval()
    
    if dest_proj is not None: 
        dest_proj.eval()
    if position_proj is not None: 
        position_proj.eval()
    
    avg_dest, avg_position = 0.,0.
    dest_cnt, position_cnt = 1,1,
    total_loss, total_loss_cnt = 0., 1
    
    with torch.no_grad():
        val_runs = 0
        while True:
            val_runs += 1
            # reinit loss to zero every iter
            loss = 0.
            try : 
                val_batch = next(val_dest_aug_mask_perm_dataloader)
            except StopIteration as e:
                val_dest_aug_mask_perm_dataloader = get_dataloader_fast(val_fname,val_tmlen2trajidx,
                                                                n_samples=284997,n_processors=9,
                                                                num_workers=4,
                                                                batch_size=config.batch_size,
                                                                transform=(Destination(), Augmented(),Masked(),Reversed())) # Reversed(), Permuted()
                val_batch   = next(val_dest_aug_mask_perm_dataloader)
            
            if val_batch is None :
                val_runs -= 1
                continue
            

            if "dest" in config.del_tasks:
                val_batch_position = val_batch
        
            elif "position" in config.del_tasks:
                val_batch_dest = val_batch
                
            ####### Destination ###############################################################
            if 'dest' in config.del_tasks:
                loss_destination = None
            else :
                try : 
                    loss_destination, out_tm, h_t, w_uh_t, negs, neg_term, acc = compute_destination_loss(val_batch_dest, 
                                                                traj_encoder,
                                                                dest_proj,
                                                                graphregion,
                                                                config, is_val=True)

                    loss += loss_destination
                except Exception as e:
                    traceback.print_exc()
                    log_error.write(traceback.format_exc())
                    loss_destination = None
                    if val_batch_dest is not None:
                        if val_batch_dest.traj_len.size(0) == 1: # batchsize = 1, skip the iteration
                            val_runs -= 1
                            continue
                    pass

            ####################################################################################
            ####### Position ###############################################################
            if 'position' in config.del_tasks:
                loss_position = None
            else : # train position task
                try:
                    loss_position = compute_position_loss(val_batch_position, traj_encoder, position_proj, 
                                                                     graphregion, config, criterion_ce)

                    loss += loss_position
                    #print("loss_perm", loss_perm)
                except Exception as e:
                    traceback.print_exc()
                    log_error.write(traceback.format_exc())
        #             print(e)
                    loss_position = None
                    if val_batch_position is not None:
                        if batch_position.traj_len.size(0) == 1: # batchsize = 1, skip the iteration
                            val_runs -= 1
                            continue

                    pass
            
            ####################################################################################
            if loss > 0:
                total_loss += loss
                total_loss_cnt  += 1
            if (loss_destination is not None):
#                 print("loss_destination: ", loss_destination.item())
                avg_dest += loss_destination.item()
                dest_cnt += 1
            if (loss_position is not None):
#                 print("loss_destination: ", loss_destination.item())
                avg_position += loss_position.item()
                position_cnt += 1
            
            
            # skipping the validation
            if val_limits:
                if val_runs > val_limits :
                    print("Reached the validation limits {}".format(val_limits))
                    break
            #print()
        # averaging  
        avg_loss = total_loss/total_loss_cnt
        avg_dest, avg_position = avg_dest/dest_cnt, avg_position/position_cnt
        
        return avg_loss, avg_dest, avg_position
           
def train_one_epoch(dest_aug_mask_perm_dataloader,
                    traj_encoder, dest_proj, position_proj,
                    optimizer, scheduler,  criterion_ce, graphregion, config, log_f, log_error):
    
    traj_encoder.train()
    if dest_proj is not None: 
        dest_proj.train()
    if position_proj is not None: 
        position_proj.train()
    
    losses = AverageMeter()
    losses_dest = AverageMeter()
    losses_position = AverageMeter()

    train_runs = 0
    sample_cnt = 0
    
    losses_hist = []
    losses_dest_hist = []
    losses_position_hist = []
    
    while True: 
        
        train_runs += 1
        
        # re-init loss to zero every iter
        loss = 0.
        try:
            train_batch = next(dest_aug_mask_perm_dataloader)
        except StopIteration as e:
            log_f.write("All dataloader ran out, finishing {}-th epoch's training. \n".format(config.epoch))
            print("All dataloader ran out, finishing {}-th epoch's training. \n".format(config.epoch))
            break
            
            
            
        if train_batch is None: # all filtered out: length<10 or [-1]
            train_runs -= 1
            continue
            
        if "dest" in config.del_tasks:
            batch_position = train_batch
        
        elif "position" in config.del_tasks:
            batch_dest = train_batch
            
        ####### Destination ###############################################################
        if 'dest' in config.del_tasks:
            loss_destination = None
        else : 
            try : 
                loss_destination, out_tm, h_t, w_uh_t, _neg_term, neg_term, acc, cells_y, answer_y = compute_destination_loss(batch_dest, 
                                                            traj_encoder,
                                                            dest_proj,
                                                            graphregion,
                                                            config, is_val=True)
                #print("loss_destination", loss_destination)
                loss_destination = loss_destination
                loss += loss_destination

            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
    #             print(e)
                loss_destination = None
                if batch_dest is not None:
                    if batch_dest.traj_len.size(0) == 1: # batchsize = 1, skip the iteration
                        train_runs -= 1
                        continue
                pass

        ####################################################################################
        ####### position #######################################################################
        if 'position' in config.del_tasks:
            loss_position = None
        else : # train position task
            try:
                loss_position = compute_position_loss(batch_position, traj_encoder, position_proj, 
                                                                 graphregion, config, criterion_ce)
                
                loss += loss_position
                #print("loss_perm", loss_perm)
            except Exception as e:
                traceback.print_exc()
                log_error.write(traceback.format_exc())
    #             print(e)
                loss_position = None
                if batch_position is not None:
                    if batch_position.traj_len.size(0) == 1: # batchsize = 1, skip the iteration
                        train_runs -= 1
                        continue
        
                pass
        ####################################################################################
        if dest_proj is not None: # on destination finetune
            if (loss_destination is None) :
                log_f.write("loss_destination none, at {}-th epoch's training: check errordata_e{}_step{}.pkl \n".format(config.epoch, config.epoch, train_runs))
                print("loss_destination none, at {}-th epoch's training: check errordata_e{}_step{}.pkl \n".format(config.epoch, config.epoch, train_runs)) 
                pickle.dump((batch_dest), 
                            open('errordata_e{}_step{}.pkl'.format(config.epoch, train_runs),'wb'))
                train_runs -= 1
                continue
        elif position_proj is not None: # on position finetune
            if (loss_position is None) :
                log_f.write("loss_position none, at {}-th epoch's training: check errordata_e{}_step{}.pkl \n".format(config.epoch, config.epoch, train_runs))
                print("loss_position none, at {}-th epoch's training: check errordata_e{}_step{}.pkl \n".format(config.epoch, config.epoch, train_runs)) 
                pickle.dump((batch_position), 
                            open('errordata_e{}_step{}.pkl'.format(config.epoch, train_runs),'wb'))
                train_runs -= 1
                continue
        
        
        sample_cnted = False
        try:
            losses.update(loss.item(),)
        except: 
            print("Error occured, losses: ",loss, loss_destination, loss_position)
        if loss_destination is not None: # update destination loss
            losses_dest.update(loss_destination.item(), )
            sample_cnt += batch_dest.tm_len.size(0)
            sample_cnted = True
        if loss_position is not None: # update position loss
            losses_position.update(loss_position.item(), )
            sample_cnt += batch_position.tm_len.size(0)
            sample_cnted = True
        
            
        if train_runs % 10 == 0:
            print("dest acc: ", acc)
            print("dest h_t: ", h_t, torch.norm(h_t))
            print("dest cells_y: ", [(cell_id% graphregion.numx, cell_id// graphregion.numx) for cell_id in cells_y ] )
            print("dest answer_y: ", [(cell_id% graphregion.numx, cell_id// graphregion.numx) for cell_id in answer_y ])

#             print("dest _neg_term: ", _neg_term)
            

#             print("logits_perm, target_perm: ", logits_perm, target_perm)
#             print('batch_queries', batch_queries)
#             print('h_t', h_t)
#             print('w_uh_t',w_uh_t)
#             print('_neg_term', _neg_term)
#             print('neg_term', neg_term)
            
            
            losses_hist.append(losses.val)
            losses_dest_hist.append(losses_dest.val)
            losses_position_hist.append(losses_position.val)
            
            log_f.write('Train Epoch:{} approx. [{}/{}] total_loss:{:.2f}({:.2f})\n'.format(config.epoch, 
                                                                            sample_cnt,
                                                                            config.n_trains,
                                                                            losses.val,
                                                                            losses.avg
                                                                    ))
            log_f.write('loss_destination:{:.2f}({:.2f}) \nloss_position:{:.2f}({:.2f}) \n\n'.format( 
                losses_dest.val, losses_dest.avg, losses_position.val, losses_position.avg,) )
            
            print('Train Epoch:{} approx. [{}/{}] total_loss:{:.2f}({:.2f})'.format(config.epoch, 
                                                                            sample_cnt,
                                                                            config.n_trains,
                                                                            losses.val,
                                                                            losses.avg
                                                                    ))
            print('loss_destination:{:.2f}({:.2f}) \nloss_position:{:.2f}({:.2f})  \n'.format( 
                losses_dest.val, losses_dest.avg, losses_position.val, losses_position.avg,
            ))          
            log_f.flush()
            log_error.flush()
            
        if train_runs % 4500 == 0: 
            log_f.write("At step 4500, save model {}.pt\n".format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs+1)))
            print("At step 4500, save model {}.pt\n".format(config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs+1)))
            ######
            if 'position' in config.del_tasks:
                models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                               dest_proj.__class__.__name__:dest_proj.state_dict(),}
                
            elif 'dest' in config.del_tasks:
                models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                               position_proj.__class__.__name__:position_proj.state_dict(),}
            
            torch.save(models_dict, 
                       os.path.join('models', config.name+'_num_hid_layer_'+str(config.num_hidden_layers) + '_step{}'.format(train_runs) +'.pt'))
            
        
            ######
            
            
        optimizer.zero_grad()

        loss.backward()

        # every iter
        optimizer.step()
        
    torch.save((losses_hist, losses_dest_hist, losses_position_hist,),
               os.path.join('train_hist',
                            config.name+'_loss_hist'+\
                            '_hidlayer_'+str(config.num_hidden_layers)+\
                            'e'+str(config.epoch)+'.pt'))
    

def main():
    EPOCHS = 1000
    config = Config(graphregion.vocab_size)
    CUDA = config.CUDA
    
    use_gpu = torch.cuda.is_available()
    config.device = torch.device('cuda:{}'.format(CUDA)) if use_gpu else torch.device('cpu')
    config.dtype = torch.float32

    # init model
    # main encoder
    traj_encoder = TrajectoryEncoder(config)
    traj_encoder.to(config.device, config.dtype)

    # proj layers
    if 'dest' in config.del_tasks:
        # position_proj = PositionProjHead(config)
        # position_proj.to(config.device, config.dtype)
        # position_proj.apply(weights_init_classifier)
        dest_proj = None
    elif 'position' in config.del_tasks:
        dest_proj = DestinationProjHead(config)
        dest_proj.to(config.device, config.dtype)
        dest_proj.apply(weights_init_classifier)
        position_proj = None
    
     

    
    if config.resume : 
        
        savedmodels_dict = torch.load(os.path.join("models", config.path_state),
                                     map_location=torch.device('cuda:{}'.format(CUDA)))
        
        traj_encoder.load_state_dict(savedmodels_dict['TrajectoryEncoder'],)
        
        # freeze the params
        for param in traj_encoder.parameters():
            param.requires_grad = False
            
        s_epoch = config.s_epoch
        
    else : # init the whole params
        dest_proj.apply(weights_init_classifier)
#         raise ValueError("Invalid Downstream resume setting: you must resume on a pretrained model -- by Doyoung")
    
    
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
    if "dest" in config.del_tasks :
        optimizer = torch.optim.AdamW(position_proj.parameters(),
                                  lr=.001 )
    elif "position" in config.del_tasks :
        optimizer = torch.optim.AdamW(dest_proj.parameters(),
                                  lr=.001 )
    else :
        raise ValueError("Invalid Downstream optimizer setting: you must resume on a pretrained model -- by Doyoung")
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[1, 2, 3,], 
                                                     gamma=0.1)
    if s_epoch > 0:
        for _ in range(s_epoch): 
            scheduler.step()
        print("At start epoch {}, Optimizer LR : {}".format(s_epoch, optimizer.param_groups[0]['lr']))
        
    val_best_dest, val_best_position = float('inf'), float('inf')
    
    
    for epoch in range(s_epoch, EPOCHS):

        # init dataloader
        log_f.write("{} epoch: initializing dataloaders \n".format(epoch+1))
        print("{} epoch: initializing dataloaders \n".format(epoch+1))
        
        if "dest" in config.del_tasks :
            dest_aug_mask_perm_dataloader = get_dataloader_fast(train_fname,tmlen2trajidx,
                                                                n_samples=1133657,n_processors=36,
                                                                num_workers=4,
                                                            batch_size=config.batch_size,
                                                                transform=Normal()
                                                               ) 
            val_dest_aug_mask_perm_dataloader = get_dataloader_fast(val_fname,val_tmlen2trajidx,
                                                                n_samples=284997,n_processors=9,
                                                                num_workers=4,
                                                                batch_size=config.batch_size,
                                                                transform=Normal() 
                                                                   )
        elif "position" in config.del_tasks :
            dest_aug_mask_perm_dataloader = get_dataloader_fast(train_fname,tmlen2trajidx,
                                                                n_samples=1133657,n_processors=36,
                                                                num_workers=4,
                                                            batch_size=config.batch_size,
                                                                transform=Destination()
                                                               ) 
            val_dest_aug_mask_perm_dataloader = get_dataloader_fast(val_fname,val_tmlen2trajidx,
                                                                n_samples=284997,n_processors=9,
                                                                num_workers=4,
                                                                batch_size=config.batch_size,
                                                                transform=Destination() 
                                                                   )
                                                                          
        
        
        ## train_one_epoch model ################################################################
        log_f.write("{} epoch: start training \n\n".format(epoch+1))
        print("{} epoch: start training \n".format(epoch+1))
        config.epoch = epoch+1
        train_one_epoch(dest_aug_mask_perm_dataloader,
                         traj_encoder, dest_proj, position_proj,
                         optimizer, scheduler,  criterion_ce, graphregion, config, log_f, log_error,
                       )

        ## validate model #######################################################################
        # once an epoch finishes, validate model's performance
        log_f.write("{} epoch: start validating \n\n".format(epoch+1))
        print("{} epoch: start validating \n".format(epoch+1))

        if 'position' in config.del_tasks:
            models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                               dest_proj.__class__.__name__:dest_proj.state_dict(),}
                
        elif 'dest' in config.del_tasks:
            models_dict = {traj_encoder.__class__.__name__:traj_encoder.state_dict(),
                               position_proj.__class__.__name__:position_proj.state_dict(),}
            
        # val_* : lower is better
        # do validation
        avg_loss, val_avg_dest, val_avg_position = validation(val_dest_aug_mask_perm_dataloader,                          traj_encoder, dest_proj, position_proj,
                                                                                                     graphregion,
                                                                           config, criterion_ce, 
                                                                           log_f,log_error,
                                                                           val_limits=config.val_limits,
                                                                           )
        # print log
        log_f.write('Validation Epoch:{} total_loss:{:.2f}\n'.format(config.epoch, avg_loss,))
        log_f.write('best_destination:{:.2f} \nbest_position:{:.2f} \n\n'.format( val_best_dest, val_best_position, ))
        log_f.write('val_destination:{:.2f} \nval_position:{:.2f} \n\n'.format( val_avg_dest, val_avg_position,))
        
        print('Validation Epoch:{} total_loss:{:.2f}\n'.format(config.epoch, avg_loss,))
        print('best_destination:{:.2f} \nbest_position:{:.2f} \n\n'.format( val_best_dest, val_best_position,))
        print('val_destination:{:.2f} \nval_position:{:.2f} \n\n'.format( val_avg_dest, val_avg_position))
        
        
        
        saved_modelname = ''
        if val_avg_dest < val_best_dest: # save
            saved_modelname += 'dest_'
            val_best_dest = val_avg_dest
            
        if val_avg_position < val_best_position: # save
            saved_modelname += 'position_'
            val_best_position = val_avg_position
        
        
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

# making these global 
tmlen2trajidx = pickle.load(open('data/porto/tmlen2trajidx.pkl','rb'))
val_tmlen2trajidx = pickle.load(open('data/porto/val_tmlen2trajidx.pkl','rb'))

    
if __name__ == '__main__':
#     tmlen2trajidx = pickle.load(open('data/porto/tmlen2trajidx.pkl','rb'))
#     val_tmlen2trajidx = pickle.load(open('data/porto/val_tmlen2trajidx.pkl','rb'))

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()


