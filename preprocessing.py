import h5py

import pandas as pd
import pathlib
import folium
import ast
from math import sin, ceil, floor
import numpy as np
from collections import defaultdict, Counter
from sklearn.neighbors import KDTree
import pickle

import data_utils as utils

data_dir = pathlib.PosixPath("data/")

class SpatialRegion(object) : 
    
    def __init__(self, dataset_name, minlon, minlat, maxlon, maxlat, 
                 xstep, ystep, 
#                  numx, numy, 
                 minfreq=50, maxvocab_size=50000, knn_k=5, 
                 vocab_start=4,
#                  cellcount, hotcell, hotcell_kdtree,
#                  hotcell2vocab, vocab2hotcell, vocab_size, is_built
                ):
        self.dataset_name = dataset_name
        self.minfreq = minfreq
        self.maxvocab_size = maxvocab_size
        self.knn_k = knn_k
        self.vocab_start = vocab_start
        
        # compute minx, miny, maxx, maxy from the followings.
        self.minlon = minlon
        self.minlat = minlat
        self.maxlon = maxlon
        self.maxlat = maxlat
        
        self.minx, self.miny, self.maxx, self.maxy = None,None,None,None
        
        self.xstep, self.ystep = xstep, ystep
        self.numx, self.numy = None, None
        
        self.cellcount, self.hotcell, self.hotcell2vocab = None,None,None
        self.vocab2hotcell, self.vocab_size = None,None
        self.is_built = False
        self.hotcell_kdtree = None
        self.build_region()
                
    def build_region(self):
        self.minx, self.miny = utils.lonlat2meters(self.minlon,self.minlat)
        self.maxx, self.maxy = utils.lonlat2meters(self.maxlon,self.maxlat)
        self.numx = ceil(round(self.maxx-self.minx, ndigits=6) / self.xstep)
        self.numy = ceil(round(self.maxy-self.miny, ndigits=6) / self.ystep)
        
    def coord2cell(self, x,y):
        """
        mapping x,y to cell_id
        @param x,y : coordinate in meter metric
        """
        xoffset = floor(round(x - self.minx, ndigits=6) / self.xstep)
        yoffset = floor(round(y - self.miny, ndigits=6) / self.ystep)

        return yoffset * self.numx + xoffset

    def cell2coord(self, cell_id):
        yoffset = cell_id // self.numx
        xoffset = cell_id % self.numx
        y = self.miny + (yoffset + 0.5) * self.ystep #(cell_size)
        x = self.minx + (xoffset + 0.5) * self.xstep
        return x, y
    
    def gps2cell(self,lon,lat):
        """
        mapping lon, lat to cell_id through coord2cell()
        """
        x,y = utils.lonlat2meters(lon,lat)
        return self.coord2cell(x,y) # cell_id
    
    def cell2gps(self, cell_id):
        x, y = self.cell2coord(cell_id)
        return utils.meters2lonlat(x,y) # lon, lat
    
    def gps2offset(self, lon, lat):
        """
        mapping lon, lat to coord in the region(self instance)
        """
        x, y = utils.lonlat2meters(lon, lat)
        xoffset = round(x - self.minx, ndigits=6) / self.xstep
        yoffset = round(y - self.miny, ndigits=6) / self.ystep
        return xoffset, yoffset
    
    def is_inregion(self, lon, lat):
        """
        @param lon, lat
        """
        if (self.minlon <= lon < self.maxlon) and \
        (self.minlat <= lat < self.maxlat):
            return True
        else : 
            return False
    
    def make_vocab(self, trips_path, trips_len=None, zerolen_tripids=None):
        """
        ##### NOT IN USE #### for csv format
        # @param trips_path : pd.DataFrame that trips are concatenated
        # @param trips_len : dict id:(start,len)
        # @param zerolen_tripids : list id
        
        #####################
        
        for hd5 format
        @param trips_path ::string ".hd5"; each trip is found in trips["trips/{}".format(num)]
        @param trips_len : None
        """
        self.cellcount = defaultdict(int)
        
        num_out_region = 0
        with h5py.File(trips_path, 'r') as f:
            for num in range(len(f["trips"].keys())):
                if num+1 in set(zerolen_tripids): continue
                trip = f["trips/"+str(num+1)][()] # nd.array (2,traj_len)

                for p in range(trip.shape[1]):
                    lon, lat = trip[:,p]
                    if self.is_inregion(lon, lat):
                        cell_id = self.gps2cell(lon,lat) # cell_id
                        self.cellcount[cell_id] += 1
                    else : # not in region
                        num_out_region += 1

                if num % 300 == 299 : 
                    print("Processed {} trips".format(num+1))
                    
        max_num_hotcells = min(self.maxvocab_size, len(self.cellcount))
        topcellcount = sorted(self.cellcount.items(), 
                              key=lambda x : -x[1],)[:max_num_hotcells] # descending
        print("max_num_hotcells: {} \nmax_count of hotcells: {}".format(max_num_hotcells, topcellcount[0][-1]))
        
        # the biggest idx with minfreq
        minfreq_idx = np.argwhere(np.array(topcellcount)[:,1] == self.minfreq)[-1,0]
        self.hotcell = np.array([cell_id for cell_id, _ in np.array(topcellcount[:minfreq_idx+1])]) # (len_cell_ids)
        print("num of hotcell : {}".format(len(self.hotcell)))
        
        ## build the map between cell and vocab id
        self.hotcell2vocab = dict([(cell_id, i+self.vocab_start) for (i, cell_id) in enumerate(self.hotcell)])
        
        #region.vocab2hotcell = map(reverse, region.hotcell2vocab)
        self.vocab2hotcell = {vocab:cell for (cell,vocab) in self.hotcell2vocab.items()}
        
        ## vocabulary size
        self.vocab_size = self.vocab_start + len(self.hotcell)
        
        self.built = True
        ## build the hot cell kdtree to facilitate search
        
        # coord : (len_hotcells, 2: (x,y))
        coord = np.array(list(map(self.cell2coord, self.hotcell))) # self.hotcell : (len_hotcells)
        self.hotcell_kdtree = KDTree(coord)

    def knearest_hotcells(self, cell_ids, k):
        """
        @param cell_ids :: iterables
        return knearest_hotcells_id : (len_cells_ids, k), knndists : (len_cells_ids, k)
        """
        assert self.built == True
        ## cell_ids should be iterable
        coord = np.array(list(map(self.cell2coord, cell_ids))) # (len_hotcells, 2: (x,y))
        dists, indice = self.hotcell_kdtree.query(coord,k=k) # indice here is indice of self.hotcell
        return self.hotcell[indice], dists 
    
    def nearest_hotcell(self, cell_ids,):
        assert self.built == True
        if isinstance(cell_ids, int):
            cell_ids = [cell_ids]
        cell_id,_ = self.knearest_hotcells(cell_ids, k=1)
        return cell_id # : (None, 1)
    
    def save_KNVocabs(self,):
        V, D = np.zeros((self.knn_k, self.vocab_size)), np.zeros((self.knn_k, self.vocab_size))
        for vocab in range(self.vocab_start):
            V[:, vocab] = vocab
            D[:, vocab] = 0.
            
        kcells, dists = self.knearest_hotcells(list(self.vocab2hotcell.values()), k=5) # (#hotcells, k)
        uniq, inv = np.unique(kcells, return_inverse = True)
        kvocabs = np.array([self.hotcell2vocab[u_cell] for u_cell in uniq])[inv].reshape(kcells.shape)
        print(kvocabs.shape)
        
        V[:, self.vocab_start:] = kvocabs.transpose().astype('int')
        D[:, self.vocab_start:] = dists.transpose()
        
        pickle.dump({"V": V,"D": D}, 
                    open(data_dir/self.dataset_name/"{ds}KNVocabs_{cell_sz}.pkl".format(ds = self.dataset_name,
                                                                                        cell_sz = self.xstep
                                                                                       ), "wb"))
    def anycell2vocab(self, cell_id):
        """
        mapping a cell_id to vocab where the cell_id is not necessarily a hotcell
        if a cell_id is not one of hotcells, it is replaced with the nearest hotcell.
        """
        if cell_id in self.hotcell2vocab: # one of hotcells
            return self.hotcell2vocab[cell_id]
        else: # not a hotcell
            hotcell_id = self.nearest_hotcell(cell_id)[0,0] #(1,1)->scalar
            return self.hotcell2vocab[hotcell_id]
    
    def gps2vocab(self, lon,lat):
        if self.is_inregion(lon, lat):
            cell_id = self.gps2cell(lon, lat)
            return self.anycell2vocab(cell_id) # vocab
        else : 
            return "UNK"
        
    def trip2seq(self, trip):
        """
        @param trip : (2, traj_len) ::nd.array
        """
        seq = []
        for p in range(trip.shape[1]):
            # gps
            lon, lat = trip[:, p]
            seq.append(self.gps2vocab(lon, lat))
            
        return seq
    
    def seq2trip(self,seq):
        trip = np.zeros((2,len(seq)),dtype=np.float32)
        for point in range(len(seq)):
            hotcell_id = self.vocab2hotcell.get(seq[point], -1)
            if hotcell_id == -1 : raise ValueError
            lon, lat = self.cell2gps(hotcell_id)
            trip[:, point] = np.array([lon, lat])
            
        return trip
    
    def tripmeta(self, trip):
        """
        @param trip = (2, traj_len) ::nd.array
        """
        mins, maxs = np.min(trip, axis=1), np.max(trip, axis=1)
        lon_centroid, lat_centroid = mins + (maxs-mins)/2
        xoffset, yoffset = self.gps2offset(lon_centroid, lat_centroid)
        return xoffset, yoffset
    
    def seqmeta(self, seq):
        trip = self.seq2traj(seq)
        return self.tripmeta(trip)
    
    def seq2str(self, seq):
        """
        @param seq : list of points
        """
        seq_str = " ".join(list(map(str, seq))) + "\n"
        return seq_str
        
    def createTrainVal(self, trips_path, datapath, injectnoise,
                       ntrain, nval, nsplit=5, min_length=20, max_length=100,
                       zerolen_tripids=None
                      ):
        """
        @param datapath :: pathlib.POSIX("datapath")
        self.createTrainVal(trip_path, datapath, utils.downsampling,
                        800, 200, nsplit=5, min_length=20, max_length=100)
        @param zerolen_tripids :: list
        """
        trainsrc = open(datapath/self.dataset_name/"train.src", "w")
        traintrg = open(datapath/self.dataset_name/"train.trg", "w")
        trainmta = open(datapath/self.dataset_name/"train.mta", "w")
        
        validsrc = open(datapath/self.dataset_name/"valid.src", "w")
        validtrg = open(datapath/self.dataset_name/"valid.trg", "w")
        validmta = open(datapath/self.dataset_name/"valid.mta", "w")
        
        with h5py.File(trips_path, 'r') as f:
            for num in range(ntrain+nval):
                if num % 300 == 299 : 
                    print("Scanned {} trips".format(num+1))
                    
                if num+1 in set(zerolen_tripids): continue
                    
                trip = f["trips/"+str(num+1)][()] # nd.array (2,traj_len)                
                if not (min_length<= trip.shape[1]<=max_length):continue
                    
                trg = self.seq2str(self.trip2seq(trip))
                meta = self.tripmeta(trip)
#                 print(meta)
                mta = "{:.2f} {:.2f}\n".format(meta[0], meta[1])
                    
                noisetrips = injectnoise(trip, nsplit)
                srcio, trgio, mtaio = (trainsrc,traintrg,trainmta) if num<ntrain else (validsrc,validtrg,validmta)
                for noisetrip in noisetrips:
                    src = self.seq2str(self.trip2seq(noisetrip))
                    srcio.write(src)
                    trgio.write(trg)
                    mtaio.write(mta)
        
        trainsrc.close()
        traintrg.close()
        trainmta.close()
        validsrc.close()
        validtrg.close()
        validmta.close()