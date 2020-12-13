#%%
import pandas as pd
import numpy as np
import h5py
import random 
#%%
def porto2h5(filepath, limit=None, fname=None, is_first=True):
    log_f = str(fname).split("/")[-1].split(".")[0]
    log = open("log/{}.txt".format(log_f), "w")
        
    df = pd.read_csv(filepath)
    df = df[df.MISSING_DATA == False]
    if limit:
        df = df.loc[:limit,]

    print("Processing {} trips...".format(len(df)))
    print(df.shape)
    df.sort_values("TIMESTAMP", inplace = True)
#     return
    with h5py.File(fname,"w") as f:
        num, num_incompleted, num_zerolen = 0, 0, 0
        for trip in df.POLYLINE:
            num += 1
#             if is_first:
#                 if num>800000:continue
#             else : # second_half
#                 if num<=800000:continue
                
                    
            if num % 10000 == 0 : 
                log.write("Scanning {}\n".format(str(num)))
                print("Scanning {}\n".format(str(num)))
            try:
                trip = np.array(eval(trip)) # (len, 2)
            except :
                num_incompleted +=1 
                continue
            tripLength = len(trip)
            if tripLength == 0 : 
                num_zerolen += 1
                continue  
            f["/trips/{}".format(num)] = trip.transpose()
            f["/timestamps/{}".format(num)] = list(range(0,tripLength*15,15))
                
        log.write("Incompleted trip: {}\nSaved {} trips\nZerolenTrip: {} trips.".format(num_incompleted, num, num_zerolen))
        log.close()
        print("Incompleted trip: {}\nSaved {} trips\nZerolenTrip: {} trips.".format(num_incompleted, num, num_zerolen))
        
# %% create a sequece of trjaectory (tripid, timestamp, lon, lat)
def porto2standardcsv(filepath, limit=None, fname=None):
    df = pd.read_csv(filepath)
    df = df[df.MISSING_DATA == False]
    if limit:
        df = df.loc[:limit,]
    trips = pd.DataFrame(columns = {"tripid","timestamps","lon","lat"})
    for i, row in df.iterrows():
        gps = np.array(eval(row.POLYLINE))
        tripLength = len(gps)
        if tripLength == 0 : continue
        trip = pd.DataFrame({
            'tripid' : np.repeat(row.TRIP_ID,tripLength),
            'timestamps' : np.repeat(row.TIMESTAMP,tripLength) + np.arange(0,tripLength)*15,
            'lon' : gps[:,0],
            'lat' : gps[:,1]
        })
        trips = trips.append(trip,ignore_index=True)
        if i % 1000 == 0 : print("Processed %d rows." % (i))
    if fname:
        trips.to_csv(fname,index=False)
    else : 
        trips.to_csv("/data/porto/preprocessed_porto.csv",index=False)
    
    return trips
# %%
"""
Distorting a trip using Gaussian noise
"""
def _distort(trip, rate, radius = 50.0):
    noisetrip = trip.copy()
    for i in range(1,noisetrip.shape[0]): # trip 몇개 들어오는거지? shape 맞춰야해.
        if random.random() <= rate:
            x, y = lonlat2meters(noisetrip[:,0],noisetrip[:,1])


"""
Accepting one trip and producing its 10 different noise rate distorted variants
"""
def distort(trip, nsplit):
    noisetrips = []

    for rate in range(0,0.9,0.1):
        noisetrip = _distort(trip, rate)
        noisetrips.append(noisetrip)
    return noisetrips

"""
Downsampling one trip, rate is dropping rate
"""
def _downsampling(trip, rate):
    keep_idx = [0]
    for i in range(1,trip.shape[1]-1): # trip 몇개 들어오는거지? shape 맞춰야해.
        if random.random() > rate : keep_idx.append(i)
    keep_idx.append(trip.shape[1]-1)
    return trip[:, keep_idx].copy()


"""
Accepting one trip and producing its 9 different lowsampling rate variants
"""
def downsampling(trip, nsplit):
    noisetrips = []
    dropping_rates = np.arange(0.,0.8,0.1)
    for rate in dropping_rates:
        noisetrip = _downsampling(trip, rate)
        noisetrips.append(noisetrip)
    return noisetrips

"""
First downsampling and then distorting the trip, producing its 20 different variants
"""
def downsamplingDistort(trip, nsplit):
    noisetrips = []
    dropping_rates = [0, 0.2, 0.4, 0.5, 0.6]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    for dropping_rate in dropping_rates :
        noisetrip1 = downsampling(trip, dropping_rate)
        for distorting_rate in distorting_rates :
            noisetrip2 = distort(noisetrip1, distorting_rate)
            noisetrips.append(noisetrip2)
    return noisetrips

# %%
"""
longitude, latitude to Web Mercator coordinate
"""
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * np.log((1 + t) / (1 - t))


def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = np.exp(y / 3189068.5)
    lat = np.arcsin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat
