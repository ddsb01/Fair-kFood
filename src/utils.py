import warnings 
warnings.filterwarnings('ignore')

import os 
import sys 
import copy
import lzma
import time
import pickle 
import random
import argparse
import itertools
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from copy import deepcopy
import matplotlib.pyplot as plt 
from collections import defaultdict

import logging

random.seed(1234567)

# Default Dict's Default value
def default_value():
    ''' 
    default value for defaultdict
    '''
    return -1

# Ensure Directory Paths
def ensure_dir(dir_path):
    ''' 
    Creates directory at "dir_path" if not already present.
    '''
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"Created directory: {dir_path}")
    return


# Logger
def get_logger(logs_path, log_filename):
    ''' 
    Initialize and return the logger.
    '''
    ensure_dir(logs_path)
    logging.basicConfig(filename=os.path.join(logs_path, log_filename), 
                        format='%(asctime)s  %(message)s', filemode='w')
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    return logger


# Store Data Structures
def picklify(data_struct, filename):
    with open(filename, 'wb') as f:
       pickle.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return

def depicklify(filename):
    with open(filename, 'rb') as f:
        data_struct = pickle.load(f)
    return data_struct

def picklify_compress(data_struct, filename):
    ''' 
    picklify + compression using the LZMA algorithm 
    '''
    print("Picklifying and Compressing ...")
    print("It might take a while ...")
    with lzma.open(filename, 'wb') as f:
        pickle.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return

def picklify_decompress(filename):
    # decompression will take extra time
    with lzma.open(filename, 'rb') as f:
        data_struct = pickle.load(f)
    return data_struct

def dillify(data_struct, filename):
    with open(filename, 'wb') as f:
        dill.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return

def depdillify(filename):
    with open(filename, 'rb') as f:
        data_struct = dill.load(f)
    return data_struct

def dillify_compress(data_struct, filename):
    print("Dillifying and Compressing ...")
    print("It might take a while ...")
    with lzma.open(filename, 'wb') as f:
        dill.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return

def dillify_decompress(filename):
    with lzma.open(filename, 'rb') as f:
        data_struct = dill.load(f)
    return data_struct

# Euclidean distance for points in R^2
def euclidean_dist(c1, c2):
    lat1, lng1 = c1[0], c1[1]
    lat2, lng2 = c2[0], c2[1] 
    dist = np.sqrt((lat1-lat2)**2 + (lng1-lng2)**2) 
    return dist

# vectorized more efficient way for all pairs distances:
def L2Distance(data):
    # data: 2D-array of size num_samples x 2 (lat-lng values) 
    transposed = np.expand_dims(data, axis=1)
    distance = np.power(data-transposed, 2)
    distance = np.power(np.abs(distance).sum(axis=2), 0.5) 
    return distance


def Convert2Seconds(entry):
    try: entry = entry.strip().split()
    except: print(entry)
    date, _time = entry[0], entry[1]
    hrs,mins,secs = map(int, _time.split(':'))
    seconds = hrs*3600+mins*60+secs
    return seconds


def get_load_characteristics(day, slot_size=30*60):
    if day<7: orders_df = pd.read_csv(f'./data/{city}/orders/{day}/_final_orders.csv')
    else: 
        orders_df = pd.read_csv(f'./data/{city}/orders/{day}/orders.csv')
        orders_df = orders_df.rename(columns={'restaurant_id':'rest_id'})
        orders_df = orders_df.dropna()
        orders_df['placed_ts'] = orders_df['placed_time'].apply(lambda x:Convert2Seconds(x))
    # print(orders_df)
    grouped_orders = orders_df.groupby('rest_id')
    rests = np.unique(orders_df['rest_id'])

    rest_load = pd.DataFrame(columns=['rest_id', 'slot', 'load'])
    numSlots = 86400//slot_size
    slots = [x for x in range(numSlots)]
    rest_load['rest_id'] = np.ndarray.flatten(np.array([[rest]*numSlots for rest in rests]))
    rest_load['slot'] = np.ndarray.flatten(np.array([slots]*len(rests)))
    rest_load['load'] = [0]*numSlots*len(rests)

    for rest in rests:
        rest_data = grouped_orders.get_group(rest)
        for slot, upperLimit in enumerate(range(slot_size, 86400+1, slot_size)):
            numRequests = rest_data[(rest_data['placed_ts']>=upperLimit-slot_size) & (rest_data['placed_ts']<=upperLimit)].shape[0]
            rest_load.loc[(rest_load.rest_id==rest) & (rest_load.slot==slot), 'load'] = numRequests
    storePath = os.path.join(data_path, f'{city}/orders/{day}/rest_load.csv')
    rest_load.to_csv(f'/home/daman/Desktop/k_server/code/data/{city}/orders/{day}/rest_load.csv', index=False)
    return


def lorenz(arrs, algos, colors, linestyles, plotFrac=0.25, type=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7.5)
    ax.grid(True, linestyle='dotted',color='0.3')
    plt.rcParams['font.size'] = '26'
    plt.rcParams['figure.figsize'] = (10,7.5)

    for idx, i in enumerate(algos):
        arr = np.array(arrs[idx]) 
        arr = np.sort(arr)
        plt.rcParams.update({'font.size': 14})
        scaled_prefix_sum = ( arr.cumsum() / arr.sum() )*100
        n = int(plotFrac * len(scaled_prefix_sum)) 
        ub = plotFrac*100 # upper-bound on plot
        lorenz_curve = np.insert(scaled_prefix_sum[:n], 0, 0) 
        plt.plot(np.linspace(0.0,ub,lorenz_curve.size), lorenz_curve, label=algos[idx], color=colors[idx], linestyle=linestyles[idx], linewidth=3) 
    plt.plot([0,ub],[0,ub],color='Black',linestyle='--') # line b/w points (0,0) and (ub,ub)
    plt.legend(loc='upper left',prop={'size':20})
    plt.xlabel('% of All Servers',fontsize=28)
    plt.ylabel('% of Total Rewards',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(f"./lorenz_{type}_{plotFrac}.pdf", dpi=300)


