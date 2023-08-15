import warnings 
warnings.filterwarnings('ignore')

import os 
import sys 
import copy
import dill 
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

# # vectorized more efficient way for all pairs distances:
def L2Distance(data):
    # data: 2D-array of size num_samples x 2 (lat-lng values) 
    transposed = np.expand_dims(data, axis=1)
    distance = np.power(data-transposed, 2)
    distance = np.power(np.abs(distance).sum(axis=2), 0.5) 
    return distance

# ----------------------------------------------------------
# The status attribute will provide the current status of the model after solving. The possible status values in Gurobi include:
# GRB.OPTIMAL: The model has been solved to optimality.
# GRB.INFEASIBLE: The model is infeasible.
# GRB.UNBOUNDED: The model is unbounded.
# GRB.INF_OR_UNBD: The model is either infeasible or unbounded.
# GRB.CUTOFF: The model has been stopped by a user-specified cutoff.
# GRB.ITERATION_LIMIT: The solver has reached the iteration limit.
# GRB.NODE_LIMIT: The solver has reached the node limit.
# GRB.TIME_LIMIT: The solver has reached the time limit.
# GRB.SOLUTION_LIMIT: The solver has reached the solution limit.
# GRB.INTERRUPTED: The solver has been interrupted.
# GRB.NUMERIC: The model encountered a numerical error.
# GRB.SUBOPTIMAL: The model has been solved to a suboptimal solution.
# ----------------------------------------------------------
# import gc 
# del vars_and_costs 
# gc.collect()
# ----------------------------------------------------------
# # Python program to kill itself it if uses more than given amount of RAM usage!
# import resource
# import sys

# # Define the maximum allowed memory usage in bytes
# max_memory_usage = 2 * 1024 * 1024 * 1024  # 2GB

# # Get the current memory usage
# current_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# # Check if the current memory usage exceeds the threshold
# if current_memory_usage > max_memory_usage:
#     print("Memory usage exceeded the threshold. Terminating the program.")
#     sys.exit()

# # Rest of your program code...