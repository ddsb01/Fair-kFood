import warnings
warnings.filterwarnings('ignore')

import os 
import sys 
import copy
import dill 
import glob
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

import gurobipy as gb 
from gurobipy import GRB 

import networkx as nx 

import gc
import logging
from typing import List, Dict

from utils import *
from offline_solution import * 
from eval import calculate_infeasibility, get_server_rewards, get_optimal_cost

# random.seed(1234567)
INF = 1e20
eps = 1e-6
# ------------------------------------------------------------------ #
# Below methods might use variables declared exclusively in "run.py" #
# ------------------------------------------------------------------ #

class Server: 
    def __init__(self, id:int, init_node:int):
        self.id = id 
        self.curr_node = init_node
        self.active = {ts:False for ts in range(NUM_TIMESTAMPS+1)} 
        self.reward = 0.0
        self.requests = 0 # number of requests served


class Request:
    def __init__(self, placed_ts:int, prep_ts:int, deliver_ts:int, rest_node:int, cust_node:int):
        # timestamps
        self.placed_ts = placed_ts 
        self.prep_ts = prep_ts
        self.deliver_ts = deliver_ts 
        # locations
        self.rest_node = rest_node 
        self.cust_node = cust_node
        # status: serverd (or assigned)=1; unserved (or unassigned)=0
        self.status = 0 


def get_active_servers(intervals_datapath, t):
    ''' 
    This is applicable only to Swiggy data; NOT applicable to synthetic data!
    '''
    # os.remove(all_drivers_path)
    # os.remove(active_drivers_path)

    all_drivers_path = os.path.join(_intervals_datapath, f'all_drivers.pkl')
    active_drivers_path = os.path.join(_intervals_datapath, f'active_drivers_{t}.pkl')

    if os.path.exists(all_drivers_path) and os.path.exists(active_drivers_path):
        print("Reading from", all_drivers_path)
        print("Reading from", active_drivers_path)
        all_drivers = depicklify(all_drivers_path)
        active_drivers = depicklify(active_drivers_path)
    else:
        driver_files = sorted(glob.glob(os.path.join(intervals_datapath, '*')))
        num_files = len(driver_files)

        driver_ids = []
        for filename in driver_files:
            vh_id = filename.split("/")[-1][:-4]
            driver_ids.append(vh_id)

        all_drivers = {vh:{ts:0 for ts in range(86400+1)} for vh in driver_ids} 

        final_drivers = [] # drivers active at any time within first 't' hrs
        for idx, filename in enumerate(driver_files):
            taken = 0
            vid = driver_ids[idx]
            try: file_df = pd.read_csv(filename)
            except: continue 
            num_shifts = int(file_df.shape[0]/2)
            for i in range(num_shifts):
                start_ts = int(file_df.iloc[i*2].values[0].split()[0])
                end_ts = int(file_df.iloc[(i*2)+1].values[0].split()[0])
                for j in range(start_ts, end_ts+1):
                    if not taken and j<=t*3600: 
                        final_drivers.append(vid)
                        taken = 1
                    # exception handling to tackle: "invalid start_ts" or "invalid end_ts"
                    try: all_drivers[vid][j] = 1
                    except: continue
        picklify(all_drivers, all_drivers_path)

        active_drivers = {vh:all_drivers[vh] for vh in final_drivers}
        picklify(active_drivers, active_drivers_path)
        if t==24: active_drivers = all_drivers
    return active_drivers
    


def _weighted_random(curr_locations, requests_df, apsp_lengths, apsp_paths, heuristic=None):
    ''' 
    When a request 'r' arrives, it is assigned to the 'd-th' driver with probability proportional to probs[d] if he is active
    The probabilites are updated after the assignment of each request 
    post=> choose from the servers which are available and reachable
    '''
    numRuns = 5
    ACTIVE_DRIVERS = get_active_servers(ints_path, t_hrs)
    REWARDS = []
    ASSIGN_STATUS = []

    # breakpoint()

    def accumulate_results(arr):
        n = len(arr)
        acc_arr = np.sum(arr, axis=0)/n # avg over rows
        return acc_arr.tolist()
    
    def _update_normalize(rewards):
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            max_val = max(arr)
            arr = [(max_val-val) for val in arr] # INVERT 
            min_val, max_val, tot = min(arr), max(arr), sum(arr)
            if min_val!=max_val:
                arr = [(val/tot) for val in arr] # NORMALIZE 
            else: arr = [1/n]*n
        return arr
    
    def update_normalize(rewards):
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            max_val = max(arr)
            arr = [(max_val-val) for val in arr] # INVERT 
            min_val, max_val = min(arr), max(arr)
            if min_val==max_val: arr = [1]*n
        return arr
    
    def _update_normalize_alt(rewards): 
        # NOT equivalent to 'update_normalize'
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            arr = [(1/val+eps) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            min_val, max_val, tot = min(arr), max(arr), sum(arr)
            if min_val!=max_val: # also avoids division by 0
                arr = [(val/tot) for val in arr] # NORMALIZE 
            else: arr = [1/n]*n
        return arr       
    
    def update_normalize_alt(rewards): 
        # NOT equivalent to 'update_normalize'
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            # arr = [(1/val+eps) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            min_val, max_val = min(arr), max(arr)
            if min_val==max_val: arr = [1]*n
            else:
                arr = [20*((val-min_val)/(max_val-min_val)) for val in arr] # SCALING (to avoid overflow on exponentiation)
                arr = [(1/(2**val)) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
        return arr 
    
    ##
    for run in range(numRuns):
        seed_val = int(np.random.randint(1,100)*100)
        print("SEED VALUE:", seed_val)
        random.seed(seed_val)
        
        curr_locs = copy.deepcopy(curr_locations)
        vir_locs = copy.deepcopy(curr_locations) # virtual locations for heuristic

        active_drivers = copy.deepcopy(ACTIVE_DRIVERS)
        num_servers = len(active_drivers)
        rewards = {s:0.0 for s in list(active_drivers.keys())}
        num_requests = requests_df.shape[0]
        assign_status = [0]*num_requests
        probs = [1/num_servers]*num_servers

        for idx, request in tqdm(requests_df.iterrows(), total=requests_df.shape[0]):
            if idx: probs = update_normalize_alt(list(rewards.values()))
            pts, dts = request.prep_ts, request.deliver_ts
            if pts>=86400: continue # Invalid; Ignore
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist 

            # make a list of eligible 'available' and 'reachable' drivers for current request
            eligible_drivers = []
            sub_probs = {}
            # breakpoint()
            for s_idx, did in enumerate(active_drivers.keys()):
                curr_server_node = vir_locs[s_idx] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = (active_drivers[did][pts] and active_drivers[did][dts]) # Test "availability"
                else: test_val = (active_drivers[did][pts] and active_drivers[did][dts]) and sp_time<=fpt # Test "availability" and "reachability"

                if test_val:
                    eligible_drivers.append(did)
                    sub_probs[did] = rewards[did]
            assert len(eligible_drivers)==len(sub_probs), "Incorrect length of probability list"
            
            # print(len(eligible_drivers))
            winner = None
            if len(eligible_drivers)>0:
                # active-time-adjusted probabilities: 
                pat_probs = copy.deepcopy(sub_probs) # per-active-timestep rewards
                for vid in pat_probs:
                    num_active_ts = 0 
                    for ts in range(pts):
                        num_active_ts += active_drivers[vid][ts] 
                    num_active_ts = max(request.prep_time, num_active_ts)
                    pat_probs[vid] /= num_active_ts 

                # select a server proportional to it's weight; 
                sub_probs = update_normalize_alt(list(pat_probs.values())) 
                chosen = random.choices(range(len(eligible_drivers)), sub_probs, k=1)[0]
                it = eligible_drivers[chosen] # driver id
                curr_server_node = curr_locs[chosen] 
                
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']
    
                # compute changes
                first_mile_reward = sp_dist
                last_mile_reward = lmd
                # update server data
                if only_last_mile: rewards[it] += last_mile_reward 
                else: rewards[it] += (first_mile_reward + last_mile_reward)
            
                for t in range(pts, dts+1):
                    active_drivers[it][t] = 0
                curr_locs[chosen] = c_node
                vir_locs[chosen] = c_node
                assign_status[idx] += 1
                assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
                winner = chosen # index 
            else:
                # the chosen server was uneligible to server the request; 
                # hence the request remains unserved
                continue

            # < Double-Coverage heuristic > # [all servers move (virtuall, w/o cost) towards the current request with equal speed until it gets served]
            # <=> here, since movement is discrete node-to-node, all servers move to the request node  
            if heuristic=='double-coverage':
                # for it in range(num_servers):
                #     if it!=winner and curr_locs[it]!='src':
                #         curr_locs[it] = r_node
                # <=> here, since movement is discrete node-to-node, each servers moves to the next nearest node to the request node 
                
                # move towards the current request node
                for it, vid in enumerate(active_drivers.keys()):
                    if it!=winner:
                        d_node = vir_locs[it]
                        if d_node!='src' and not active_drivers[vid][pts]: # only idle drivers that are not at 'src' will move
                            path_to_rest = apsp_paths[r_node][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                            num_path_nodes = len(path_to_rest)
                            steps = 1 # num_path_nodes//2
                            next_node = path_to_rest[(num_path_nodes-1)-steps]
                            if steps<=num_path_nodes:
                                vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next node in the shortest path from d_node to r_node
                            else: vir_locs[it] = r_node # <=> path_to_rest[-1]

                # move all unobstructed (available) servers towards the nearest request node
                for it, vid in enumerate(active_drivers.keys()):
                    if it!=winner:
                        d_node = vir_locs[it]
                        if d_node!='src' and active_drivers[vid][pts]:
                            nearestRest = requestNodes[np.argmin([apsp_lengths[rest][d_node]['time'] for rest in requestNodes])]
                            path_to_rest = apsp_paths[nearestRest][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                            num_path_nodes = len(path_to_rest)
                            steps = 1 # num_path_nodes//2
                            if steps<=num_path_nodes:
                                vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next-node in the shortest path from d_node to r_node
                                # rewards[it] += apsp_lengths[nearestRest][d_node]['dist']*0.1 # 0.1 is the waiting-time-travel per unit distance reward!
                            else: vir_locs[it] = nearestRest # <=> path_to_rest[-1]

        # accumulate results 
        REWARDS.append(rewards)
        ASSIGN_STATUS.append(assign_status)
    
    REWARDS = [list(r.values()) for r in REWARDS]
    return REWARDS, ASSIGN_STATUS


def _round_robin(curr_locs, requests_df, apsp_data):
    active_servers = get_active_servers(ints_path, t_hrs)
    num_servers = len(active_drivers)
    rewards = {s:0.0 for s in list(active_drivers.keys())}
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    vids = list(active_drivers.keys())
    
    s_idx = 0 # server index of server to which the current request should be assigned
    for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
        pts, dts = request.prep_ts, request.deliver_ts
        if pts>=86400: continue
        r_node, c_node = request.rest_node, request.cust_node
        fpt = request.prep_time
        lmd = request.deliver_dist

        eligible_drivers = []
        for vidx, d in enumerate(active_servers.keys()):
            # storing vidx is required for curr_locs later
            curr_server_node = curr_locs[vidx] 
            vid = vids[vidx]
            if curr_server_node=='src':
                sp_dist, sp_time = apsp_data[curr_server_node][r_node]['dist'], apsp_data[curr_server_node][r_node]['time']
            else:
                sp_dist, sp_time = apsp_data[r_node][curr_server_node]['dist'], apsp_data[r_node][curr_server_node]['time']
            if ONLY_AVAIL: test_val = active_servers[d][pts] # Test "availability"
            else: test_val = active_servers[d][pts] and sp_time<=fpt # Test "availability" and "reachability"
            if test_val: eligible_drivers.append((vidx, d))

        win_idx, winner = None, None
        if len(eligible_drivers)>0:
            for vidx, vid in eligible_drivers:
                if vidx>=s_idx:
                    win_idx, winner = vidx, vid 
                    break 

            if winner is None: continue 
            
            curr_server_node = curr_locs[win_idx] 
            first_mile_reward = sp_dist
            last_mile_reward = lmd
            # update server data
            if only_last_mile: rewards[s_idx] += last_mile_reward 
            else: rewards[vid] += (first_mile_reward + last_mile_reward)
                    
            for t in range(pts, dts+1): active_servers[vid][t] = 0
            curr_locs[win_idx] = c_node
            assign_status[idx] += 1
            assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
            # o/w the request could not be assigned to the s_idx-th server
        else: s_idx = (win_idx+1) % num_servers

    return list(rewards.values()), assign_status


def _greedy(curr_locs, requests_df, apsp_lengths, apsp_paths, strategy, heuristic=None):
    '''
    strategy:
        'min': assign the order to the active server 'd' with minimum current reward 
               it is a special case of 'weighted_random' with probs[x]=1 if x==d else 0
        'min_diff': assign the order such that the max_reward-min_reward after assigning this order will be minimum
                it is similar to "FairFoody's" bipartite graph weight heuristic

    '''
    active_servers = get_active_servers(ints_path, t_hrs)
    num_servers = len(active_servers)
    rewards = {s:0.0 for s in list(active_servers.keys())}
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    vids = list(active_servers.keys())

    vir_locs = copy.deepcopy(curr_locs)
    if strategy=='min':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            pts, dts = request.prep_ts, request.deliver_ts
            if pts>=86400: continue
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist
            
            # make a list of eligible 'available' and 'reachable' drivers for current request
            eligible_drivers = []
            for vidx, d in enumerate(active_servers.keys()):
                # storing idx is required for curr_locs later
                curr_server_node = vir_locs[vidx] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active_servers[d][pts] # Test "availability"
                else: test_val = active_servers[d][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if test_val: eligible_drivers.append((vidx, d)) 

            # print(len(eligible_drivers))
            win_idx, winner = None, None
            if len(eligible_drivers)>0:
                pat_rewards = {vid:0 for vid in eligible_drivers} # per-active-timestep rewards
                for (vidx, vid) in eligible_drivers:
                    num_active_ts = 0 
                    for ts in range(pts):
                        num_active_ts += active_servers[vid][ts] 
                    num_active_ts = max(request.prep_time+request.deliver_time, num_active_ts)
                    pat_rewards[vid] = rewards[vid]/num_active_ts
             
                curr_min = float('inf')
                for (vidx, vid) in eligible_drivers:
                    if pat_rewards[vid]<curr_min:
                        curr_min = pat_rewards[vid] 
                        win_idx, winner = vidx, vid

                curr_server_node = curr_locs[win_idx] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']
    
                # compute changes
                first_mile_reward = sp_dist
                last_mile_reward = lmd
                # update server data
                if only_last_mile: rewards[winner] += last_mile_reward 
                else: rewards[winner] += (first_mile_reward + last_mile_reward)
                
                for t in range(pts, dts+1): active_servers[winner][t] = 0
                curr_locs[win_idx] = c_node
                vir_locs[win_idx] = c_node
                assign_status[idx] += 1
                assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
            else:
                # the chosen server was uneligible to server the request
                # hence the request remains unserved
                continue
            

            # < Double-Coverage heuristic > # [all servers move (virtuall, w/o cost) towards the current request with equal speed until it gets served]
            # <=> here, since movement is discrete node-to-node, all servers move to the request node  
            if heuristic=='double-coverage':
                # # move towards the current request node
                # for it, vid in enumerate(active_servers.keys()):
                #     if it!=win_idx and active_servers[vid][pts]: # vid!=winner
                #         d_node = curr_locs[it]
                #         if d_node=='src': continue
                #         path_to_rest = apsp_paths[r_node][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                #         num_path_nodes = len(path_to_rest)
                #         steps = 1 # num_path_nodes//2
                #         if steps<=num_path_nodes:
                #             curr_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next node in the shortest path from d_node to r_node
                #         else: curr_locs[it] = r_node # <=> path_to_rest[-1]

                # move towards the nearest request node
                for it, vid in enumerate(active_drivers.keys()):
                    if it!=win_idx:
                        d_node = vir_locs[it]
                        if d_node!='src' and active_drivers[vid][pts]:
                            nearestRest = requestNodes[np.argmin([apsp_lengths[rest][d_node]['time'] for rest in requestNodes])]
                            path_to_rest = apsp_paths[nearestRest][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                            num_path_nodes = len(path_to_rest)
                            steps = 1 # num_path_nodes//2
                            if steps<=num_path_nodes:
                                vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next-node in the shortest path from d_node to r_node
                                # rewards[it] += apsp_lengths[nearestRest][d_node]['dist']*0.1 # 0.1 is the waiting-time-travel per unit distance reward!
                            else: vir_locs[it] = nearestRest # <=> path_to_rest[-1]

                
    elif strategy=='min_diff':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            pts, dts = request.prep_ts, request.deliver_ts
            if pts>=86400: continue
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist
            # breakpoint()
            # Try every server, choose the one which gives min difference between the min and max reward
            # assign with a one step look ahead to the state of 'rewards'  
            winner = None # the server to which the request should be assigned
            winner_reward = None
            min_diff = INF
            for it in range(num_servers):
                curr_server_node = curr_locs[it] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                vid = vids[it]
                if ONLY_AVAIL: test_val = (active_servers[vid][pts] and active_servers[vid][dts]) # Test "availability"
                else: test_val = (active_servers[vid][pts] and active_servers[vid][dts]) and sp_time<=fpt # Test "availability" and "reachability"

                if not test_val: continue
                
                curr_server_node = curr_locs[it] 
                first_mile_reward = sp_dist
                last_mile_reward = lmd

                rewards_copy = copy.deepcopy(rewards)
                if only_last_mile: rewards_copy[vid] += last_mile_reward
                else: rewards_copy[vid] += (first_mile_reward + last_mile_reward)
                
                pat_rewards = copy.deepcopy(rewards_copy) # per-active-timestep rewards
                for vid in pat_rewards:
                    num_active_ts = 0 
                    for ts in range(pts):
                        num_active_ts += active_drivers[vid][ts] 
                    num_active_ts = max(request.prep_time+1, num_active_ts)
                    pat_rewards[vid] /= num_active_ts 

                diff = max(pat_rewards.values()) - min(pat_rewards.values())
                if diff<min_diff:
                    min_diff = diff 
                    winner = it
                    winner_reward = rewards_copy[vid]
                    assign_status[idx] = 1
            # assert winner!=-1 and winner_reward!=-1, 'Incorrect computation of winner'
            # update changes
            if assign_status[idx]: # => winner!=-1
                win_id = vids[winner]
                rewards[win_id] = winner_reward
                for t in range(pts, dts+1):
                    active_servers[win_id][t] = 0
                curr_locs[winner] = c_node

    return list(rewards.values()), assign_status



def weighted_random(num_servers, curr_locations, requests_df, apsp_lengths, apsp_paths, heuristic=None):
    ''' 
    When a request 'r' arrives, it is assigned to the 'd-th' driver with probability proportional to probs[d] if he is active
    The probabilites are updated after the assignment of each request 
    post=> choose from the servers which are available and reachable
    '''
    REWARDS, ASSIGN_STATUS = [], []
    numRuns = 5

    def _update_normalize(rewards):
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            max_val = max(arr)
            arr = [(max_val-val) for val in arr] # INVERT 
            min_val, max_val, tot = min(arr), max(arr), sum(arr)
            if min_val!=max_val:
                arr = [(val/tot) for val in arr] # NORMALIZE 
            else: arr = [1/n]*n
        return arr
    
    def update_normalize(rewards):
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            max_val = max(arr)
            arr = [(max_val-val) for val in arr] # INVERT 
            min_val, max_val = min(arr), max(arr)
            if min_val==max_val: arr = [1]*n
        return arr
    
    def _update_normalize_alt(rewards): 
        # NOT equivalent to 'update_normalize'
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            # arr = [(1/val+eps) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            max_val = max(arr)
            # arr = [(1/2**(max_val-val)) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            min_val, max_val = min(arr), max(arr)
            if min_val!=max_val: # also avoids division by 0
                arr = [(val/tot) for val in arr] # NORMALIZE 
            else: arr = [1/n]*n
        return arr      
    
    def update_normalize_alt(rewards): 
        # NOT equivalent to 'update_normalize'
        n = len(rewards)
        arr = copy.deepcopy(rewards)
        if n>0:
            # arr = [(1/val+eps) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            min_val, max_val = min(arr), max(arr)
            if min_val==max_val: arr = [1]*n
            else:
                arr = [20*((val-min_val)/(max_val-min_val)) for val in arr] # SCALING (to avoid overflow on exponentiation)
                arr = [(1/(2**val)) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
        return arr     
    
    for run in range(numRuns):
        seed_val = int(np.random.randint(1,11)*100)
        print("SEED VALUE:", seed_val)
        random.seed(seed_val)

        curr_locs = copy.deepcopy(curr_locations)
        vir_locs = copy.deepcopy(curr_locations) # virtual locations for heuristic

        rewards = [0.0]*num_servers 
        num_requests = requests_df.shape[0]
        assign_status = [0]*num_requests
        active = [{ts:1 for ts in range(NUM_TIMESTAMPS+1)} for d in range(num_servers)]
        probs = [1/num_servers]*num_servers

        for idx, request in tqdm(requests_df.iterrows(), total=requests_df.shape[0]):
            if idx: probs = update_normalize_alt(rewards) # update_normalize_alt(rewards)
            # breakpoint()
            pts, dts = request.prep_ts, request.deliver_ts
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist 

            # make a list of eligible 'available' and 'reachable' drivers for current request
            eligible_drivers = []
            sub_probs = []
            for d in range(num_servers):
                # curr_server_node = curr_locs[d]
                curr_server_node = vir_locs[d] # check with virtual locations
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
                else: test_val = active[d][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if test_val:
                    eligible_drivers.append(d)
                    sub_probs.append(rewards[d])
            assert len(eligible_drivers)==len(sub_probs), "Incorrect length of probability list"
            
            # print(len(eligible_drivers))
            winner = None
            if len(eligible_drivers)>0:
                # select a server proportional to it's weight; 
                sub_probs = update_normalize_alt(sub_probs) # update_normalize_alt(sub_probs)
                # assert sum(sub_probs)<=1.01, f'Incorrect weight normalization; current sum is {sum(sub_probs)}'
                # assert sum(sub_probs)>=0.99, f'Incorrect weight normalization; current sum is {sum(sub_probs)}'
                # breakpoint()
                it = random.choices(eligible_drivers, sub_probs, k=1)[0]
                curr_server_node = curr_locs[it] # Pay using actual locations
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']
    
                # compute changes
                first_mile_reward = sp_dist
                last_mile_reward = lmd
                # update server data
                if only_last_mile: rewards[it] += last_mile_reward 
                else: rewards[it] += (first_mile_reward + last_mile_reward)
            
                for t in range(pts, dts+1): active[it][t] = 0 
                curr_locs[it] = c_node 
                vir_locs[it] = c_node
                assign_status[idx] = 1
                # assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
                winner = it
            else:
                # the chosen server was uneligible to server the request; 
                # hence the request remains unserved
                continue
            

            # < Double-Coverage heuristic > # [all servers move (virtuall, w/o cost) towards the current request with equal speed until it gets served]
            # <=> here, since movement is discrete node-to-node, all servers move to the request node  
            if heuristic=='double-coverage':        
                # move towards the nearest "request node"
                for it in range(num_servers):
                    # if 'src' in set(curr_locs): continue
                    if it!=winner: # vid!=winner
                        d_node = vir_locs[it]
                        if d_node=='src' or not active[it][pts]: continue
                        nearestRest = requestNodes[np.argmin([apsp_lengths[rest][d_node]['time'] for rest in requestNodes])]
                        path_to_rest = apsp_paths[nearestRest][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                        num_path_nodes = len(path_to_rest)
                        steps = 1 # num_path_nodes//2
                        if steps<=num_path_nodes:
                            vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next-node in the shortest path from d_node to r_node
                            # rewards[it] += apsp_lengths[nearestRest][d_node]['dist']*0.1 # 0.1 is the waiting-time-travel per unit distance reward!
                        else: vir_locs[it] = nearestRest # <=> path_to_rest[-1]         

        # accumulate results
        REWARDS.append(rewards)
        ASSIGN_STATUS.append(assign_status) 
    return REWARDS, ASSIGN_STATUS


def round_robin(num_servers, curr_locs, requests_df, apsp_data):
    rewards = [0.0]*num_servers
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    active = [{ts:1 for ts in range(NUM_TIMESTAMPS+1)} for d in range(num_servers)]

    # only upto this line, 'round_robin' differs from '_round_robin'

    s_idx = 0 # server index of server to which the current request should be assigned
    for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
        pts, dts = request.prep_ts, request.deliver_ts
        r_node, c_node = request.rest_node, request.cust_node
        fpt = request.prep_time
        lmd = request.deliver_dist
        # if s_idx-th server is not active then we skip it
        # make a list of eligible 'available' and 'reachable' drivers for current request
        eligible_drivers = []
        for d in range(num_servers):
            curr_server_node = curr_locs[d] 
            if curr_server_node=='src':
                sp_dist, sp_time = apsp_data[curr_server_node][r_node]['dist'], apsp_data[curr_server_node][r_node]['time']
            else:
                sp_dist, sp_time = apsp_data[r_node][curr_server_node]['dist'], apsp_data[r_node][curr_server_node]['time']

            if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
            else: test_val = active[d][pts] and sp_time<=fpt # Test "availability" and "reachability"

            if test_val: eligible_drivers.append(d)

        winner = None
        if len(eligible_drivers)>0:
            # assign the request to the next eligible driver in the round 
            for did in eligible_drivers:
                if did>=s_idx: 
                    winner = did
                    break 
            
            if winner is None: continue

            curr_server_node = curr_locs[winner]
            if curr_server_node=='src':
                sp_dist, sp_time = apsp_data[curr_server_node][r_node]['dist'], apsp_data[curr_server_node][r_node]['time']
            else:
                sp_dist, sp_time = apsp_data[r_node][curr_server_node]['dist'], apsp_data[r_node][curr_server_node]['time']

            # compute changes
            first_mile_reward = sp_dist
            last_mile_reward = lmd
            # update server data
            if only_last_mile: rewards[winner] += last_mile_reward 
            else: rewards[winner] += (first_mile_reward + last_mile_reward)
            
            for t in range(pts, dts+1): active[winner][t] = 0
            curr_locs[winner] = c_node
            assign_status[idx] += 1
            assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
        else:
            # the chosen server was uneligible to server the request; 
            # hence the request remains unserved
            continue
        s_idx = (winner + 1) % num_servers

    return rewards, assign_status


def greedy(num_servers, curr_locs, requests_df, apsp_lengths, apsp_paths, strategy, heuristic=None):
    '''
    strategy:
        'min': assign the order to the active server 'd' with minimum current reward 
               it is a special case of 'weighted_random' with probs[x]=1 if x==d else 0
        'min_diff': assign the order such that the max_reward-min_reward after assigning this order will be minimum
                it is similar to "FairFoody's" bipartite graph weight heuristic

    '''
    rewards = [0.0]*num_servers
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    active = [{ts:1 for ts in range(NUM_TIMESTAMPS+1)} for d in range(num_servers)] 

    vir_locs = copy.deepcopy(curr_locs)
    if strategy=='min':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            pts, dts = request.prep_ts, request.deliver_ts
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist
            
            # make a list of eligible 'available' and 'reachable' drivers for current request
            eligible_drivers = []
            for d in range(num_servers):
                # curr_server_node = curr_locs[d]
                curr_server_node = vir_locs[d] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
                else: test_val = active[d][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if test_val: eligible_drivers.append(d)

            # print(len(eligible_drivers))
            winner = None
            if len(eligible_drivers)>0:
                curr_min = float('inf')
                for didx in eligible_drivers:
                    if rewards[didx]<curr_min:
                        curr_min = rewards[didx] 
                        winner = didx

                curr_server_node = curr_locs[winner] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']
    
                # compute changes
                first_mile_reward = sp_dist
                last_mile_reward = lmd
                # update server data
                if only_last_mile: rewards[winner] += last_mile_reward 
                else: rewards[winner] += (first_mile_reward + last_mile_reward)
                
                for t in range(pts, dts+1): active[winner][t] = 0
                curr_locs[winner] = c_node
                vir_locs[winner] = c_node
                assign_status[idx] = 1
                assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
            else:
                # the chosen server was uneligible to server the request; 
                # hence the request remains unserved
                continue
            

            # < Double-Coverage heuristic > # [all servers move (virtuall, w/o cost) towards the current request with equal speed until it gets served]
            # <=> here, since movement is discrete node-to-node, all servers move to the request node  
            if heuristic=='double-coverage':
                # move towards the nearest "request node"
                for it in range(num_servers):
                    # if 'src' in set(curr_locs): continue
                    if it!=winner: # vid!=winner
                        d_node = vir_locs[it]
                        if d_node=='src' or not active[it][pts]: continue
                        nearestRest = requestNodes[np.argmin([apsp_lengths[rest][d_node]['time'] for rest in requestNodes])]
                        path_to_rest = apsp_paths[nearestRest][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                        num_path_nodes = len(path_to_rest)
                        steps = 1 # num_path_nodes//2
                        if steps<=num_path_nodes:
                            vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps] # the next-node in the shortest path from d_node to r_node
                            # rewards[it] += apsp_lengths[nearestRest][d_node]['dist']*0.1 # 0.1 is the waiting-time-travel per unit distance reward!
                        else: vir_locs[it] = nearestRest # <=> path_to_rest[-1]       

                
    elif strategy=='min_diff':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            pts, dts = request.prep_ts, request.deliver_ts
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist
            # breakpoint()
            
            # make a list of eligible 'available' and 'reachable' drivers for current request
            eligible_drivers = []
            for d in range(num_servers):
                curr_server_node = curr_locs[d] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
                else: test_val = active[d][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if test_val: eligible_drivers.append(d)
            
            # Try every server, choose the one which gives min difference between the min and max reward
            # assign with a one step look ahead to the state of 'rewards'  
            winner = None # the server to which the request should be assigned
            winner_reward = None
            min_diff = INF
            for it in eligible_drivers:
                curr_server_node = curr_locs[it] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active[it][pts] # Test "availability"
                else: test_val = active[it][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if not test_val: continue
                
                first_mile_reward = sp_dist
                last_mile_reward = lmd

                rewards_copy = copy.deepcopy(rewards)
                if only_last_mile: rewards_copy[it] += last_mile_reward
                else: rewards_copy[it] += (first_mile_reward + last_mile_reward)

                diff = max(rewards_copy) - min(rewards_copy)
                if diff<=min_diff:
                    min_diff = diff 
                    winner = it 
                    winner_reward = rewards_copy[it]
                    assign_status[idx] = 1
            
            # assert winner!=-1 and winner_reward!=-1, 'Incorrect computation of winner'
            # update changes
            if assign_status[idx]: # => winner!=-1
                rewards[winner] = winner_reward
                for t in range(pts, dts+1): active[winner][t] = 0
                curr_locs[winner] = c_node

    return rewards, assign_status




# ''' 
if __name__=='__main__':
    # get program inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['A', 'B', 'C'], default='A', 
                        type=str, required=True, help='City name')
    parser.add_argument('--day', choices=[x for x in range(10)], default=1,
                        type=int, required=True, help='Day index')
    parser.add_argument('--t', choices=[-1]+[x for x in range(1,25)], default=-1, 
                        type=int, required=True, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='time',
                        type=str, required=True, help='Weight variable for road network')
    parser.add_argument('--only_last_mile', choices=[0, 1], default=False, 
                        type=int, required=True, help='Consider only last-mile rewards (or costs)')
    parser.add_argument('--method', choices=['random', 'random*', 'min', 'min*', 'min_diff', 'round-robin'], default='random', 
                        type=str, required=True, help='Type of Objective function')
    parser.add_argument('--timestep', default=1,
                        type=int, required=False, help='The entire consists of "(86400//timestep)" units of time')
    parser.add_argument('--odd_dataset', choices=[1, 2], default=1,
                        type=int, required=False, help='whether to take odd timestamped data <=> slotted data with timestep=2 (choose 2 as input) AND distinct prep_tss !')
    parser.add_argument('--init', choices=[0,1], default=0,
                        type=int, required=False, help='Consider the initial locations of the servers')
    parser.add_argument('--batch_size', choices=[1,2,3,4,5,6,7,8,9,10], default=1, 
                    type=int, required=False, help='Average batch-/cluster-size when "batching" is done')
    parser.add_argument('--always_active', choices=[0,1], 
                        type=int, required=True)
    parser.add_argument('--test_only_availability', choices=[0,1],
                        type=int, required=True)
    args = parser.parse_args()

    # create input variables
    city = args.city 
    day = args.day 
    t_hrs = args.t
    weight_var = args.weight_var
    only_last_mile = args.only_last_mile
    method = args.method
    timestep = args.timestep 
    ODD = args.odd_dataset
    INIT = args.init
    batch_size = args.batch_size 
    ALWAYS_ACTIVE = args.always_active
    ONLY_AVAIL = args.test_only_availability

    # metadata
    data_path = '/home/daman/Desktop/k_server/code/data/'
    logs_path = os.path.join(data_path, city, 'logs')
    # ints_path = os.path.join(data_path, city, 'drivers/de_intervals')
    _intervals_datapath = os.path.join(data_path, city, f'drivers/{day}')
    ints_path = os.path.join(data_path, city, f'drivers/{day}/de_intervals')
    name2id = {'A':10, 'B':1, 'C':4}
    id2name = {v:k for k,v in name2id.items()}

    # LOAD DATA
    print("Loading data ...") 
    if timestep==1:
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders.csv'))
        # orders_data = orders_data.rename(columns={'rest_node':'begin_node', 'cust_node':'end_node'})
    else:
        # this violates distinct prep_ts assumption
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_slotted_{timestep}.csv'))
    
    if ODD==2:
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_odd_timestamps.csv'))
        
    apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_dist_t={t_hrs}.pkl'))
    apsp_time = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_time_t={t_hrs}.pkl'))   

    paths_dist = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_lists_dist_t={t_hrs}.pkl'))
    paths_time = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_lists_time_t={t_hrs}.pkl')) 

    # the above 2 files only contain the shortest paths between the restaruant nodes active within the first t_hrs and all other nodes in the road network.           
    road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/{day}/u_v_time_dist'), header=None, sep=' ', names=['u', 'v', 'time', 'dist'])
    idx2node = pd.read_csv(os.path.join(data_path, f'{city}/map/index_to_node_id.csv'), header=None, names=['idx', 'node_id'])
    node2idx = pd.read_csv(os.path.join(data_path, f'{city}/map/node_id_to_index.csv'), header=None, names=['node_id', 'idx'])
    drivers_init_path = pd.read_csv(os.path.join(data_path, f'{city}/drivers/{day}/driver_init_nodes.csv'))
    
    # consider only those orders placed until t_hrs since 00:00:00
    orig_orders_data = deepcopy(orders_data)
    orders_data = orders_data[orders_data.placed_time <= t_hrs*3600].reset_index() 
    # orders_data = orders_data[orders_data.placed_time <= 1800].reset_index() # 30-minutes
    # orders_data = orders_data[orders_data.placed_time <= 600].reset_index() # 10-minutes
    # orders_data = orders_data[orders_data.placed_time <= 300].reset_index() # 5-minutes
    # orders_data = orders_data[orders_data.placed_time <= 180].reset_index() # 3-minutes
    requestNodes = list(np.unique(orders_data.rest_node)) 

    if INIT: 
        orders_data = orders_data[orders_data.placed_time <= 120].reset_index() # 2-minutes # less time since each server must have a separate flow in this case

    all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist=apsp_dist, apsp_time=apsp_time)
    paths = parse_all_pair_shortest_paths(apsp_dist=paths_dist, apsp_time=paths_time)
    # # consider only those orders placed in the t-th hour
    # orders_data = orders_data[((t_hrs-1)*3600) <= orders_data.placed_time <= (t_hrs*3600)] 

    # Create required variables and data 
    ALL_NODES = idx2node.idx.values # or np.arange(idx2node.shape[0])
    NUM_NODES = idx2node.shape[0]  # or len(ALL_NODES)
    ORIG_ALL_NODES = ALL_NODES
    ORIG_NUM_NODES = NUM_NODES
    
    ###### INPUT SUMMARY BEGINS ###########
    NUM_REQUESTS = orders_data.shape[0] 
    # NUM_SERVERS = drivers_data.shape[0] * batch_size # when considering all 24 hours' data
    # NUM_SERVERS = (NUM_REQUESTS//10) * batch_size # heuristic
    total_num_rests = orig_orders_data['rest_node'].unique().shape[0] 
    # let's assume that this (#active restaurants in day "day") is the max number of drivers active at the same the during a day 
    curr_num_rests = orders_data['rest_node'].unique().shape[0] # number of active restaurants in chosen subset of data
    # _NUM_SERVERS = min(curr_num_rests, total_num_rests) # heuristic
    # _NUM_SERVERS = NUM_REQUESTS
    # if timestep>1: _NUM_SERVERS = NUM_REQUESTS//2 + 2
    active_drivers = get_active_servers(ints_path, t_hrs)
    # breakpoint()
    num_active_drivers = len(active_drivers)
    NUM_SERVERS = min(num_active_drivers, NUM_REQUESTS)
    # We don't want NUM_SERVERS>NUM_REQUESTS because fractional servers treatment can't work with such scenarios
    # as it might lead to no infeasibility (by dividing each server equally among all requests) even when it's a clear case of infeasibility!
    
    if INIT: # better for real-data
        curr_rest_nodes = list(orders_data['rest_node'].unique())
        INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)
    # if INIT: # better for synthetic data
    #     INIT_NODES = get_init_nodes(drivers_init_path)
    else:
        # dummy node with 0-distance from all nodes
        INIT_NODES = ['src']*NUM_SERVERS
        all_pairs_shortest_paths['src'] = {node:{'dist':0, 'time':0} for node in range(ORIG_NUM_NODES)}
        # for node in ORIG_ALL_NODES:
        #     all_pairs_shortest_paths[node]['src'] = {'dist':0, 'time':0}

    max_deliver_timestamp = np.max(orders_data['deliver_ts'].values)
    NUM_TIMESTAMPS = max_deliver_timestamp  # time-step = 1 second # 1 day = 86400 seconds
    MAX_FPT = np.max(orders_data.prep_time.values)
    MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
    MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
    MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
    MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

    print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
    print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
    print(f"Number of requests (or orders): {NUM_REQUESTS}")
    print(f"Number of servers (or drivers): {NUM_SERVERS}")
    # print(f"Total number of nodes in the LP or Flow network: {NUM_VARS}")
    # # There exist 97282 edges in the road network
    
    # GET LOGGER:
    global logger
    log_filename = ''
    if INIT: log_filename = f"{method}_online_INIT_{t_hrs}_{NUM_REQUESTS}_{NUM_SERVERS}_{weight_var}.log" 
    elif ODD==2: log_filename = f"online_ODD_{t_hrs}_{NUM_REQUESTS}_{NUM_SERVERS}_{weight_var}.log" 
    elif timestep>1: log_filename = f"online_SLOT_{t_hrs}_{NUM_REQUESTS}_{NUM_SERVERS}_{weight_var}.log"
    else: log_filename = f"{method}{ALWAYS_ACTIVE}_online_{t_hrs}_{NUM_REQUESTS}_{NUM_SERVERS}_{weight_var}.log"
    logger = get_logger(logs_path, log_filename) 

    logger.info(f"NUM_NODES : {NUM_NODES}")
    logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
    logger.info(f"NUM_SERVERS : {NUM_SERVERS}")
    logger.info(f"NUM_TIMESTAMPS : {NUM_TIMESTAMPS}")
    logger.info(f"t : {t_hrs}")
    ########## INPUT SUMMARY ENDS #################
    # breakpoint()


    # SOLVING #
    solve_start_time = time.time()
    print("Online solution started ...")
    if method=='random':
        if ALWAYS_ACTIVE:
            rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths) 
        else: 
            rewards, assign_status = _weighted_random(INIT_NODES, orders_data, all_pairs_shortest_paths, paths)
    if method=='random*':
        if ALWAYS_ACTIVE:
            rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic='double-coverage') 
        else: 
            rewards, assign_status = _weighted_random(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic='double-coverage')
    elif method=='min':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min')
        else: 
            rewards, assign_status = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min')
    elif method=='min*':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic='double-coverage')
        else: 
            rewards, assign_status = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic='double-coverage')
    elif method=='min_diff':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min_diff')
        else: 
            rewards, assign_status = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min_diff', heuristic='double-coverage') # heuristic is absent anyways with min_diff
    elif method=='round-robin':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = round_robin(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths)
        else: 
            rewards, assign_status = _round_robin(INIT_NODES, orders_data, all_pairs_shortest_paths)

    solve_end_time = time.time() 
    solve_time = solve_end_time - solve_start_time
    print(f"Execution time : {solve_time/3600} hrs")
    logger.info(f"Execution time : {solve_time/3600} hrs")
    # breakpoint()


    # EVALUATION # 
    # assert NUM_SERVERS==len(rewards)
    if method=='random' or method=='random*':
        num_unserved = NUM_REQUESTS - np.mean([sum(status) for status in assign_status])
        unused_servers = np.mean([rewardList.count(0.0) for rewardList in rewards])
        gini_idx = np.mean([gini_index(rewardList) for rewardList in rewards])
        avg_dist = np.mean([get_avg_distance(rewardList) for rewardList in rewards])
        print(f'MINIMUM REWARD: {min([min(rlist) for rlist in rewards])}')
        print(f'MAXIMUM REWARD: {max([max(rlist) for rlist in rewards])}')
        logger.info(f'MINIMUM REWARD: {min([min(rlist) for rlist in rewards])}')
        logger.info(f'MAXIMUM REWARD: {max([max(rlist) for rlist in rewards])}')
        logger.info(f'REWARDS: {[rewardList for rewardList in rewards]}')
    else: 
        num_unserved = NUM_REQUESTS-sum(assign_status)
        unused_servers = rewards.count(0.0)
        gini_idx = gini_index(rewards)
        avg_dist = get_avg_distance(rewards)
        print(f'MINIMUM REWARD: {min(rewards)}')
        print(f'MAXIMUM REWARD: {max(rewards)}')
        logger.info(f'MINIMUM REWARD: {min(rewards)}')
        logger.info(f'MAXIMUM REWARD: {max(rewards)}')
        logger.info(f'REWARDS: {rewards}')

    print(f'{num_unserved} out of total {NUM_REQUESTS} requests remain unserved')
    print(f'{unused_servers} out of {NUM_SERVERS} servers remain unused (have 0 rewards).')
    print(f'gini: {gini_idx}')
    print(f'cost: {avg_dist}')

    if not ALWAYS_ACTIVE:
        if method=='random' or method=='random*':
            # compute per_ts_rewards == rewards/total-active-time
            per_ts_gini = 0
            for rlist in rewards:
                per_ts_rewards = copy.deepcopy(rlist)
                for vidx, vid in enumerate(active_drivers.keys()):
                    num_active_ts = 0
                    for ts in range(t_hrs*3600+1):
                        num_active_ts += active_drivers[vid][ts]
                    num_active_ts = max(MAX_DELIVERY_TIME, num_active_ts)
                    assert num_active_ts>0, 'Inactive driver being considered' 
                    per_ts_rewards[vidx] /= num_active_ts
                per_ts_gini += gini_index(list(per_ts_rewards))
            per_ts_gini /= len(rewards)
            print(f'ts_gini: {per_ts_gini}')
        else:
            # compute per_ts_rewards == rewards/total-active-time
            per_ts_rewards = copy.deepcopy(rewards)
            for vidx, vid in enumerate(active_drivers.keys()):
                num_active_ts = 0
                for ts in range(t_hrs*3600+1):
                    num_active_ts += active_drivers[vid][ts]
                num_active_ts = max(MAX_DELIVERY_TIME, num_active_ts)
                assert num_active_ts>0, 'Inactive driver being considered' 
                per_ts_rewards[vidx] /= num_active_ts
            per_ts_gini = gini_index(list(per_ts_rewards))
            print(f'ts_gini: {per_ts_gini}')
            

    # logger.info(f'MINIMUM REWARD: {min(rewards)}')
    # logger.info(f'MAXIMUM REWARD: {max(rewards)}')
    logger.info(f'{num_unserved} out of total {NUM_REQUESTS} requests remain unserved')
    logger.info(f'{unused_servers} out of {NUM_SERVERS} servers remain unused (have 0 rewards).')
    logger.info(f'rewards:\n{rewards}')
    logger.info(f'gini: {gini_idx}')
    logger.info(f'cost: {avg_dist}')
    logger.info(f'assign_status:\n{assign_status}')

    # breakpoint()
    # return ALL_NODES, NUM_NODES, NUM_REQUESTS, NUM_TIMESTAMPS, NUM_SERVERS, UB, ODD, INIT
# ''' 

'''
# main() for SYNTHETIC data
if __name__=='__main__':
    # get program inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['X'], default='X', 
                        type=str, required=False, help='City name')
    parser.add_argument('--num_timesteps', choices=[100,200,500,1000,2000],
                        type=int, required=True)
    parser.add_argument('--num_nodes', choices=[50,100,500,1000], 
                        type=int, required=True, help='# nodes in metric space')
    parser.add_argument('--num_requests', choices=[10,30,50,100,250,500,1000],
                        type=int, required=True)
    parser.add_argument('--num_servers', choices=[5,10,20,25,30,50,100,150,200],
                        type=int, required=True)
    parser.add_argument('--edge_prob', choices=[0.1,0.2,0.5,0.6,0.7,0.9], default=0.5, 
                        type=float, required=False)
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='dist',
                        type=str, required=False, help='Weight variable for road network')
    parser.add_argument('--only_last_mile', choices=[0, 1], default=False, 
                        type=int, required=True, help='Consider only last-mile rewards (or costs)')
    parser.add_argument('--method', choices=['random', 'random*', 'round-robin', 'min', 'min*', 'min_diff'], default='random', 
                        type=str, required=False, help='Type of Objective function')
    parser.add_argument('--init', choices=[0,1], default=0,
                        type=int, required=False, help='Consider the initial locations of the servers')
    parser.add_argument('--batch_size', choices=[1,2,3,4,5,6,7,8,9,10], default=1, 
                    type=int, required=False, help='Average batch-/cluster-size when "batching" is done')
    parser.add_argument('--test_only_availability', choices=[0,1],
                        type=int, required=True)
    args = parser.parse_args()

    # create input variables
    city = args.city   
    NUM_TIMESTEPS = args.num_timesteps
    NUM_NODES = args.num_nodes
    NUM_REQUESTS = args.num_requests 
    NUM_SERVERS = args.num_servers
    edge_prob = args.edge_prob
    weight_var = args.weight_var
    only_last_mile = args.only_last_mile
    method = args.method
    INIT = args.init
    batch_size = args.batch_size 
    ONLY_AVAIL = args.test_only_availability
    # finally, 
    if INIT==1: UB =  1 
    
    # metadata
    data_path = '/home/daman/Desktop/k_server/code/data/'
    logs_path = os.path.join(data_path, city, 'logs')
    ints_path = os.path.join(data_path, city, 'drivers/de_intervals')
    drivers_init_path = os.path.join(data_path, f'{city}/drivers/init_nodes_{NUM_REQUESTS}_{NUM_NODES}.csv')

    # LOAD DATA
    print("Loading data ...") 
    orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/orders_{NUM_REQUESTS}_{NUM_NODES}_{NUM_TIMESTEPS}.csv'))
    apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/apsp_dist_{NUM_NODES}_p{edge_prob}.pkl'))
    apsp_time = depicklify(os.path.join(data_path, f'{city}/map/apsp_time_{NUM_NODES}_p{edge_prob}.pkl'))
    paths_dist = depicklify(os.path.join(data_path, f'{city}/map/apsp_dist_lists_{NUM_NODES}_p{edge_prob}.pkl'))
    paths_time = depicklify(os.path.join(data_path, f'{city}/map/apsp_time_lists_{NUM_NODES}_p{edge_prob}.pkl'))
    road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/metric_space_{NUM_NODES}_p{edge_prob}.csv'))
    # apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/mst_apsp_dist_{NUM_NODES}_p{edge_prob}.pkl'))
    # apsp_time = depicklify(os.path.join(data_path, f'{city}/map/mst_apsp_time_{NUM_NODES}_p{edge_prob}.pkl'))
    # paths_dist = depicklify(os.path.join(data_path, f'{city}/map/mst_apsp_dist_lists_{NUM_NODES}_p{edge_prob}.pkl'))
    # paths_time = depicklify(os.path.join(data_path, f'{city}/map/mst_apsp_time_lists_{NUM_NODES}_p{edge_prob}.pkl'))
    # road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/mst_{NUM_NODES}_p{edge_prob}.csv'))
    all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist, apsp_time)
    paths = parse_all_pair_shortest_paths(paths_dist, paths_time)
    ALL_NODES = range(1, NUM_NODES+1)


    # NUM_SERVERS = NUM_REQUESTS//2
    requestNodes = list(np.unique(orders_data.rest_node)) 

    ###### INPUT SUMMARY BEGINS ###########
    if INIT: # better for real-data
        curr_rest_nodes = list(orders_data['rest_node'].unique())
        INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)
    # if INIT: # better for synthetic data
    #     INIT_NODES = get_init_nodes(drivers_init_path)
    else:
        # dummy node with 0-distance from all nodes
        INIT_NODES = ['src']*NUM_SERVERS
        all_pairs_shortest_paths['src'] = {node:{'dist':0, 'time':0} for node in ALL_NODES}
        # for node in ORIG_ALL_NODES:
        #     all_pairs_shortest_paths[node]['src'] = {'dist':0, 'time':0}

    max_deliver_timestamp = np.max(orders_data['deliver_ts'].values)
    NUM_TIMESTAMPS = max_deliver_timestamp 
    MAX_FPT = np.max(orders_data.prep_time.values)
    MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
    MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
    MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
    MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

    print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
    print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
    print(f"Number of requests (or orders): {NUM_REQUESTS}")
    print(f"Number of servers (or drivers): {NUM_SERVERS}")
    # print(f"Total number of nodes in the LP or Flow network: {NUM_VARS}")
    # # There exist 97282 edges in the road network
    
    # GET LOGGER:
    global logger
    log_filename = ''
    if INIT: log_filename = f"online_INIT_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.log" 
    else: log_filename = f"{edge_prob}_{method}{only_last_mile}_online_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.log"
    logger = get_logger(logs_path, log_filename) 

    logger.info(f"NUM_NODES : {NUM_NODES}")
    logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
    logger.info(f"NUM_SERVERS : {NUM_SERVERS}")
    logger.info(f"NUM_TIMESTAMPS : {NUM_TIMESTAMPS}")
    ########## INPUT SUMMARY ENDS #################
    # breakpoint()
    
    # SOLVING #
    solve_start_time = time.time()  
    print("Online solution started ...")
    if method=='random':
        rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic=None)
    elif method=='random*':
        rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic='double-coverage')
    elif method=='min':
        rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic=None)
    elif method=='min*':
        rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic='double-coverage')
    elif method=='min_diff':
        rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min_diff', heuristic=None)
    elif method=='round-robin':
        rewards, assign_status = round_robin(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths)

    solve_end_time = time.time() 
    solve_time = solve_end_time - solve_start_time
    print(f"Execution time : {solve_time/3600} hrs")
    logger.info(f"Execution time : {solve_time/3600} hrs")
    # breakpoint()

    # EVALUATION # 
    if method=='random' or method=='random*':
        num_unserved = NUM_REQUESTS - np.mean([sum(status) for status in assign_status])
        unused_servers = np.mean([rewardList.count(0.0) for rewardList in rewards])
        gini_idx = np.mean([gini_index(rewardList) for rewardList in rewards])
        avg_dist = np.mean([get_avg_distance(rewardList) for rewardList in rewards])
        print(f'MINIMUM REWARD: {min([min(rlist) for rlist in rewards])}')
        print(f'MAXIMUM REWARD: {max([max(rlist) for rlist in rewards])}')
        logger.info(f'MINIMUM REWARD: {min([min(rlist) for rlist in rewards])}')
        logger.info(f'MAXIMUM REWARD: {max([max(rlist) for rlist in rewards])}')
    else: 
        num_unserved = NUM_REQUESTS-sum(assign_status)
        unused_servers = rewards.count(0.0)
        gini_idx = gini_index(rewards)
        avg_dist = get_avg_distance(rewards)
        print(f'MINIMUM REWARD: {min(rewards)}')
        print(f'MAXIMUM REWARD: {max(rewards)}')
        logger.info(f'MINIMUM REWARD: {min(rewards)}')
        logger.info(f'MAXIMUM REWARD: {max(rewards)}')
        
    print(f'{num_unserved} out of total {NUM_REQUESTS} requests remain unserved')
    print(f'{unused_servers} out of {NUM_SERVERS} servers remain unused (have 0 rewards).')
    print(f'gini: {gini_idx}')
    print(f'cost: {avg_dist}') 

    logger.info(f'MINIMUM REWARD: {min(rewards)}')
    logger.info(f'MAXIMUM REWARD: {max(rewards)}')
    logger.info(f'{num_unserved} out of total {NUM_REQUESTS} requests remain unserved')
    logger.info(f'{unused_servers} out of {NUM_SERVERS} servers remain unused (have 0 rewards).')
    logger.info(f'rewards:\n{rewards}')
    logger.info(f'gini: {gini_idx}')
    logger.info(f'cost: {avg_dist}')
    logger.info(f'assign_status:\n{assign_status}')
    # breakpoint()
'''