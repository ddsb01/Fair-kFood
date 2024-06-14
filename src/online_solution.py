import warnings
warnings.filterwarnings('ignore')

import os 
import sys 
import copy
import glob
import lzma
import time
import bisect
import pickle 
import random
import argparse
import itertools
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from copy import deepcopy
from collections import defaultdict

import networkx as nx 
from bisect import bisect_left, bisect_right

import gc
import logging
from typing import List, Dict

from utils import *
from offline_solution import * 
from eval import calculate_infeasibility, get_server_rewards, get_optimal_cost

random.seed(1234567)
INF = 1e20
eps = 1e-6
EPS_REWARD = 1e-1 # to avoid server reward being 0 when it gets assigned to a request with 0 last mile reward!
# So any server having only a EPS_REWARD reward => The server was at 'src' so the first_mile_reward was 0 
#                                               => that request's r_node and c_node were same so the last_mile_reward was 0


def get_active_servers(intervals_datapath, t):
    ''' 
    This is applicable only to Swiggy data; NOT applicable to synthetic data!
    '''
    # os.remove(all_drivers_path)
    # os.remove(active_drivers_path)

    all_drivers_path = os.path.join(_intervals_datapath, f'all_drivers.pkl')
    active_drivers_path = os.path.join(_intervals_datapath, f'active_drivers_{t}.pkl')
    init_path = os.path.join(_intervals_datapath, f'init_nodes{t}.pkl') ######

    init_nodes = [] ######

    if os.path.exists(all_drivers_path) and os.path.exists(active_drivers_path) and os.path.exists(init_path):
        print("Reading from", all_drivers_path)
        all_drivers = depicklify(all_drivers_path)
        print("Reading from", active_drivers_path)
        active_drivers = depicklify(active_drivers_path)
        print("Reading from", init_path)
        init_nodes = depicklify(init_path)
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
                start_node = int(file_df.iloc[i*2].values[0].split()[1]) ######
                end_ts = int(file_df.iloc[(i*2)+1].values[0].split()[0])
                if not taken: init_nodes.append(start_node) ######
                for j in range(start_ts, end_ts+1):
                    if not taken and j<=t*3600: 
                        final_drivers.append(vid)
                        taken = 1
                    # exception handling to tackle: "invalid start_ts" or "invalid end_ts"
                    try: all_drivers[vid][j] = 1
                    except: continue
        picklify(all_drivers, all_drivers_path)
        picklify(init_nodes, init_path)

        active_drivers = {vh:all_drivers[vh] for vh in final_drivers}
        picklify(active_drivers, active_drivers_path)
        if t==24: active_drivers = all_drivers
    # return active_drivers
    return active_drivers, init_nodes ######


def micro_resource_util(utility,activity):
    num_active_ts = 0
    num_total_ts = 0
    for vh in utility.keys():
        for ts in utility[vh].keys():
            if activity[vh][ts]==1:
                num_total_ts += 1
                if utility[vh][ts]==0: 
                    num_active_ts += 1
    return num_active_ts/num_total_ts


def get_resource_util(utility,activity):
    macro_util = 0.0
    for vh in utility.keys():
        num_active_ts = 0
        num_total_ts = 0
        for ts in utility[vh].keys():
            if activity[vh][ts]==1:
                num_total_ts += 1
                if utility[vh][ts]==0: 
                    num_active_ts += 1
        curr_util = num_active_ts/num_total_ts
        macro_util += curr_util
    return macro_util/len(utility)


# ----------------------------------------------------------------------------------------

'''
Methods for Food Delivery data:
- Here we always consider real-world work shifts of the drivers.
- The baseline (offline algo) to compare against is "FairFoody".
'''

def _weighted_random(curr_locations, requests_df, apsp_lengths, apsp_paths, heuristic=None):
    ''' 
    When a request 'r' arrives, it is assigned to the 'd-th' driver with probability proportional to probs[d] if he is active
    The probabilites are updated after the assignment of each request 
    post=> choose from the servers which are available and reachable
    '''
    numRuns = 5
    ACTIVE_DRIVERS = get_active_servers(ints_path, t_hrs)[0]
    REWARDS = []
    ASSIGN_STATUS = []
    UTILITY = []

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
    
    for run in range(numRuns):
        seed_val = int(np.random.randint(1,100)*100)
        print("SEED VALUE:", seed_val)
        random.seed(seed_val)
        
        curr_locs = copy.deepcopy(curr_locations)
        vir_locs = copy.deepcopy(curr_locations) # virtual locations for heuristic

        active_drivers = copy.deepcopy(ACTIVE_DRIVERS)
        vids = [int(k) for k in active_drivers.keys()]
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
            for s_idx, did in enumerate(active_drivers.keys()):
                curr_server_node = vir_locs[s_idx] 
                if curr_server_node=='src':
                    sp_dist, sp_time = apsp_lengths[curr_server_node][r_node]['dist'], apsp_lengths[curr_server_node][r_node]['time']
                else:
                    sp_dist, sp_time = apsp_lengths[r_node][curr_server_node]['dist'], apsp_lengths[r_node][curr_server_node]['time']

                if ONLY_AVAIL: test_val = active_drivers[did][pts] # Test "availability"
                else: test_val = active_drivers[did][pts] and sp_time<=fpt # Test "availability" and "reachability" # don't check, just assume that active_driver[did][dts] would be 1 (as is done by FoodMatch and FairFoody)

                if test_val:
                    eligible_drivers.append(did)
                    sub_probs[did] = rewards[did]
            assert len(eligible_drivers)==len(sub_probs), "Incorrect length of probability list"
            
            # assert len(eligible_drivers)>0, f"No eligible servers for the {idx}-th request"
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
                winner = random.choices(eligible_drivers, sub_probs, k=1)[0]
                it = vids.index(int(winner))
                curr_server_node = curr_locs[it] 
                
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
            
                for t in range(pts, dts+1): active_drivers[winner][t] = 0
                
                curr_locs[it] = c_node
                vir_locs[it] = c_node
                assign_status[idx] += 1
                assert assign_status[idx]<=1, 'Assigned more than 1 server per request' 
            else:
                # the chosen server was uneligible to server the request; 
                # hence the request remains unserved
                continue

            # < Double-Coverage heuristic > # [all servers move (virtuall, w/o cost) towards the current request with equal speed until it gets served]
            # <=> here, since movement is discrete node-to-node, all servers move to the request node  
            if heuristic=='double-coverage':
                # move all unobstructed (available) servers towards the nearest request node
                # NOTE: Since the road network does not have uniformaly placed nodes wrt distance or time, therefore the requirement of all idel servers moving with *equal* speeds can't be realized easily
                # Therefore, we move node-by-node, which essentially translates to idle server movements with (slightly? practical?) unequal speeds.  
                for it, vid in enumerate(active_drivers.keys()):
                    if vid!=winner:
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
        UTILITY.append(active_drivers)
    
    REWARDS = [list(r.values()) for r in REWARDS]
    return REWARDS, ASSIGN_STATUS, UTILITY


def _round_robin(curr_locs, requests_df, apsp_data):
    active_servers = get_active_servers(ints_path, t_hrs)[0]
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
            elg_indices = [entry[0] for entry in eligible_drivers]
            elg_idx = bisect.bisect(elg_indices, s_idx) % len(eligible_drivers)
            win_idx, winner = eligible_drivers[elg_idx]

            if winner is None: continue 
            
            curr_server_node = curr_locs[win_idx] 
            first_mile_reward = sp_dist
            last_mile_reward = lmd
            # update server data
            if only_last_mile: rewards[winner] += last_mile_reward 
            else: rewards[winner] += (first_mile_reward + last_mile_reward)
                    
            for t in range(pts, dts+1): active_servers[vid][t] = 0
            curr_locs[win_idx] = c_node
            assign_status[idx] += 1
            assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
            # o/w the request could not be assigned to the s_idx-th server
        else: continue # s_idx = (win_idx+1) % num_servers
        s_idx = (win_idx+1) % num_servers

    return list(rewards.values()), assign_status, active_servers


def _greedy(curr_locs, requests_df, apsp_lengths, apsp_paths, strategy, heuristic=None):
    '''
    strategy:
        'min': assign the order to the active server 'd' with minimum current reward 
               it is a special case of 'weighted_random' with probs[x]=1 if x==d else 0
        'min_diff': assign the order such that the max_reward-min_reward after assigning this order will be minimum
                it is similar to "FairFoody's" bipartite graph weight heuristic

    '''
    active_servers = get_active_servers(ints_path, t_hrs)[0]
    num_servers = len(active_servers)
    rewards = {s:0.0 for s in list(active_servers.keys())}
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    vids = list(active_servers.keys())

    vir_locs = copy.deepcopy(curr_locs)
    if strategy=='min':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            # this is okay for low request loads, but for high-load cases it'd be better to loop over timestepwise rather than requests
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
            moveTime = 0
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
                
                moveTime = sp_time
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
            # <=> here, since movement is discrete node-to-node, all servers move to the request node or almost reach the request node  
            if heuristic=='double-coverage':
                # move towards the current request node
                # NOTE: this can be made way more efficient by pre-computing the nearest restaurant nodes (or hashing the *all* pair shortest paths beforehand) 
                # and parallelizing the tracking of idle server movements using multiprocessing or multithreading since these movements are independent (will need considerable changes in the flow of the code)
                for it, vid in enumerate(active_drivers.keys()):
                    if it==win_idx: continue
                    d_node = vir_locs[it]
                    if d_node!='src' and active_drivers[vid][pts]:
                        nearestRest = requestNodes[np.argmin([apsp_lengths[rest][d_node]['time'] for rest in requestNodes])]
                        path_to_rest = apsp_paths[nearestRest][d_node]['time'] # list of nodes: [r_node, ..., d_node]
                        num_path_nodes = len(path_to_rest)
                        path_times = [apsp_lengths[nearestRest][intNode]['time'] for intNode in path_to_rest]
                        assert path_times[0]==0, 'First node in the path should be the nearestRest node iteself!'
                        times = [path_times[-1]-t for t in path_times[::-1]]
                        steps = max(0, bisect_left(times, moveTime)-1) 
                        vir_locs[it] = path_to_rest[(num_path_nodes-1)-steps]

                
    elif strategy=='min_diff':
        for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
            pts, dts = request.prep_ts, request.deliver_ts
            if pts>=86400: continue
            r_node, c_node = request.rest_node, request.cust_node
            fpt = request.prep_time
            lmd = request.deliver_dist

            active_timesteps = {s:0 for s in list(active_servers.keys())} # active timestepsof each driver till the pts of the current order
            for _vid in active_servers:
                num_active_ts = 0
                for ts in range(pts): 
                    num_active_ts += active_drivers[_vid][ts]
                    num_active_ts = max(request.prep_time+1, num_active_ts) # a weak assumption that the server will serve the assigned request beyond the workshift
            # the above active_timesteps computation can be made efficient by only traversing the timesteps between the previous and current orders' prep_ts
            
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

                vid = vids[it] ## vehicle id of it-th server
                # if ONLY_AVAIL: test_val = (active_servers[vid][pts] and active_servers[vid][dts]) # Test "availability"
                # else: test_val = (active_servers[vid][pts] and active_servers[vid][dts]) and sp_time<=fpt # Test "availability" and "reachability"
                if ONLY_AVAIL: test_val = active_servers[vid][pts] # Test "availability"
                else: test_val = active_servers[vid][pts] and sp_time<=fpt # Test "availability" and "reachability"

                if not test_val: continue
                
                curr_server_node = curr_locs[it] 
                first_mile_reward = sp_dist
                last_mile_reward = lmd

                rewards_copy = copy.deepcopy(rewards)
                if only_last_mile: rewards_copy[vid] += last_mile_reward
                else: rewards_copy[vid] += (first_mile_reward + last_mile_reward)
                # just because the first_mile_reward can be different for different servers, min_diff is different from min
                
                # normalize the server rewards by respective num_active_timesteps 
                for _vid in rewards_copy:
                    rewards_copy[_vid] /= active_timesteps[_vid]

                # pat_rewards = copy.deepcopy(rewards_copy) # per-active-timestep rewards
                # for _vid in pat_rewards:
                #     num_active_ts = 0 
                #     for ts in range(pts):
                #         num_active_ts += active_drivers[_vid][ts] 
                #     num_active_ts = max(request.prep_time+1, num_active_ts) # a weak assumption that the server will serve the assigned request beyond the workshift
                #     pat_rewards[_vid] /= num_active_ts 

                # diff = max(pat_rewards.values()) - min(pat_rewards.values())
                
                diff = max(rewards_copy.values()) - min(rewards_copy.values())
                if diff<min_diff:
                    min_diff = diff 
                    winner = it
                    # winner_reward = rewards_copy[vid]
                    winner_reward = rewards_copy[vid]*active_timesteps[vid]
                    assign_status[idx] = 1

            # update changes
            if assign_status[idx]: # => winner!=-1
                win_id = vids[winner]
                rewards[win_id] = winner_reward
                for t in range(pts, dts+1):
                    active_servers[win_id][t] = 0
                curr_locs[winner] = c_node

    return list(rewards.values()), assign_status, active_servers
    
# ----------------------------------------------------------------------------------------

''' 
Methods for Food Delivery data (without workshifts), Synthetic data, and Quick Commerce data:
- Assumption: All servers remain active for the entire duration (t_hrs); that is why gini index computation is not separately time-normalized (bcz it's time-normalized by default)
- The baseline (offline algo) to compare against is the "Flow-based-LP"
'''

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
            arr = [(1/val+eps) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
            max_val = max(arr)
            arr = [(1/2**(max_val-val)) for val in arr] # INVERT (since higher current reward => lower-probability of getting next request )
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
            plcd_ts, pts, dts = request.placed_ts, request.prep_ts, request.deliver_ts
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

                if GAMMA*fpt<=fpt+apsp_lengths[r_node][c_node]['time']: fpt = GAMMA*fpt

                if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
                else: test_val = (active[d][pts] and sp_time<=fpt) 

                if test_val:
                    eligible_drivers.append(d)
                    sub_probs.append(rewards[d])
            assert len(eligible_drivers)==len(sub_probs), "Incorrect length of probability list"
            
            winner = None
            if len(eligible_drivers)>0:
                # select a server proportional to it's weight; 
                sub_probs = update_normalize_alt(sub_probs) # update_normalize_alt(sub_probs)
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
                # for t in range(plcd_ts, dts+1): active[it][t] = idx 
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
                        # NOTE: it's possible that the travel time between next node to d_node on the path to nearestRest might be greater than the 
                        # travel time between d_node and r_node. It might even be lesser. We assume that d_node to next-node travel is always possible 
                        # within d_node-r_node travel time. Ideally next-node should be chosen such that d_node to next-node distance is equal to the 
                        # d_node to r_node distance      

        # accumulate results
        REWARDS.append(rewards)
        ASSIGN_STATUS.append(assign_status) 
    return REWARDS, ASSIGN_STATUS


def round_robin(num_servers, curr_locs, requests_df, apsp_data):
    rewards = [0.0]*num_servers
    num_requests = requests_df.shape[0]
    assign_status = [0]*num_requests
    active = [{ts:1 for ts in range(NUM_TIMESTAMPS+1)} for d in range(num_servers)]

    s_idx = 0 # server index of server to which the current request should be assigned
    for idx, request in tqdm(requests_df.iterrows(), total=num_requests):
        plcd_ts, pts, dts = request.placed_ts, request.prep_ts, request.deliver_ts
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

            # testing happens at 'placed_ts' of the 'request'
            if ONLY_AVAIL: test_val = active[d][pts] # Test "availability"
            else: test_val = (active[d][pts] and sp_time<=fpt)

            if test_val: eligible_drivers.append(d)

        winner = None
        if len(eligible_drivers)>0:
            winner_idx = bisect.bisect_left(eligible_drivers,s_idx) % len(eligible_drivers)
            winner = eligible_drivers[winner_idx]
            # breakpoint()

            if winner is None: continue 

            # if winner is not None:
            curr_server_node = curr_locs[winner]
            if curr_server_node=='src':
                sp_dist, sp_time = apsp_data[curr_server_node][r_node]['dist'], apsp_data[curr_server_node][r_node]['time']
            else:
                sp_dist, sp_time = apsp_data[r_node][curr_server_node]['dist'], apsp_data[r_node][curr_server_node]['time']

            # compute changes
            first_mile_reward = sp_dist
            last_mile_reward = lmd
            # update server data
            if only_last_mile: rewards[winner] += max(last_mile_reward, EPS_REWARD) 
            else: rewards[winner] += (first_mile_reward + max(last_mile_reward, EPS_REWARD)) # the server reward shouldn't be 0 because of request having 0 reward!!
            
            for t in range(pts, dts+1): active[winner][t] = 0
            curr_locs[winner] = c_node
            assign_status[idx] += 1
            assert assign_status[idx]<=1, 'Assigned more than 1 server per request'
        else: continue
            # the chosen server was uneligible to server the request; 
            # hence the request remains unserved
        # assert winner!=idx, f"Request is {idx} BUT Server is {winner}"
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
            # <=> here, since movement is discrete node-to-node, all servers move to the request node (or a node close to the request node)
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
                # if diff<=min_diff: # this condition systematically avoids the servers at 'src'!!
                if diff<min_diff:
                    min_diff = diff 
                    winner = it 
                    winner_reward = rewards_copy[it]
                    assign_status[idx] = 1
            
            
            assert winner!=-1 and winner_reward!=-1, 'Incorrect computation of winner'
            # update changes
            if assign_status[idx]: # => winner!=-1
                rewards[winner] = winner_reward
                for t in range(pts, dts+1): active[winner][t] = 0
                curr_locs[winner] = c_node
        
    return rewards, assign_status


# ----------------------------------------------------------------------------------------

if __name__=='__main__':
     # get program inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['A', 'B', 'C'], default='A', 
                        type=str, required=True, help='City name')
    parser.add_argument('--day', choices=[x for x in range(10)], default=1,
                        type=int, required=True, help='Day index')
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='dist',
                        type=str, required=False, help='Weight variable for road network')
    parser.add_argument('--only_last_mile', choices=[0, 1], default=0, 
                        type=int, required=False, help='Consider only last-mile rewards (or costs)')
    parser.add_argument('--method', choices=['random', 'random*', 'min', 'min*', 'min_diff', 'round-robin'], default='random', 
                        type=str, required=True, help='Type of Objective function')
    parser.add_argument('--test_only_availability', choices=[0,1], default=0,
                        type=int, required=False, help='Test only availability and not reachability.')
    parser.add_argument('--batch_size', choices=[1,2,3,4,5,6,7,8,9,10], default=1, 
                    type=int, required=False, help='Average batch-/cluster-size when "batching" is done') # currently batching is not supported
    
    ## food delivery specific args:
    parser.add_argument('--start', choices=[-1]+[x for x in range(25)], default=0, 
                        type=int, required=False, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--end', choices=[-1]+[x for x in range(25)], default=8, 
                        type=int, required=False, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--timestep', default=1,
                        type=int, required=False, help='The entire consists of "(86400//timestep)" units of time')
    parser.add_argument('--odd_dataset', choices=[1, 2], default=1,
                        type=int, required=False, help='whether to take odd timestamped data <=> slotted data with timestep=2 (choose 2 as input) AND distinct prep_tss !')
    parser.add_argument('--init', choices=[0,1], default=0,
                        type=int, required=False, help='Consider the initial locations of the servers')
    parser.add_argument('--always_active', choices=[0,1], default=1,
                        type=int, required=False, help='All servers are always active (no consideration of work-shifts <=> server unavailability can only be caused by the service of an order).')
    
    ## only synthetic data specific args:

    ## only quick commerce data specific args: None

    args = parser.parse_args()
    city = args.city # this variable decides the dataset
    
    data_path = './data/'

    if city=='A':
        '''
        FOOD DELIVERY DATASET
        '''
        day = args.day 
        start = args.start
        t_hrs = args.end
        weight_var = args.weight_var
        only_last_mile = args.only_last_mile
        method = args.method
        timestep = args.timestep 
        ODD = args.odd_dataset
        INIT = args.init
        batch_size = args.batch_size 
        ALWAYS_ACTIVE = args.always_active
        ONLY_AVAIL = args.test_only_availability
        GAMMA = 1

        # metadata
        data_path = '/home/daman/Desktop/k_server/code/data/'
        logs_path = os.path.join(data_path, city, 'logs')
        ints_path = os.path.join(data_path, city, f'drivers/{day}/de_intervals')
        _intervals_datapath = os.path.join(data_path, city, f'drivers/{day}')
        drivers_init_path = os.path.join(data_path, city, "drivers/1/driver_init_nodes.csv")
        name2id = {'A':10, 'B':1, 'C':4}
        id2name = {v:k for k,v in name2id.items()}

        # LOAD DATA
        print("Loading data ...") 
        if timestep==1:
            orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/_final_orders.csv')) # non-distinct timesteps # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        else:
            # this violates distinct prep_ts assumption
            orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_slotted_{timestep}.csv'))
        
        if ODD==2:
            orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_odd_timestamps.csv'))
        
        if t_hrs==13 or t_hrs==14:
            apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_dist_t=15.pkl'))
            apsp_time = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_time_t=15.pkl'))   
            paths_dist = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_lists_dist_t=15.pkl'))
            paths_time = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_lists_time_t=15.pkl'))
        else:
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
        orders_data = orders_data[(orders_data.placed_time>=start*3600) & (orders_data.placed_time<=t_hrs*3600)].reset_index()
        requestNodes = list(np.unique(orders_data.rest_node)) 
        requestNodesSet = set(requestNodes)
        restNode2Id =  {n:[] for n in requestNodes} # there may be multiple restIds corresponding to the same node
        for rowIdx,row in orders_data.iterrows():
            node, rest = row.rest_node, row.rest_id
            restNode2Id[node].append(rest)
        for k,v in restNode2Id.items(): restNode2Id[k] = np.unique(restNode2Id[k]) 
        rest_load = pd.read_csv(os.path.join(data_path,city, f'orders/{day}/rest_load.csv')) 

        all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist=apsp_dist, apsp_time=apsp_time)
        paths = parse_all_pair_shortest_paths(apsp_dist=paths_dist, apsp_time=paths_time)

        # Create required variables and data 
        ALL_NODES = idx2node.idx.values # or np.arange(idx2node.shape[0])
        NUM_NODES = idx2node.shape[0]  # or len(ALL_NODES)
        ORIG_ALL_NODES = ALL_NODES
        ORIG_NUM_NODES = NUM_NODES
        
        ###### INPUT SUMMARY BEGINS ###########
        NUM_REQUESTS = orders_data.shape[0] 
        total_num_rests = orig_orders_data['rest_node'].unique().shape[0] 
        curr_num_rests = orders_data['rest_node'].unique().shape[0] # number of active restaurants in chosen subset of data
        active_drivers, INIT_NODES = get_active_servers(ints_path, t_hrs) ##### in thesis
        num_active_drivers = len(active_drivers) 
        NUM_SERVERS = min(num_active_drivers, NUM_REQUESTS)
        
        if INIT:
            # dummy node with 0-distance from all nodes
            INIT_NODES = ['src']*NUM_SERVERS # in paper
            all_pairs_shortest_paths['src'] = {node:{'dist':0, 'time':0} for node in range(ORIG_NUM_NODES)} # infinite radius but there is some cost of moving from src node to request node
            # all_pairs_shortest_paths['src'] = {node:{'dist':498.09, 'time':19.92} for node in range(ORIG_NUM_NODES)} # infinite radius but there is some cost of moving from src node to request node % assuming a speed of 25m/s

        min_deliver_timestep = np.min(orders_data['placed_ts'].values)
        max_deliver_timestamp = np.max(orders_data['deliver_ts'].values)
        NUM_TIMESTAMPS = max_deliver_timestamp - min_deliver_timestep + 1 # time-step = 1 second # 1 day = 86400 seconds
        MAX_FPT = np.max(orders_data.prep_time.values)
        MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
        MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
        MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
        MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

        print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
        print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
        print(f"Number of requests (or orders): {NUM_REQUESTS}")
        print(f"Number of servers (or drivers): {NUM_SERVERS}")
        
        # GET LOGGER:
        # global logger
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

    elif city=='X':
        '''
        SYNTHETIC DATASET
        '''
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
        GAMMA = 1.0
        start = 0
        end = NUM_TIMESTEPS*3600
        ALWAYS_ACTIVE = 1

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
        all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist, apsp_time)
        paths = parse_all_pair_shortest_paths(paths_dist, paths_time)
        ALL_NODES = range(1, NUM_NODES+1)
        requestNodes = list(np.unique(orders_data.rest_node)) 

        ###### INPUT SUMMARY BEGINS ###########
        if INIT: # (will require averaging over multiple sets of INIT_NODES selections)
            curr_rest_nodes = list(orders_data['rest_node'].unique())
            INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)
        else:
            # dummy node with 0-distance from all nodes
            INIT_NODES = ['src']*NUM_SERVERS
            all_pairs_shortest_paths['src'] = {node:{'dist':0, 'time':0} for node in ALL_NODES}

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

    elif city=='Q':
        '''
        QUICK COMMERCE DATASET
        '''
        day = args.day
        NUM_TIMESTAMPS = 5000 #args.num_hours*3600
        weight_var = args.weight_var
        only_last_mile = args.only_last_mile
        method = args.method
        timestep = args.timestep 
        INIT = args.init
        ONLY_AVAIL = args.test_only_availability
        GAMMA = 1
        # metadata
        data_path = '/home/daman/Desktop/k_server/code/data/'
        logs_path = os.path.join(data_path, city, 'logs')
        ALWAYS_ACTIVE = 1

        # LOAD DATA
        print("Loading data ...") 
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/_orders_{day}.csv'))
        nodes = pd.read_csv(os.path.join(data_path, f'{city}/map/_nodes.csv'))
        rests = pd.read_csv(os.path.join(data_path, f'{city}/orders/_rests.csv'))
        requestNodes = rests.rest_node.values
        apsp = depicklify(os.path.join(data_path, f'{city}/map/apsp.pkl'))
        paths = depicklify(os.path.join(data_path, f"{city}/map/paths.pkl"))
        NUM_NODES = nodes.shape[0]
        NUM_RESTS = rests.shape[0]
        NUM_REQUESTS = orders_data.shape[0]
        ALL_NODES = range(1,NUM_NODES+1)

        NUM_SERVERS = 200 #NUM_REQUESTS//10
        INIT_NODES = ['src']*NUM_SERVERS
        # breakpoint()
        ## INPUT SUMMARY BEGINS ##
        max_deliver_timestamp = np.max(orders_data.deliver_ts.values)
        MAX_FPT = np.max(orders_data.prep_time.values)
        MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
        MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
        MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
        MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

        print(f"# Points in the metric space (or #Nodes in road network):{NUM_NODES}")
        print(f"# Total timestamps possible: {NUM_TIMESTAMPS}")
        print(f"Number of requests (or orders): {NUM_REQUESTS}")
        print(f"Number of servers (or drivers): {NUM_SERVERS}")

        # GET LOGGER:
        global logger
        log_filename = ''
        if INIT: log_filename = f"online_INIT_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.log" 
        else: log_filename = f"{method}{only_last_mile}_online_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.log"
        logger = get_logger(logs_path, log_filename) 

        logger.info(f"NUM_NODES : {NUM_NODES}")
        logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
        logger.info(f"NUM_SERVERS : {NUM_SERVERS}")
        logger.info(f"NUM_TIMESTAMPS : {NUM_TIMESTAMPS}")
        ########## INPUT SUMMARY ENDS #################

    else:
        assert False, "Invalid selection of Dataset!"


    # SOLVING #
    solve_start_time = time.time()
    print("Online solution started ...")
    if method=='random':
        if ALWAYS_ACTIVE:
            rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths) 
        else: 
            rewards, assign_status, utility = _weighted_random(INIT_NODES, orders_data, all_pairs_shortest_paths, paths)
    if method=='random*':
        if ALWAYS_ACTIVE:
            rewards, assign_status = weighted_random(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic='double-coverage') 
        else: 
            rewards, assign_status, utility = _weighted_random(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, heuristic='double-coverage')
    elif method=='min':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min')
        else: 
            rewards, assign_status, utility = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min')
    elif method=='min*':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic='double-coverage')
        else: 
            rewards, assign_status, utility = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min', heuristic='double-coverage')
    elif method=='min_diff':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = greedy(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min_diff')
        else: 
            rewards, assign_status, utility = _greedy(INIT_NODES, orders_data, all_pairs_shortest_paths, paths, strategy='min_diff', heuristic='double-coverage') # heuristic is absent anyways with min_diff
    elif method=='round-robin':
        if ALWAYS_ACTIVE: 
            rewards, assign_status = round_robin(NUM_SERVERS, INIT_NODES, orders_data, all_pairs_shortest_paths)
        else: 
            rewards, assign_status, utility = _round_robin(INIT_NODES, orders_data, all_pairs_shortest_paths)

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
        # breakpoint()
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

    logger.info(f'{num_unserved} out of total {NUM_REQUESTS} requests remain unserved')
    logger.info(f'{unused_servers} out of {NUM_SERVERS} servers remain unused (have 0 rewards).')
    logger.info(f'rewards:\n{rewards}')
    logger.info(f'gini: {gini_idx}')
    logger.info(f'cost: {avg_dist}')
    logger.info(f'assign_status:\n{assign_status}')


