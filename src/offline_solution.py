import warnings
warnings.filterwarnings('ignore')

import os 
import sys 
import copy
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
from collections import defaultdict

import gurobipy as gb 
from gurobipy import GRB 

import networkx as nx 

import gc
import logging

from utils import *
from eval import *

random.seed(1234567)


def parse_requests(requests_data):
    ''' 
    Converts the orders data into the required "node" format of the flow network.
    '''
    print("Parsing requests data ...")
    # Store v_{br_i}'s, v_{er_i}'s, and v_{tr_i}'s for all requests r_i; i \in [NUM_REQUESTS]
    begin_nodes = []    # (rest_node, placed_timestamp) 
    end_nodes = []      # (rest_node, food_prep_timestamp)
    terminal_nodes = [] # (cust_node, deliver_timestamp) 
    requests = []       # (v_br, v_er, v_tr)

    for idx, request in tqdm(requests_data.iterrows(), total=requests_data.shape[0]):
        # gather data
        v_br = (request.rest_node, request.placed_ts)
        v_er = (request.rest_node, request.prep_ts)
        v_tr = (request.cust_node, request.deliver_ts)
        request_ds = (v_br, v_er, v_tr) 
        # store data
        begin_nodes.append(v_br) 
        end_nodes.append(v_er)
        terminal_nodes.append(v_tr)
        requests.append(request_ds)

    return begin_nodes, end_nodes, terminal_nodes, requests 


def parse_all_pair_shortest_paths(apsp_dist, apsp_time):   
    ''' 
    Merges the shortest path data calculated based on "time" and "dist" attributes.
    '''
    print("Parsing all-pairs shortest path data ...") 
    outer_keys = list(apsp_dist.keys()) # rest_nodes active within t_hrs
    inner_keys = list(apsp_dist[outer_keys[0]].keys()) # all_nodes in road_network~40K

    print(f"Total number of nodes : {len(inner_keys)}")
    print(f"Number of restaurants active within the given duration : {len(outer_keys)}")

    all_pairs_shortest_paths = {
        node_1: {
            node_2: {'dist':0, 'time':0} for node_2 in inner_keys
        } for node_1 in outer_keys
    }
    
    for node_1 in tqdm(outer_keys, total=len(outer_keys)):
        for node_2 in inner_keys:
            all_pairs_shortest_paths[node_1][node_2]['dist'] = apsp_dist[node_1][node_2]
            all_pairs_shortest_paths[node_1][node_2]['time'] = apsp_time[node_1][node_2]
    
    return all_pairs_shortest_paths


def get_active_servers(intervals_datapath, t):
    ''' 
    This is applicable only to Swiggy data; NOT applicable to synthetic data!
    '''
    # os.remove(all_drivers_path)
    # os.remove(active_drivers_path)

    all_drivers_path = os.path.join(_intervals_path, f'all_drivers.pkl')
    active_drivers_path = os.path.join(_intervals_path, f'active_drivers_{t}.pkl')

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
        # picklify(all_drivers, all_drivers_path)

        active_drivers = {vh:all_drivers[vh] for vh in final_drivers}
        # picklify(active_drivers, active_drivers_path)
        if t==24: active_drivers = all_drivers
    return len(active_drivers)

# ----------------------------------------------------------

def construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var):
    ''' 
    We maintain separate flow-variables for each server.
    "timestamps" \in {1,2,...,NUM_TIMESTAMPS}
    '''
    # print("Adding Variables ...")
    num_total_vars = 0

    all_in_out_vars = []
    all_source_vars = []
    all_sink_vars = []
    all_self_vars = []    
    all_end_terminal_vars, end_terminal_costs = [], []
    all_into_end_node_vars, into_end_node_costs = [], []
    infeasibility_vars = []
    min_reward_var = None
    
    for server_idx in tqdm(range(NUM_SERVERS), total=NUM_SERVERS):
        source_vars = []
        sink_vars = []
        self_vars = []
        end_terminal_vars = []
        into_end_node_vars = []

        in_out_vars = {} 
        for node_idx in tqdm(ALL_NODES, total=NUM_NODES):
            for timestamp in range(1, NUM_TIMESTAMPS+1, ODD):
                node = (node_idx, timestamp) 
                in_out_vars[node] = {'in':[], 'out':[]} 

        print("Adding Type-1,2,3 variables ...")
        for node_idx in tqdm(ALL_NODES, total=NUM_NODES):
            # ------------------------------------------------------------------------------------
            # [ Type-1 ] : SOURCE-node to (All Nodes, timestamp=1) flow variables
            if INIT==1:
                if node_idx==INIT_NODES[server_idx]:
                    src_var = model.addVar(
                                    lb=1.0, ub=1.0, obj=0.0, 
                                    vtype=GRB.CONTINUOUS,
                                    name=f"{server_idx}_source_to_{node_idx}"
                                    )
                else:
                    src_var = model.addVar(
                                        lb=0.0, ub=1.0, obj=0.0, 
                                        vtype=GRB.CONTINUOUS,
                                        name=f"{server_idx}_source_to_{node_idx}"
                                        )
            else:
                src_var = model.addVar(
                                        lb=0.0, ub=1.0, 
                                        obj=0.0, 
                                        vtype=GRB.CONTINUOUS,
                                        # vtype=GRB.BINARY,
                                        name=f"{server_idx}_source_to_{node_idx}"
                                        )
            source_vars.append(src_var)
            # These edges are the only incoming edges to the nodes at timestamp=0
            in_out_vars[(node_idx, 1)]['in'].append(src_var)
            num_total_vars += 1
            # ------------------------------------------------------------------------------------
            # [ Type-2 ] : (All Nodes, timestamp=len(NUM_TIMESTAMPS)) to SINK-node flow variables
            sink_var = model.addVar(
                                lb=0.0, ub=1.0, 
                                obj=0.0, 
                                vtype=GRB.CONTINUOUS,
                                # vtype=GRB.BINARY,
                                name=f"{server_idx}_sink_to_{node_idx}"
                                )
            sink_vars.append(sink_var)
            ## These edges are the only outgoing edges from the nodes at timestep=NUM_TIMESTAMPS-1
            in_out_vars[(node_idx, NUM_TIMESTAMPS)]['out'].append(sink_var)
            num_total_vars += 1
            # ------------------------------------------------------------------------------------
            # [ Type-3 ] : SELF-node flow variables 
            for timestamp in range(1, NUM_TIMESTAMPS, ODD):
                self_var = model.addVar( 
                                    # lb=0.0, ub=UB,
                                    lb=0.0, ub=1,  
                                    obj=0.0, 
                                    vtype=GRB.CONTINUOUS,
                                    # vtype=GRB.BINARY, 
                                    name=f"{server_idx}_self_{node_idx}_{timestamp}_to_{timestamp+1}"
                                    )
                # self_vars.append(self_var) # not really needed to store
                ## These edges go _out_ from (node_idx, timestamp) 
                in_out_vars[(node_idx, timestamp)]['out'].append(self_var)
                in_out_vars[(node_idx, timestamp+1*ODD)]['in'].append(self_var)

                num_total_vars += 1
        # ------------------------------------------------------------------------------------
        # [ Type-4 ] : End-Terminal flow variables 
        print("Adding Type-4 variables ...")
        for request_idx, (end_node, term_node) in tqdm(enumerate(zip(end_nodes, terminal_nodes))):
            if weight_var=='time':
                prepared_timestamp = end_node[1] 
                delivered_timestamp =  term_node[1]
                cost = int(delivered_timestamp - prepared_timestamp) 
            elif weight_var=='dist':
                cost = int(orders_data.iloc[request_idx].deliver_dist)

            if len(end_terminal_costs) < NUM_REQUESTS:
                end_terminal_costs.append(cost)

            et_var = model.addVar(
                                lb=0.0, ub=UB, 
                                obj=0.0,
                                vtype=GRB.CONTINUOUS,
                                # vtype=GRB.BINARY, 
                                name=f"{server_idx}_end-{end_node}_to_terminal-{term_node}"
                                )
            end_terminal_vars.append(et_var)
            # end-terminal-edges go _out_ from end-node and _in_ to terminal-node
            in_out_vars[end_node]['out'].append(et_var)
            in_out_vars[term_node]['in'].append(et_var)

            num_total_vars += 1
        # ------------------------------------------------------------------------------------
        # [ Type-5 ] : Request-prep-time dependent flow variables 
        print("Adding Type-5 variables ...")
        for request_idx, (begin_node, end_node) in tqdm(enumerate(zip(begin_nodes, end_nodes)), total=NUM_REQUESTS):
            begin_node_idx = begin_node[0]
            end_node_idx = end_node[0]
            assert begin_node_idx==end_node_idx

            placed_timestamp = begin_node[1]
            prepared_timestamp = end_node[1] 
            prep_time = prepared_timestamp - placed_timestamp
            constant_speed = 1 # m/s

            # if weight_var=='dist':
            #     shortest_path_time = shortest_path_time/constant_speed 

            # window of timestamps from which to choose the reachable servers = range(placed_timestamp, prepared_timestamp)
            _end_nodes = set(end_nodes)
            curr_request_vars = []
            curr_request_costs = []

            for _idx, node_idx in enumerate(ALL_NODES):
                try:
                    shortest_path_time = all_pairs_shortest_paths[node_idx][end_node_idx]['time']
                except:
                    shortest_path_time = all_pairs_shortest_paths[end_node_idx][node_idx]['time']
                if timestep>1: shortest_path_time = shortest_path_time//timestep
                if node_idx==end_node_idx: continue # already covered by self_vars
                
                # only one 'var' (hence, one 'cost') will be added per node!
                curr_node_var = None
                curr_node_cost = None
                for ts_idx, timestamp in enumerate(range(prepared_timestamp-1*ODD, placed_timestamp-1*ODD, -1*ODD)):
                    # breakpoint()
                    candidate_end_node = (node_idx, timestamp) 
                    if (shortest_path_time <= prep_time): 
                        # flow-variables are added only for reachable cases
                        ien_var = model.addVar(
                                            lb=0.0, ub=UB, 
                                            obj=0.0,
                                            vtype=GRB.CONTINUOUS,
                                            # vtype=GRB.BINARY, 
                                            name=f"{server_idx}_request-{request_idx}:from-{(node_idx,timestamp)}_to_end-{end_node}"
                                            )
                        if weight_var=='dist':
                            try:
                                cost = all_pairs_shortest_paths[node_idx][end_node_idx]['dist'] 
                            except:
                                cost = all_pairs_shortest_paths[end_node_idx][node_idx]['dist']
                        elif weight_var=='time':
                            scaling_factor = 10 
                            cost = shortest_path_time + scaling_factor*(prepared_timestamp - timestamp) # waiting time 
                        
                        in_out_vars[(node_idx, timestamp)]['out'].append(ien_var)
                        in_out_vars[end_node]['in'].append(ien_var)
                        curr_node_var = ien_var
                        curr_node_cost = cost 
                        num_total_vars += 1
                        break

                if curr_node_var is not None: 
                    curr_request_vars.append(curr_node_var)
                    curr_request_costs.append(curr_node_cost)
            
            assert len(curr_request_vars)<=NUM_NODES # cuz only one variable is added per node 

            into_end_node_vars.append(curr_request_vars)
            if len(into_end_node_costs) < NUM_REQUESTS:
                into_end_node_costs.append(curr_request_costs)


        all_in_out_vars.append(in_out_vars)
        all_source_vars.append(source_vars)
        all_sink_vars.append(sink_vars)
        all_self_vars.append(self_vars)
        all_end_terminal_vars.append(end_terminal_vars)
        all_into_end_node_vars.append(into_end_node_vars)

        del in_out_vars 
        gc.collect()
    # ------------------------------------------------------------------------------------       
    # [ Type-6 ] : (In)Feasibility variables
    print("Adding Type-6 variables ...")
    ## Infeasibiity_variables[i] * MAX_DELIVERY_TIME (or DIST) will be minimized in the objective  
    for request_idx in range(NUM_REQUESTS):
        infeasibility_var = model.addVar(
                                        obj=0.0, # => delegating includsion of objective co-efficient to setObjective() method
                                        vtype=GRB.BINARY,
                                        name=f"unserved_{request_idx}"
                                        )
        infeasibility_vars.append(infeasibility_var)
        num_total_vars += 1  
    # ------------------------------------------------------------------------------------
    # [ Type-7 ] : Minimum Reward (or Earning) vairable 
    print("Adding Type-7 variables ...")
    ## this is the variable that'll be maximized in the maxmin objective
    min_reward_var = model.addVar(
                                lb=0.0, 
                                obj=0.0, 
                                vtype=GRB.CONTINUOUS,
                                # vtype=GRB.BINARY, 
                                name="minimum reward/earning accumulated by any server") # NOT doing obj=1.0 here, 
                                                                           # rather delegating includsion of objective co-efficient to setObjective() method
    num_total_vars += 1


    all_vars_and_costs = {
                            'in_out_vars':all_in_out_vars,
                            'source_vars':all_source_vars, 
                            'sink_vars':all_sink_vars,
                            'self_vars':all_self_vars, 
                            'end_terminal_vars':all_end_terminal_vars,
                            'end_terminal_costs':end_terminal_costs,
                            'into_end_node_vars':all_into_end_node_vars,
                            'into_end_node_costs':into_end_node_costs,
                            'infeasibility_vars':infeasibility_vars,
                            'min_reward_var':min_reward_var
                        }
    print("Variables added!")
    print(f"TOTAL NUMBER OF *VARIABLES* IN THE MODEL : {num_total_vars}")
    logger.info(f"TOTAL NUMBER OF *VARIABLES* IN THE MODEL : {num_total_vars}")
    return all_vars_and_costs

# ----------------------------------------------------------

def construct_and_set_constraints(model, vars_and_costs, only_last_mile=True):
    '''
    For each server, we maintain a separate flow. Hence, need to add some of the constraints separately. 
    '''
    num_total_constrs = 0 

    (   
        in_out_vars,
        source_vars,
        sink_vars,
        self_vars,
        end_terminal_vars,
        end_terminal_costs,
        into_end_node_vars,
        into_end_node_costs,
        infeasibility_vars,
        min_reward_var
    ) = vars_and_costs.values()
    server_rewards = []

    first_mile_costs = into_end_node_costs
    last_mile_costs = end_terminal_costs
    
    print("Adding Type-1,2,3,4 constraints")
    for server_idx in range(NUM_SERVERS):
        # [ Type-1 ] : Source-node out-flow = NUM_SERVERS <=> Source-node out-flow = 1 for each server 
        model.addConstr(gb.quicksum(source_vars[server_idx][node_idx] for node_idx in range(NUM_NODES))==1)
        # model.addConstr(source_vars[server_idx][0]==1) # equivalent to all servers having initial locations at node 0 # WHY does this not work??
        num_total_constrs += 1 

        # [ Type-2 ] : Sink-node in-flow = NUM_SERVERS 
        model.addConstr(gb.quicksum(sink_vars[server_idx][node_idx] for node_idx in range(NUM_NODES))==1)
        num_total_constrs += 1

        # [ Type-3 ] : Flow-conservation
        in_out_data = in_out_vars[server_idx] 
        for node_idx, timestamp in itertools.product(ALL_NODES, range(1, NUM_TIMESTAMPS+1, ODD)):
            curr_node = (node_idx, timestamp) 
            in_nodes = in_out_data[curr_node]['in']
            out_nodes = in_out_data[curr_node]['out']
            assert in_nodes!=[] and out_nodes!=[], "flow-conservation assertion"
            model.addConstr(gb.quicksum(in_nodes)==gb.quicksum(out_nodes))
            num_total_constrs += 1
    

        # [ Type-4 ] : Minimum reward 
        first_mile_vars = into_end_node_vars[server_idx] 
        last_mile_vars = end_terminal_vars[server_idx] 
       
        first_mile_reward = gb.LinExpr()  
        for request_idx in range(NUM_REQUESTS):
            curr_fm_vars = first_mile_vars[request_idx]
            curr_fm_costs = first_mile_costs[request_idx]
            assert len(curr_fm_vars)==len(curr_fm_costs)
            assert len(curr_fm_vars)<=len(ALL_NODES)
            # -----
            for flow_var, reward_value in zip(curr_fm_vars, curr_fm_costs):
                first_mile_reward.addTerms(reward_value, flow_var)
            # -----
        last_mile_reward = gb.quicksum((flow_var * reward_value) for flow_var, reward_value in zip(last_mile_vars, last_mile_costs)) 

        if only_last_mile: curr_server_reward = last_mile_reward
        else: curr_server_reward = first_mile_reward + last_mile_reward

        server_rewards.append(curr_server_reward)

        model.addConstr(min_reward_var <= curr_server_reward) 
        num_total_constrs += 1


    # Net input flow (first-mile-flow) into an end_node should be 1 (or UB) [exactly equal to the net required last-mile-flow]
    # into end node vars should sum to 1 for each requests over all servers
    first_mile_vars = into_end_node_vars
    for req_idx in range(NUM_REQUESTS):
        tot_vars = gb.LinExpr() 
        for idx in range(NUM_SERVERS):
            curr_fm_vars = first_mile_vars[idx][req_idx]
            for var in curr_fm_vars:
                tot_vars.addTerms(1, var) 
        model.addConstr(tot_vars, GRB.LESS_EQUAL, UB)


    # [ Type-5 ] : Feasibility 
    print("Adding Type-5 constraints ...")
    for request_idx in range(NUM_REQUESTS):
        last_mile_flows = end_terminal_vars # or end_terminal_flows
        curr_constr_terms = []
        for server_idx in range(NUM_SERVERS):
            curr_constr_terms.append(last_mile_flows[server_idx][request_idx])
        curr_constr_terms.append(UB * infeasibility_vars[request_idx])
        # NOT ADDING THE ABOVE TERM <=> PUTTING A CONSTRAINT THAT INFEASIBLITY VARS MUST BE 0
        # THERE IS NO NEED OF INFEASIBLITY VARIABLES !!
        assert len(curr_constr_terms)==NUM_SERVERS+1
        model.addConstr(gb.quicksum(curr_constr_terms)==UB)
        num_total_constrs += 1

    vars_and_costs['server_rewards'] = server_rewards
    print("Constraints added!")
    print(f"TOTAL NUMBER OF *CONSTRAINTS* IN THE MODEL : {num_total_constrs}")
    logger.info(f"TOTAL NUMBER OF *CONSTRAINTS* IN THE MODEL : {num_total_constrs}")
    return vars_and_costs

# ----------------------------------------------------------

def optimize_model(model, model_name):
    model.Params.Threads = 16
    # model.Params.NodefileStart = 0.5
    model.optimize()
    status = model.status 
    assert status==GRB.OPTIMAL, "An optimal solution could not be found !"
    if city in ['A','B','C']:
        path = os.path.join(data_path, 'results/offline', f'{model_name}_{NUM_REQUESTS}_{NUM_SERVERS}_{t_hrs}_{weight_var}.sol')
    elif city=='X':
        path = os.path.join(data_path, 'results/offline', f'{model_name}_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.sol')
    elif city=='Q':
        path = os.path.join(data_path, 'results/offline', f'{model_name}_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.sol')
    # model.write()
    return model

# ----------------------------------------------------------

def k_server_general(begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var, only_last_mile, objective_type):
    ''' 
    Effective and Feasible solution:
        Effective: minimize the sum of ther server rewards 
        Feasible: either set a very high infeasibility penalty 
                        or add constraints that all infeasibility variables should be 0. (for this, make sure that there are enough number of servers) 
    '''
    ## Step-1: Instantiate gurobi Model
    print("INSTANTIATING FLOW-BASED MIP MODEL ...")
    model_name = "flow_mip"
    model = gb.Model("Flow-network based MIP model")
    # model.setParam('NonConvex', 2)

    ## Step-2: Add Variables 
    print("ADDING VARIABLES ...")
    vars_and_costs = construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var) 
    assert weight_var=='dist', 'Pruning with "dist" as cost is fine (equivalent to case with no pruning) BUT pruning should not be done with "time" as cost.' 

    ## Step-4: Add Constraints
    print("ADDING CONSTRAINTS ...")
    vars_and_costs = construct_and_set_constraints(model, vars_and_costs, only_last_mile) 

    ## Step-3 : Set Objective  
    print("SETTING MAXMIN OBJECTIVE ...")
    min_reward_var = vars_and_costs['min_reward_var']
    server_rewards = vars_and_costs['server_rewards']
    infeasibility_vars = vars_and_costs['infeasibility_vars']
    if weight_var=='dist':
        infeasibility_penalty_1 = (MAX_DELIVERY_DIST/NUM_SERVERS) 
        infeasibility_penalty_2 = (MEAN_DELIVERY_DIST/NUM_SERVERS)
        infeasibility_penalty = 1e20 * MAX_DELIVERY_DIST # universal
        # infeasibility_penalty = 0
    elif weight_var=='time':
        infeasibility_penalty_1 = (MAX_DELIVERY_TIME/NUM_SERVERS) 
        infeasibility_penalty_2 = (MEAN_DELIVERY_TIME/NUM_SERVERS)
        infeasibility_penalty = 1e20 * MAX_DELIVERY_TIME # universal


    def get_total_cost_lower_bound(weight_var, only_last_mile):
        ''' 
        Note that min_total_cost is not actually the minimum total cost for given instance of the problem (for that you'd have to solve the mip for the 'min' objective)
        rather it's a lower bound on the total_cost across all possible feasible instances; feasible=>availability of sufficient number of servers (i.e., assuming no infeasibility).  
        '''
        min_total_cost = 0
        max_total_cost = 0
        total_last_mile_cost = 0
        total_first_mile_cost = 0
        
        into_end_node_costs = vars_and_costs['into_end_node_costs']
        min_filtered_costs = [] 
        max_filetered_costs = []

        for request_costs in into_end_node_costs:
            flattened_request_costs = [x for x in request_costs] 
            try:
                curr_min_edge_cost = min(flattened_request_costs)
            except:
                curr_min_edge_cost = 0
            try:
                curr_max_edge_cost = max(flattened_request_costs)
            except:
                curr_max_edge_cost = 0 
            min_filtered_costs.append(curr_min_edge_cost) 
            max_filetered_costs.append(curr_max_edge_cost)

        min_first_mile_cost = sum(min_filtered_costs) # tight lower bound
        max_first_mile_cost = sum(max_filetered_costs) # loose lower bound

        if weight_var=='dist':
            total_last_mile_cost = np.sum(orders_data.deliver_dist.values) 
        elif weight_var=='time':
            total_last_mile_cost = np.sum(orders_data.deliver_time.values)

        if only_last_mile:
            min_total_cost = total_last_mile_cost 
            max_total_cost = total_last_mile_cost
        else:
            min_total_cost = total_last_mile_cost + min_first_mile_cost
            max_total_cost = total_last_mile_cost + max_first_mile_cost
        # return min_total_cost, max_total_cost 
        return total_last_mile_cost, min_total_cost, max_total_cost 


    tot_last, min_total_cost, max_total_cost = get_total_cost_lower_bound(weight_var, only_last_mile)
    print(f"MINIMUM POSSIBLE COST: {min_total_cost}")
    print(f"MAXIMUM POSSIBLE COST: {max_total_cost}")
    logger.info(f"MINIMUM POSSIBLE COST: {min_total_cost}")
    logger.info(f"MAXIMUM POSSIBLE COST: {max_total_cost}")
    
    objectives = {}
    objectives['maxmin'] = (min_reward_var - infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars))*(-1.0) # multiplied by -1.0 => MINIMIZATION 
    # objectives['maxmin'] = gb.quicksum(z for z in infeasibility_vars) # multiplied by -1.0 => MINIMIZATION 
    objectives['min'] = gb.quicksum(server_rewards) + infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars) 
    # objectives['multi'] = gb.quicksum(server_rewards) - (NUM_SERVERS * min_reward_var) + infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars)
    objectives['multi'] = 100 * gb.quicksum(server_rewards) - (NUM_SERVERS * min_reward_var) + 100 * gb.quicksum(z for z in infeasibility_vars)
    # in 'multi' objective, we try to minimize the net server_rewards AND maximize the min_server_reward
    
    if objective_type=='bound':
        # for 'bound', we use the maxmin objective only but we'll add one more constraint;
        # the idea is that the sum of maxmin server_rewards should be bounded by a factor of the 'min' possible sum(server_rewards)
        alpha = bound_factor
        # model.addConstr(gb.quicksum(server_rewards)*(1/UB)<=alpha*min_total_cost)
        model.addConstr(gb.quicksum(server_rewards)*(1/UB)<=alpha*tot_last)
        objective = objectives['maxmin']
    else:
        objective = objectives[objective_type]    

    # the 'multi' objective is *not* really a strict 'bound' objective 
    model.setObjective(objective, sense=GRB.MINIMIZE)

    ## Step-5: Solve 
    print("SOLVING ...")
    solved_model = optimize_model(model, model_name)
    
    return solved_model, vars_and_costs

# ----------------------------------------------------------

def load_solution(begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var, only_last_mile=True):
    # Step-1: Instantiate gurobi Model
    print("INSTANTIATING FLOW-BASED MIP MODEL ...")
    model_name = "flow_mip"
    model = gb.Model("Flow-network based MIP model")

    # Step-2: Add Variables 
    print("ADDING VARIABLES ...")
    # vars_and_costs = construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var)
    vars_and_costs = construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var)

    # Step-3: Reading the solved model    
    model.read(f'{model_name}.sol')

    return model, vars_and_costs

# ----------------------------------------------------------

def dict_key_inversion(original_dict):
    outer_keys = list(original_dict.keys()) # all_nodes
    inner_keys = list(original_dict[outer_keys[0]].keys()) # rest_nodes 

    inverted_dict = {}
    for inner_key in inner_keys:
        inverted_dict[inner_key] = {}
        for outer_key in outer_keys:
            inverted_dict[inner_key][outer_key] = original_dict[outer_key][inner_key] 

    new_outer_keys = list(inverted_dict.keys()) # rest_nodes
    new_inner_keys = list(inverted_dict[new_outer_keys[0]].keys()) # all_nodes
    assert len(inner_keys)==len(new_outer_keys)
    assert len(outer_keys)==len(new_inner_keys)
    return inverted_dict


def get_truncated_all_nodes_knn(dist_dict, k):
    ALL_NODES = set()

    # inverted_apsp_dist = dict_key_inversion(apsp_dist) 
    for rest in dist_dict.keys():
        rest_dict = dist_dict[rest]
        all_nodes = np.array(list(dist_dict[rest].keys()))
        node_dists = np.array(list(rest_dict.values()))
        # take k-nearest nodes from this and store in the set
        partitioned_indices = np.argpartition(node_dists, k) 
        k_nearest_indices = partitioned_indices[:k]
        k_nearest_nodes = all_nodes[k_nearest_indices]
        ALL_NODES.update(k_nearest_nodes)
    return ALL_NODES


def get_truncated_all_nodes_randomized(dist_dict, tot, k):
    ''' 
    Randomized k-nearest nodes : sample k nodes from tot nearest nodes
    Idea (derived from the "VideoCLIP" sampling procedure) : the sampled nodes should be *more* mutually closer to each other rather than just being closer to a single (restaurant) node)

    # randomized sampling of k-nearest nodes may reduce the number of incident flow vars on the end_nodes;
    # However, the rewards are decided by the end_terminal_costs which remain the same! 
    # So, by this randomization, WHEN only_last_mile==True, the final answers will not change at all except when some end_nodes becomes unreachable
    # It most probably will make a difference when only_last_mile==False!
    '''
    ALL_NODES = set()
    # inverted_apsp_dist = dict_key_inversion(apsp_dist) 
    for rest in dist_dict.keys():
        rest_dict = dist_dict[rest]
        all_nodes = np.array(list(dist_dict[rest].keys()))
        node_dists = np.array(list(rest_dict.values()))
        # take k-nearest nodes from this and store in the set
        partitioned_indices = np.argpartition(node_dists, tot) 
        tot_nearest_indices = partitioned_indices[:tot]
        tot_nearest_nodes = all_nodes[tot_nearest_indices]
        # randomly sample k-nearest nodes from tot_nearest_nodes
        k_nearest_nodes = set(random.sample(list(tot_nearest_nodes), k))
        k_nearest_nodes.add(rest) # rest is not mandatorily sampled in the previous step
        ALL_NODES.update(k_nearest_nodes)

    return ALL_NODES

# ----------------------------------------------------------


if __name__=='__main__':
    # get program inputs
    parser = argparse.ArgumentParser()
    
    ## common arguments
    parser.add_argument('--city', choices=['A', 'B', 'C'], default='A', 
                        type=str, required=True, help='City name')
    parser.add_argument('--day', choices=[x for x in range(10)], default=1, 
                        type=int, required=False, help='Day index') # not for 'synthetic'
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='dist',
                        type=str, required=False, help='Weight variable for road network')
    parser.add_argument('--objective', choices=['maxmin', 'min', 'multi', 'bound'], default='maxmin', 
                        type=str, required=False, help='Type of Objective function')
    parser.add_argument('--bound_factor', choices=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0, 10.0], default=2.0,
                        type=float, required=False, help='optimal solution\'s of "bound" objective must be within "bound_factor" times the minimum possible cost')
    parser.add_argument('--ub', choices=[0, 1], default=0,
                        type=int, required=False, help='upper bound on the flow of end_terminal_edges; 0=>Symmetry Optimization employed!')
    parser.add_argument('--init', choices=[0,1], default=0,
                        type=int, required=False, help='Consider the initial locations of the servers')
    parser.add_argument('--only_last_mile', choices=[0, 1], default=0, 
                        type=int, required=False, help='Consider only last-mile rewards (or costs)')
    parser.add_argument('--batch_size', choices=[1,2,3,4,5,6,7,8,9,10], default=1, 
                    type=int, required=False, help='Average batch-/cluster-size when "batching" is done') # currently batching is not supported
    
    ## food delivery specific args:
    parser.add_argument('--t', choices=[-1]+[x for x in range(25)], default=-1, 
                        type=int, required=False, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--start', choices=[-1]+[x for x in range(25)], default=0, 
                        type=int, required=False, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--end', choices=[-1]+[x for x in range(25)], default=8, 
                        type=int, required=False, help='Take orders that are placed upto t-hrs since 00:00:00')
    parser.add_argument('--knn', choices=range(-1,1000),default=100,
                        type=int, required=False, help='Take only the k-nearest nodes to restaurant nodes in the road network')
    parser.add_argument('--timestep', default=1,
                        type=int, required=False, help='The entire consists of "(86400//timestep)" units of time')
    parser.add_argument('--odd_dataset', choices=[1, 2], default=1,
                        type=int, required=False, help='whether to take odd timestamped data <=> slotted data with timestep=2 (choose 2 as input) AND distinct prep_tss !')
    
    ## only synthetic data specific args:
    parser.add_argument('--num_timesteps', choices=[100,200,500,1000,2000], default=500,
                        type=int, required=False, help='number of timesteps (refer the generated dataset)')
    parser.add_argument('--num_nodes', choices=[50,100,500,1000], default=500,
                        type=int, required=False, help='# nodes in metric space (refer the generated dataset)')
    parser.add_argument('--num_requests', choices=[10,30,50,100,250,500,1000],
                        type=int, required=False, help='# requests in the dataset (refer the generated dataset)')
    parser.add_argument('--num_servers', choices=[5,10,20,25,30,50,100,150,200,250,400,500],
                        type=int, required=False)
    parser.add_argument('--edge_prob', choices=[0.1,0.2,0.5,0.6,0.7,0.9], default=0.5, 
                        type=float, required=False, help='edge connection probability chosen while generating the dataset')
    

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
        objective = args.objective
        bound_factor = args.bound_factor
        knn = args.knn
        UB = args.ub
        timestep = args.timestep 
        ODD = args.odd_dataset
        INIT = args.init
        batch_size = args.batch_size 
      
        # metadata
        logs_path = os.path.join(data_path, city, 'logs')
        ensure_dir(logs_path)
        _intervals_path = os.path.join(data_path, city, f'drivers/{day}')
        ints_path = os.path.join(data_path, city, f'drivers/{day}/de_intervals')
        drivers_init_path = os.path.join(data_path, f'{city}/drivers/{day}/driver_init_nodes.csv')
        name2id = {'A':10, 'B':1, 'C':4}
        id2name = {v:k for k,v in name2id.items()}

        # LOAD DATA
        print("Loading data ...") 
        if timestep==1: orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders.csv')) 
        else: # this violates distinct prep_ts assumption
            orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_slotted_{timestep}.csv'))
        
        if ODD==2: orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/{day}/final_orders_odd_timestamps.csv'))
            
        apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_dist_t={t_hrs}.pkl'))
        apsp_time = depicklify(os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_time_t={t_hrs}.pkl'))   
        # the above 2 files only contain the shortest paths between the restaruant nodes active within the first t_hrs and all other nodes in the road network.           
        
        road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/{day}/u_v_time_dist'), header=None, sep=' ', names=['u', 'v', 'time', 'dist'])
        idx2node = pd.read_csv(os.path.join(data_path, f'{city}/map/index_to_node_id.csv'), header=None, names=['idx', 'node_id'])
        node2idx = pd.read_csv(os.path.join(data_path, f'{city}/map/node_id_to_index.csv'), header=None, names=['node_id', 'idx'])
        
        # consider only those orders placed until t_hrs since 00:00:00
        orig_orders_data = deepcopy(orders_data)
        orders_data = orders_data[(orders_data.placed_time>=start*3600) & (orders_data.placed_time<=(t_hrs-0.75)*3600)]
        
        # Create required variables and data 
        ALL_NODES = idx2node.idx.values # or np.arange(idx2node.shape[0])
        NUM_NODES = idx2node.shape[0]  # or len(ALL_NODES)


        # MODEL INPUTS
        begin_nodes, end_nodes, terminal_nodes, requests = parse_requests(orders_data)
        all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist=apsp_dist, apsp_time=apsp_time)
        # k-nearest nodes: # considering partial road-network for tractability --> this gives an approximation of the true MILP solution
        tot = 1000 
        print(f"Considering {knn}-nearest nodes to each restaurant node only!")
        print(f"Original NUM_NODES:{NUM_NODES}") 
        # ALL_NODES = get_truncated_all_nodes(apsp_time, knn)  
        ALL_NODES = get_truncated_all_nodes_randomized(apsp_time, tot, knn) 
        # 'apsp_time' makes more sense than 'apsp_dist' while generating ALL_NODES since edges incident on end_nodes are filetered for reachability using 'time'
        if knn!=-1:
            term_node_indices = set([node[0] for node in terminal_nodes])
            ALL_NODES = ALL_NODES.union(term_node_indices)
            NUM_NODES = len(ALL_NODES)
            print(f"New NUM_NODES:{len(ALL_NODES)}") 
        ALL_NODES = list(ALL_NODES)


        ###### INPUT SUMMARY BEGINS ###########
        NUM_REQUESTS = orders_data.shape[0] 
        total_num_rests = orig_orders_data['rest_node'].unique().shape[0] 
        curr_num_rests = orders_data['rest_node'].unique().shape[0] # number of active restaurants in chosen subset of data
        num_active_servers = get_active_servers(ints_path, t_hrs)
        _NUM_SERVERS = min(num_active_servers, NUM_REQUESTS) 
        # We don't want NUM_SERVERS>NUM_REQUESTS because fractional servers treatment can't work with such scenarios
        # as it might lead to no infeasibility (by dividing each server equally among all requests) even when it's a clear case of infeasibility!

        if UB==1: # maintain separate flows
            NUM_SERVERS = _NUM_SERVERS
        else: # symmetry optimization --> k=1 with edge-weight adjustment works because the optimal solution anyways has identical flows for all servers
            NUM_SERVERS = 1 
            UB = 1/(_NUM_SERVERS)

        if INIT: # better for real-data
            curr_rest_nodes = list(orders_data['rest_node'].unique())
            INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)

        min_deliver_timestep = np.min(orders_data['deliver_ts'].values)
        max_deliver_timestamp = np.max(orders_data['deliver_ts'].values)
        NUM_TIMESTAMPS = max_deliver_timestamp # time-step = 1 second # 1 day = 86400 seconds
        MAX_FPT = np.max(orders_data.prep_time.values)
        MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
        MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
        MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
        MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

        print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
        print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
        print(f"Number of requests (or orders): {NUM_REQUESTS}")
        print(f"Number of servers (or drivers): {_NUM_SERVERS}")
        # print(f"Total number of nodes in the LP or Flow network: {NUM_VARS}")
        # There exist 97282 edges in the road network
        
        # GET LOGGER:
        global logger
        log_filename = ''
        if INIT: log_filename = f"{day}_INIT_{t_hrs}_{NUM_REQUESTS}_{_NUM_SERVERS}_{weight_var}.log" 
        elif ODD==2: log_filename = f"{day}_ODD_{t_hrs}_{NUM_REQUESTS}_{_NUM_SERVERS}_{weight_var}.log" 
        elif timestep>1: log_filename = f"{day}_SLOT_{t_hrs}_{NUM_REQUESTS}_{_NUM_SERVERS}_{weight_var}.log"
        else: log_filename = f"{day}_{t_hrs}_{NUM_REQUESTS}_{_NUM_SERVERS}_{weight_var}.log"
        logger = get_logger(logs_path, log_filename) 

        logger.info(f"NUM_NODES : {NUM_NODES}")
        logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
        logger.info(f"NUM_SERVERS : {_NUM_SERVERS}")
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
        objective = args.objective
        bound_factor = args.bound_factor
        UB = args.ub
        INIT = args.init
        batch_size = args.batch_size 
        if INIT==1: UB =  1 
        ODD = 1
        timestep = 1

        # metadata
        logs_path = os.path.join(data_path, city, 'logs')
        ensure_dir(logs_path)
        drivers_init_path = os.path.join(data_path, f'{city}/drivers/init_nodes_{NUM_REQUESTS}_{NUM_NODES}.csv')

        # LOAD DATA
        print("Loading data ...") 
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/orders_{NUM_REQUESTS}_{NUM_NODES}_{NUM_TIMESTEPS}.csv'))
        apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/apsp_dist_{NUM_NODES}_p{edge_prob}.pkl'))
        apsp_time = depicklify(os.path.join(data_path, f'{city}/map/apsp_time_{NUM_NODES}_p{edge_prob}.pkl'))
        road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/metric_space_{NUM_NODES}_p{edge_prob}.csv'))

        ALL_NODES = range(1, NUM_NODES+1)

        # MODEL INPUTS 
        begin_nodes, end_nodes, terminal_nodes, requests = parse_requests(orders_data)
        all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist, apsp_time)
        _NUM_SERVERS = min(NUM_SERVERS, NUM_REQUESTS)
        
        if UB==1: NUM_SERVERS = _NUM_SERVERS
        else:
            NUM_SERVERS = 1
            UB = 1/(_NUM_SERVERS)
        
        if INIT: 
            curr_rest_nodes = list(orders_data['rest_node'].unique())
            INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)

        max_deliver_timestamp = np.max(orders_data['deliver_ts'].values)
        NUM_TIMESTAMPS = max_deliver_timestamp  # time-step = 1 second # 1 day = 86400 seconds
        MAX_FPT = np.max(orders_data.prep_time.values)
        MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
        MEAN_DELIVERY_DIST = np.mean(orders_data.deliver_dist.values)
        MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
        MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

        ###### INPUT SUMMARY BEGINS ###########
        print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
        print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
        print(f"Number of requests (or orders): {NUM_REQUESTS}")
        print(f"Number of servers (or drivers): {_NUM_SERVERS}")

        # GET LOGGER:
        global logger
        log_filename = ''
        if INIT: log_filename = f"INIT_{NUM_REQUESTS}_{_NUM_SERVERS}_{NUM_NODES}_{weight_var}.log" 
        else: log_filename = f"{NUM_REQUESTS}_{_NUM_SERVERS}_{NUM_NODES}_{weight_var}.log"
        logger = get_logger(logs_path, log_filename) 

        logger.info(f"NUM_NODES : {NUM_NODES}")
        logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
        logger.info(f"NUM_SERVERS : {_NUM_SERVERS}")
        logger.info(f"NUM_TIMESTAMPS : {NUM_TIMESTAMPS}")
        ########## INPUT SUMMARY ENDS ################# 


    elif city=='Q':
        '''
        QUICK COMMERCE DATASET
        '''
        day = args.day
        NUM_TIMESTAMPS = 5000 
        weight_var = args.weight_var
        only_last_mile = args.only_last_mile
        objective = args.objective
        bound_factor = args.bound_factor
        UB = args.ub
        INIT = args.init
        GAMMA = 1
        ODD = 1
        timestep = 1

        # metadata
        data_path = './data/'
        logs_path = os.path.join(data_path, city, 'logs')
        ensure_dir(logs_path)

        # LOAD DATA
        print("Loading data ...") 
        orders_data = pd.read_csv(os.path.join(data_path, f'{city}/orders/orders_{day}.csv'))
        nodes = pd.read_csv(os.path.join(data_path, f'{city}/map/_nodes.csv'))
        rests = pd.read_csv(os.path.join(data_path, f'{city}/orders/rests.csv'))
        requestNodes = rests.rest_node.values
        apsp = depicklify(os.path.join(data_path, f'{city}/map/apsp.pkl'))
        paths = depicklify(os.path.join(data_path, f"{city}/map/paths.pkl"))
        NUM_NODES = nodes.shape[0]
        NUM_RESTS = rests.shape[0]
        NUM_REQUESTS = orders_data.shape[0]
        ALL_NODES = range(1,NUM_NODES+1)

        NUM_SERVERS = 200 
        INIT_NODES = ['src']*NUM_SERVERS

        # MODEL INPUTS 
        begin_nodes, end_nodes, terminal_nodes, requests = parse_requests(orders_data)
        all_pairs_shortest_paths = apsp # parse_all_pair_shortest_paths(apsp_dist, apsp_time)
        _NUM_SERVERS = min(NUM_SERVERS, NUM_REQUESTS)
        
        if UB==1: NUM_SERVERS = _NUM_SERVERS
        else:
            NUM_SERVERS = 1
            UB = 1/(_NUM_SERVERS)
        
        if INIT: 
            curr_rest_nodes = list(orders_data['rest_node'].unique())
            INIT_NODES = random.choices(curr_rest_nodes, k=NUM_SERVERS)

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
        else: log_filename = f"{objective}{only_last_mile}_online_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.log"
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
    print("Optimization started ...")
    model, vars_and_costs = k_server_general(begin_nodes, end_nodes, terminal_nodes, 
                                            all_pairs_shortest_paths, weight_var, only_last_mile, objective)
    solve_end_time = time.time() 
    solve_time = solve_end_time - solve_start_time
    print(f"Execution time : {solve_time/3600} hrs")
    logger.info(f"Execution time : {solve_time/3600} hrs")

    # Evaluation:
    num_inf, request_feasibilities = calculate_infeasibility(vars_and_costs)
    unserved_percentage = (num_inf*100)/NUM_REQUESTS    
    print(f"{num_inf} out of {NUM_REQUESTS} requests ({unserved_percentage}%) remain unserved!")
    logger.info(f"{num_inf} out of {NUM_REQUESTS} requests ({unserved_percentage}%) remain unserved!")
    logger.info(f"Request feasibilities : {request_feasibilities}")

    print()
    server_rewards = get_server_rewards(vars_and_costs, NUM_SERVERS, NUM_REQUESTS, ALL_NODES, only_last_mile)
    # if server_rewards==1 : server_rewards[0] is the sum of the server_rewards of all _NUM_SERVERS 
    for s_idx, reward in enumerate(server_rewards):
        print(f"Reward of server #{s_idx}: {reward}")
        logger.info(f"Reward of server #{s_idx}: {reward}")

    optimal_cost = get_optimal_cost(model) 
    print(f"Optimal cost : {optimal_cost}")
    logger.info(f"Optimal cost : {optimal_cost}")

    gini = gini_index(server_rewards)
    print(f"Gini index: {gini}")
    logger.info(f"Gini index: {gini}")

    avg_dist = get_avg_distance(server_rewards)
    print(f"Avg. Dist. : {avg_dist}")
    logger.info(f"Avg. Dist. : {avg_dist}")

