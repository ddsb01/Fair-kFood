import warnings 
warnings.filterwarnings('ignore')

import os 
import sys 
import copy 
import dill 
import lzma
import pickle 
import matplotlib.pyplot as plt 

import networkx as nx

import gurobipy as gb
from gurobipy import GRB 

from typing import List

def get_server_rewards(vars_and_costs, NUM_SERVERS, NUM_REQUESTS, ALL_NODES, only_last_mile=True):
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
        min_reward_var,
        server_rewards_linexprs
    ) = vars_and_costs.values()

    server_rewards = []
    if server_rewards_linexprs != []:
        for linexpr in server_rewards_linexprs:
            curr_reward = linexpr.getValue() 
            server_rewards.append(curr_reward)
        return server_rewards 

    for server_idx in range(NUM_SERVERS):
        first_mile_reward = 0.0 
        last_mile_reward = 0.0

        first_mile_vars = into_end_node_vars[server_idx] 
        last_mile_vars = end_terminal_vars[server_idx] 
        # first_mile_costs = into_end_node_costs[server_idx]
        # last_mile_costs = end_terminal_costs[server_idx]  
        first_mile_costs = into_end_node_costs
        last_mile_costs = end_terminal_costs

        # if not only_last_mile:
        #     for request_idx in range(NUM_REQUESTS):
        #         curr_request_vars = first_mile_vars[request_idx]
        #         curr_request_costs = first_mile_costs[request_idx]
        #         FPT = len(curr_request_vars)
        #         assert len(curr_request_vars)==len(curr_request_costs)

        #         for ts in range(FPT):
        #             curr_ts_vars = curr_request_vars[ts]
        #             curr_ts_costs = curr_request_costs[ts]
        #             num_reachable_nodes = len(curr_ts_vars)
        #             for node_idx in range(num_reachable_nodes):
        #                 flow_var = curr_ts_vars[node_idx] 
        #                 reward_value = curr_ts_costs[node_idx]
        #                 first_mile_reward += flow_var.X * reward_value
        
        # for "pruned" cases:
        if not only_last_mile:
            for request_idx in range(NUM_REQUESTS):
                curr_fm_vars = first_mile_vars[request_idx]
                curr_fm_costs = first_mile_costs[request_idx]
                assert len(curr_fm_vars)==len(curr_fm_costs)
                assert len(curr_fm_vars)<=len(ALL_NODES)
                
                # -----
                # for flow_var, reward_value in zip(curr_fm_vars, curr_fm_costs):
                #     first_mile_reward.addTerms(reward_value, flow_var)
                # # this is wrong because all the sum of input flows can be greater than last-mile flow; the extra flow should not be rewarded
                # -----

                # Proportionally adding the first-mile rewards (assuming all input flows contribute equally to the )
                curr_lm_var = last_mile_vars[request_idx]
                in_flow_sum = gb.quicksum(var for var in curr_fm_vars)
                for flow_var, reward_value in zip(curr_fm_vars, curr_fm_costs):
                    prop = flow_var/in_flow_sum
                    first_mile_reward.addTerms(reward_value, prop*curr_lm_var)

        for flow_var, reward_value in zip(last_mile_vars, last_mile_costs):
            last_mile_reward += (flow_var.X * reward_value)

        if only_last_mile:
            server_reward = last_mile_reward
        else:
            server_reward = first_mile_reward + last_mile_reward 
        server_rewards.append(server_reward)
    
    return server_rewards


def get_optimal_cost(model): 
    obj_var = model.getObjective() 
    obj_value = obj_var.getValue() 
    return obj_value


def calculate_infeasibility(vars_and_costs):
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
        min_reward_var,
        server_rewards
    ) = vars_and_costs.values() 

    # num_total_orders = len(infeasibility_vars) 
    num_inf = 0 
    request_feasibilities = []
    for req_idx, inf_var in enumerate(infeasibility_vars):
        # print(f"Infeasibility of request {req_idx} : {inf_var.X}")
        num_inf += inf_var.X 
        request_feasibilities.append(inf_var.X)
    return num_inf, request_feasibilities


def gini_index(incomes:List):
    # refer: https://en.wikipedia.org/w/index.php?title=Gini_coefficient&action=edit&section=5
    incomes = copy.deepcopy(incomes)
    incomes.sort()
    n = len(incomes)
    Nr = sum([(idx+1) * val for idx, val in enumerate(incomes)])
    Dr = sum(incomes) 
    gini = ((2/n) * (Nr/Dr)) - ((n+1)/n)
    return gini


def get_avg_distance(incomes:List):
    avg_inc = sum(incomes)/len(incomes)
    return avg_inc



if __name__=='__main__':
    model = gb.Model()
    # new_create_and_set_variables()
    model.read('flow_mip.sol')
    for v in model.getVars():
        print(f"{v.VarName} : {v.X}")
        break

