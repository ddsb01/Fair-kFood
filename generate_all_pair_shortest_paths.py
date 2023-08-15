import warnings 
warnings.filterwarnings('ignore')

import os
import sys
import copy
import time
import pickle 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pprint import pprint 


"""
def get_all_pair_shortest_path(road_net, weight_var='dist'):
    print("Creating graph and adding edges ...")
    graph = nx.Graph() 
    for idx, edge in tqdm(road_net.iterrows(), total=len(road_net)):
        graph.add_edge(edge.u, edge.v, dist=edge.dist, time=edge.time)

    # calculate all-pair shortest paths 
    print("Computing all-pairs shortest paths ...")
    print("This might take a while.")
    all_pair_shortest_paths = dict(nx.floyd_warshall(graph, weight=weight_var))
    
    return all_pair_shortest_paths
"""

def WRONG_get_all_pair_shortest_path(road_net, orders_df, weight_var, t_hrs=-1):
    ''' 
    Assumption:
        All road segments are 2-way. 
        This isn't true for many roads! (see ./data/../original/map_orig/"segments_mode_connected.csv" ; "edges")
    '''
    print("Creating graph and adding edges ...")
    graph = nx.Graph() 
    for idx, edge in tqdm(road_net.iterrows(), total=len(road_net)):
        graph.add_edge(edge.u, edge.v, dist=edge.dist, time=edge.time)

    shortest_paths = [] # unit=metres (if weight_var='dist')
               # unit=seconds (if weight_var='time') 
    
    if t_hrs!=-1:
        # Calculate all pairs shortest paths only for nodes related to orders (rest_nodes & cust_nodes) placed upto the t-th hour
        orders_df = orders_df[orders_df.placed_time <= t_hrs * 3600]
        rest_nodes = np.unique(orders_df.rest_node.values) 
        # cust_nodes = orders_df.cust_node.values 
        # all_nodes = np.concatenate((rest_nodes, cust_nodes))
        all_nodes = np.arange(NUM_NODES)

        shortest_paths = {
                src_idx: \
                    {dest_idx:-1 for dest_idx in rest_nodes} \
                    for src_idx in all_nodes
                }

        print(f"Number of restaurant nodes: {rest_nodes.shape[0]}")
        # THIS IS UTTER FOOLISHNESS!!
        # Here, TC: O(num_all_nodes * rest_nodes * shortest_path_calculation) 
        # Optimal (see new function) TC: O(rest_nodes * shortest_path_calculation)
        for src_idx in tqdm(all_nodes, total=len(all_nodes)):
            for dest_idx in rest_nodes:
                if shortest_paths[src_idx][dest_idx]==-1:
                    curr_length = nx.shortest_path_length(graph, src_idx, dest_idx, weight=weight_var)
                    shortest_paths[src_idx][dest_idx] = curr_length
                    shortest_paths[dest_idx][src_idx] = curr_length # since graph is undirected (see Assumption)
    
    else:
        # Calculate all pair shortest paths (for all nodes in the road network):
        ###### Optimization : Change the below code so that all pair shortest paths are only computed between rest_nodes and all_nodes
        print(f"Number of nodes: {road_net.shape[0]}")
        print("Computing all-pairs shortest paths ...")
        print("This might take a while.")
        shortest_paths = dict(nx.floyd_warshall(graph, weight=weight_var))
    

    return shortest_paths


def get_mst(edges, dists, times):
    def get_apsp(nodes, edges, dists, times):
        print("Computing all pair shortest paths ...")
        def parse_shortest_paths_output(paths_dict):
            new_paths_dict = {node_idx:dict(value) 
                            for node_idx, value in paths_dict.items()}
            return new_paths_dict
        
        graph = nx.Graph() 
        for idx, (edge, dist, time) in enumerate(zip(edges, dists, times)):
            graph.add_edge(edge[0], edge[1], dist=dist, time=time) 

        print("\tweighted by distance ...")
        apsp_dist = dict(nx.floyd_warshall(graph, weight='dist'))
        apsp_dist = parse_shortest_paths_output(apsp_dist) 
        print("\tweighted by time ...")
        apsp_time = dict(nx.floyd_warshall(graph, weight='time'))
        apsp_time = parse_shortest_paths_output(apsp_time) 

        # all-pair-shortest-path lists 
        paths_dist = {}
        paths_time = {}
        for src in tqdm(nodes, total=len(nodes)):
            paths_dist[src] = nx.shortest_path(G=graph, source=src, weight='dist')
            paths_time[src] = nx.shortest_path(G=graph, source=src, weight='time')

        return apsp_dist, apsp_time, paths_dist, paths_time
    
    def get_graph_df(edges, dists, times):
        print("Creating graph dataframe ...")
        assert len(edges)==len(dists), 'edges and dists have unequal dimesions!'
        assert len(edges)==len(times), 'edges and dists have unequal dimesions!'
        df = pd.DataFrame(columns={'u', 'v', 'dist', 'time'})
        us, vs = np.array([edge[0] for edge in edges]), np.array([edge[1] for edge in edges]) 
        dists, times = np.array(dists), np.array(times)
        # looping through edges would be really slow! Therefore,
        df['u'] = us
        df['v'] = vs 
        df['dist'] = dists
        df['time'] = times
        # ordering:
        df = df[['u', 'v', 'dist', 'time']]
        return df 
    

    print("Creating MST ...")
    graph = nx.Graph() 
    for edge, d, t in zip(edges, dists, times):
        graph.add_edge(edge[0], edge[1], dist=d, time=t)
    
    mst_graph = nx.minimum_spanning_tree(graph, weight='time')
    mst_edges = list(mst_graph.edges())
    mst_dists, mst_times = [], []
    for (u, v) in mst_edges:
        mst_dists.append(mst_graph[u][v]['dist'])
        mst_times.append(mst_graph[u][v]['time'])
    
    mst_nodes = set()
    for (u, v) in mst_edges:
        mst_nodes.add(u)
        mst_nodes.add(v) 
    mst_nodes = list(mst_nodes)

    mst_df = get_graph_df(mst_edges, mst_dists, mst_times)
    mst_apsp_time, mst_paths_time = get_apsp(mst_nodes, mst_edges, mst_dists, mst_times)
    return mst_df, mst_apsp_dist, mst_apsp_time, mst_paths_dist, mst_paths_time


def get_all_pair_shortest_path(road_net, orders_df, weight_var, t_hrs=-1):
    ''' 
    Assumption:
        All road segments are 2-way. 
        This isn't true for many roads! (see ./data/../original/map_orig/"segments_mode_connected.csv" ; "edges")
    '''
    print("Creating graph and adding edges ...")
    graph = nx.Graph() 
    for idx, edge in tqdm(road_net.iterrows(), total=len(road_net)):
        graph.add_edge(edge.u, edge.v, dist=edge.dist, time=edge.time)

    # shortest_paths = [] # unit=metres (if weight_var='dist')
                        # unit=seconds (if weight_var='time') 
    
    if t_hrs!=-1:
        # Calculate all pairs shortest paths only for nodes related to orders (rest_nodes & cust_nodes) placed upto the t-th hour
        orders_df = orders_df[orders_df.placed_time <= t_hrs*3600]
        rest_nodes = np.unique(orders_df.rest_node.values) 
        # cust_nodes = orders_df.cust_node.values 
        # all_nodes = np.concatenate((rest_nodes, cust_nodes))
        all_nodes = np.arange(NUM_NODES)

        shortest_path_lengths = {src_idx:None for src_idx in rest_nodes}
        shortest_paths = {src_idx:None for src_idx in rest_nodes}

        print(f"Number of restaurant nodes: {rest_nodes.shape[0]}")
        for src_idx in tqdm(rest_nodes, total=len(rest_nodes)):
            shortest_path_lengths[src_idx] = nx.shortest_path_length(G=graph, source=src_idx, weight=weight_var) # note that 'target' is missing => return single-source-shortest-path-lengths dict
            shortest_paths[src_idx] = nx.shortest_path(G=graph, source=src_idx, weight=weight_var) # note that 'target' is missing => return single-source-shortest-paths (in lists) dict
            # breakpoint()
        return shortest_path_lengths, shortest_paths
    else:
        # Calculate all pair shortest paths (for all nodes in the road network):
        ###### Optimization : Change the below code so that all pair shortest paths are only computed between rest_nodes and all_nodes
        print(f"Number of nodes: {road_net.shape[0]}")
        print("Computing all-pairs shortest paths ...")
        print("This might take a while.")
        shortest_paths = dict(nx.floyd_warshall(graph, weight=weight_var))
        return shortest_paths


def parse_shortest_paths_output(paths_dict):
    new_paths_dict = {node_idx:dict(value) 
                      for node_idx, value in paths_dict.items()}
    return new_paths_dict


def picklify(data_struct, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return


def depicklify(filename):
    with open(filename, 'rb') as f:
        data_struct = pickle.load(f)
    return data_struct




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['A', 'B', 'C'], default='A', 
                        type=str, required=True, help='City name')
    parser.add_argument('--day', choices=range(1,7), default=1,
                        type=int, required=True, help='Day number')
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='time',
                        type=str, required=True, help='Weight variable for road network')
    parser.add_argument('--t', choices=[-1]+[x for x in range(1,25)], default=-1, 
                        type=int, required=True, help='Take orders that are placed upto t-hrs since 00:00:00')
    args = parser.parse_args()


    city = args.city             #'A' # city \in {'A', 'B', 'C'}
    day = args.day
    weight_var = args.weight_var # 'dist' # 'dist' or 'time'
    t_hrs = args.t                  


    # load data
    print("Loading data ...")
    data_path = '/home/daman/Desktop/k_server/code/data/'
    road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/{day}/u_v_time_dist'), 
                        header=None, sep=' ', names=['u', 'v', 'time', 'dist'])
    orders_data = pd.read_csv(os.path.join(data_path, 
                        f'{city}/orders/{day}/orders.csv'), sep=' ').rename(columns={'node_id':'cust_node', 'restaurant_id':'rest_id'})
    idx2node = pd.read_csv(os.path.join(data_path, 
                        f'{city}/map/index_to_node_id.csv'), header=None, names=['idx', 'node_id'])
    NUM_NODES = idx2node.shape[0]

    # computer all-pairs shortest paths
    # print("Computing all-pairs shortest paths ...")
    start_time = time.time()
    if t_hrs!=-1: apsp_lengths, aps_paths = get_all_pair_shortest_path(road_net, orders_data, weight_var, t_hrs)
    else: apsp_lengths = get_all_pair_shortest_path(road_net, orders_data, weight_var, t_hrs)
    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"All pairs shortest paths computed in {total_time_taken/3600} minutes.")

    if t_hrs==-1: apsp_lengths = parse_shortest_paths_output(apsp_lengths)

    # save all-pairs shortest paths 
    # filename = os.path.join(data_path, f'{city}/map/all_pair_shortest_paths_{weight_var}_t={t_hrs}.pkl')
    filename1 = os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_{weight_var}_t={t_hrs}.pkl')
    filename2 = os.path.join(data_path, f'{city}/map/{day}/apsp/all_pair_shortest_paths_lists_{weight_var}_t={t_hrs}.pkl')
    
    picklify(apsp_lengths, filename1)
    if t_hrs!=-1: picklify(aps_paths, filename2)


    # # MST generation:
    # edges, dists, times = [], [], []
    # for idx, edge in road_net.iterrows():
    #     edges.append((edge.u, edge.v))
    #     dists.append(edge.dist)
    #     times.append(edge.time)
    
    # mst_df, mst_apsp_dist, mst_apsp_time, mst_paths_dist, mst_paths_time = get_mst(edges, dists, times)
    # mst_df.to_csv(os.path.join(data_path, f'{city}/map/mst_df_t={t_hrs}.csv'))
    # apsp_dist_path = os.path.join(data_path, f'{city}/map/{day}/apsp/mst_apsp_dist_t={t_hrs}.pkl')
    # apsp_time_path = os.path.join(data_path, f'{city}/map/{day}/apsp/mst_apsp_time_t={t_hrs}.pkl')
    # paths_dist_path = os.path.join(data_path, f'{city}/map/{day}/apsp/mst_paths_lists_dist_t={t_hrs}.pkl')
    # paths_time_path = os.path.join(data_path, f'{city}/map/{day}/apsp/mst_paths_lists_time_t={t_hrs}.pkl')
    
    # picklify(mst_apsp_dist, apsp_time_path)
    # picklify(mst_apsp_time, apsp_time_path)
    # picklify(mst_paths_dist, paths_dist_path)
    # picklify(mst_paths_time, paths_time_path)
    
    # '''
    # load and verify 
    # all_pair_shortest_paths = depicklify(filename1)
    # pprint(all_pair_shortest_paths)
    # '''

