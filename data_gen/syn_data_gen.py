import warnings 
warnings.filterwarnings("ignore")

import os 
import random
import argparse
import itertools
import numpy as np
import pandas as pd

import configparser 
import networkx as nx

from utils import * 

random.seed(1234567)


'''
Mean prep_time in real data is ~501 seconds and mean deliver_time is ~336 seconds which is too small as compared to a day's timespan
So, there is no way of choosing prep_time and deliver_time as a practical fraction of the entire # timesteps based on real-data
Therefore, we arbitrarily choose max prep_time and max deliver_time for synthetic data to be one-tenth of the total #timesteps 
'''

def generate_metric_space_and_save(num_nodes, edge_prob, speed, wt_type):
    ''' 
    Metric Space or Road Network generation
    
    speed: single value representing the constant speed (in m/s) in for all edges
    wt_type: distances can be 'random' or 'uniform'
    '''
    def generate_graph(num_nodes, edge_prob):
        # MODEL: Connected "Erdos-Renyi"
        print("Creating a connected graph ...")
        nodes = [(x+1) for x in range(num_nodes)]
        edges = []
        # # generate connected graph:
        degree = {v:0 for v in nodes}
        for u in tqdm(nodes, total=len(nodes)):
            while not degree[u] and u<num_nodes: # to ensure connectivity
                for v in range(u+1, num_nodes+1):
                    if u!=v and random.random()>=edge_prob:
                        edges.append((u, v))
                        degree[u] += 1
        return nodes, edges 
    
    def check_connectivity(edges):
        print("Verifying connectivity ...")
        graph = nx.Graph() 
        graph.add_edges_from(edges)
        assert nx.is_connected(graph), 'The generated graph is not connected!' 
        return 
    
    def generate_edge_weights(edges, wt_type):
        print("Generating edge weights ...")
        if wt_type=='random':
            # weights = [random.randint(speed, speed*1800) for x in edges]
            weights = [random.randint(10, 1000) for x in edges]
            # since speed is in m/s, speed*1800 is the distance travelled in 1800 seconds (30 mins) # heuristic
        elif wt_type=='uniform':
            weights = [1]*len(edges)
        return weights 
    
    def get_graph_df(edges, weights):
        print("Creating graph dataframe ...")
        assert len(edges)==len(weights), 'edges and weights have unequal dimesions!'
        df = pd.DataFrame(columns={'u', 'v', 'dist', 'time'})
        us = np.array([edge[0] for edge in edges])
        vs = np.array([edge[1] for edge in edges]) 
        wts = np.array(weights)
        # looping through edges would be really slow! Therefore,
        df['u'] = us
        df['v'] = vs 
        df['dist'] = wts
        df['time'] = (df['dist']/SPEED).apply(lambda x:int(x))
        # ordering:
        df = df[['u', 'v', 'dist', 'time']]
        return df 
    
    def get_all_pairs_shortest_paths(edges, dists):
        print("Computing all pair shortest paths ...")
        def parse_shortest_paths_output(paths_dict):
            new_paths_dict = {node_idx:dict(value) 
                            for node_idx, value in paths_dict.items()}
            return new_paths_dict
        
        graph = nx.Graph() 
        for idx, (edge, dist) in enumerate(zip(edges, dists)):
            graph.add_edge(edge[0], edge[1], dist=dist, time=dist//SPEED) 

        print("\tweighted by distance ...")
        apsp_dist = dict(nx.floyd_warshall(graph, weight='dist'))
        apsp_dist = parse_shortest_paths_output(apsp_dist) 
        print("\tweighted by time ...")
        apsp_time = dict(nx.floyd_warshall(graph, weight='time'))
        apsp_time = parse_shortest_paths_output(apsp_time) 

        # all-pair-shortest-path lists 
        paths_dist, paths_time = {}, {}
        for src in tqdm(nodes, total=len(nodes)):
            paths_dist[src] = nx.shortest_path(G=graph, source=src, weight='dist')
            paths_time[src] = nx.shortest_path(G=graph, source=src, weight='time')

        return apsp_dist, apsp_time, paths_dist, paths_time

    def get_mst(edges, weights):
        print("Creating MST ...")
        graph = nx.Graph() 
        for edge, wt in zip(edges, weights):
            graph.add_edge(edge[0], edge[1], dist=wt, time=wt//SPEED)
        
        mst_graph = nx.minimum_spanning_tree(graph, weight='time')
        mst_edges = list(mst_graph.edges())
        mst_weights = []
        for (u, v) in mst_edges:
            mst_weights.append(mst_graph[u][v]['dist'])
        return mst_edges, mst_weights

    print("Metric Space generation started ...")
    nodes, edges = generate_graph(num_nodes, edge_prob)
    check_connectivity(edges)
    weights = generate_edge_weights(edges, wt_type)
    network_df = get_graph_df(edges, weights)
    apsp_dist, apsp_time, paths_dist, paths_time = get_all_pairs_shortest_paths(edges, weights)
    print(f"Successfully generated a graph with {num_nodes} vertices and {len(edges)} edges.")
    
    return network_df, apsp_dist, apsp_time, paths_dist, paths_time


def generate_requests_data(num_reqs, num_ts, num_nodes):
    ''' 
    Requests or Orders data generation 
    Contains request placed times, preparation times, deliver times as well as corresponding timestamps.
    '''
    requests_df = pd.DataFrame(columns=['rest_node', 'cust_node', 'placed_ts', 'prep_ts', 'deliver_ts', 'prep_time', 'deliver_time', 'deliver_dist'])
    
    prep_limit = num_ts//10 
    deliver_limit = num_ts//10
    
    prep_timestamps = random.sample(range(prep_limit+1, num_ts-deliver_limit), num_reqs) # prep_ts variable is the most constrained (needs to be distinct for each request), so we resolve this first

    for it in range(num_reqs):
        # choosing space
        rest_node, cust_node = -1, -1
        rest_node = random.randint(1, num_nodes)
        while cust_node==rest_node or cust_node==-1:
            cust_node = random.randint(1, num_nodes)
        
        # choosing time
        prep_ts = prep_timestamps[it]
        placed_ts = random.randint(prep_ts-prep_limit, prep_ts-1)
        deliver_ts = random.randint(prep_ts+1, prep_ts+deliver_limit)
        prep_time = prep_ts - placed_ts
        deliver_time = deliver_ts - prep_ts
        deliver_dist = speed * deliver_time
        
        # verifying requirements
        assert rest_node!=cust_node
        assert num_ts>=100, 'Number of timesteps should be >= 100'
        assert placed_ts < prep_ts, f'{placed_ts} not < {prep_ts}'
        assert prep_ts < deliver_ts, f'{prep_ts} not < {deliver_ts}'

        # entering into dataframe
        curr_req = {'rest_node':rest_node, 
                    'cust_node':cust_node, 
                    'placed_ts':placed_ts, 
                    'prep_ts':prep_ts, 
                    'deliver_ts':deliver_ts, 
                    'prep_time':prep_time, 
                    'deliver_time':deliver_time,
                    'deliver_dist':deliver_dist}
        requests_df = requests_df.append(curr_req, ignore_index=True)
    
    requests_df = requests_df[['rest_node', 'cust_node', 'placed_ts', 'prep_ts', 'deliver_ts', 'prep_time', 'deliver_time', 'deliver_dist']] # ordering
    return requests_df


def generate_servers_data(num_servers, num_nodes):
    ''' 
    Servers or Drivers data generation
    Each server is assumed to be always active. 
    Contains initial locations or initial nodes information.
    '''
    init_locs = random.choices(list(range(num_nodes)), k=num_servers)
    init_locs_df = pd.DataFrame()
    init_locs_df['de_id'] = list(range(num_servers))
    init_locs_df['node'] = init_locs
    return init_locs_df



if __name__=='__main__':    
    # Program inputs    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--edge_prob', type=float, default=0.5, required=True)
    parser.add_argument('--speed', type=int, default=10, required=True)
    parser.add_argument('--wt_type', choices=['random', 'uniform'], type=str, required=True) 
    parser.add_argument('--num_requests', type=int, required=True)
    parser.add_argument('--num_timesteps', type=int, required=True)
    parser.add_argument('--num_servers', type=int, required=True)
    args = parser.parse_args() 
    
    num_nodes = args.num_nodes 
    edge_prob = args.edge_prob 
    speed = args.speed 
    wt_type = args.wt_type
    num_reqs = args.num_requests
    num_ts = args.num_timesteps
    num_servers = args.num_servers
    SPEED = 10
    

    # Data paths 
    syn_datapath = './data/X'
    ms_filepath = os.path.join(syn_datapath, f'map/metric_space_{num_nodes}_p{edge_prob}.csv')
    req_filepath = os.path.join(syn_datapath, f'orders/orders_{num_reqs}_{num_nodes}_{num_ts}.csv')
    ser_filepath = os.path.join(syn_datapath, f'drivers/init_nodes_{num_reqs}_{num_nodes}.csv')
    apsp_dist_path = os.path.join(syn_datapath, f'map/apsp_dist_{num_nodes}_p{edge_prob}.pkl')
    apsp_time_path = os.path.join(syn_datapath, f'map/apsp_time_{num_nodes}_p{edge_prob}.pkl')
    paths_dist_path = os.path.join(syn_datapath, f'map/apsp_dist_lists_{num_nodes}_p{edge_prob}.pkl')
    paths_time_path = os.path.join(syn_datapath, f'map/apsp_time_lists_{num_nodes}_p{edge_prob}.pkl')
    
    # Metric space
    net_df, apsp_dist, apsp_time, paths_dist, paths_time, mst_df, mst_apsp_dist, mst_apsp_time, mst_paths_dist, mst_paths_time = generate_metric_space_and_save(num_nodes, edge_prob, speed, wt_type)
    net_df.to_csv(ms_filepath, index=False)
    print(f"Generated network stored successfully at {ms_filepath}")
    picklify(apsp_dist, apsp_dist_path)
    picklify(apsp_time, apsp_time_path) 
    picklify(paths_dist, paths_dist_path)
    picklify(paths_time, paths_time_path)
    
    # Requests 
    requests_df = generate_requests_data(num_reqs, num_ts, num_nodes) 
    requests_df.to_csv(req_filepath, index=False)
    print(f"Generated requests data stored successfully at {req_filepath}")

    # Servers
    init_locs = generate_servers_data(num_servers, num_nodes)
    init_locs.to_csv(ser_filepath, index=False) 
    print(f"Generated servers data stored successfully at {ser_filepath}")